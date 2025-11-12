from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from db import Base, engine, SessionLocal
from models import Document, Chunk, Entity, EntityChunk
from schemas import IngestRequest, BulkIngestRequest, IngestResult, SearchResponseItem
from ingest_utils import detect_lang_optional
from text_utils import chunk_text
from crud import ingest_document, add_chunks_with_embeddings, extract_and_link_entities
from embed import embed_texts

app = FastAPI(title="OKC API")

def get_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except:
        db.rollback()
        raise
    finally:
        db.close()

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)

@app.post("/ingest", response_model=IngestResult)
def ingest(payload: IngestRequest, db: Session = Depends(get_db)):
    lang = payload.lang or detect_lang_optional(payload.text) or "en"
    doc_id, deduped = ingest_document(db, payload.title, payload.url, payload.text, lang=lang)
    if deduped:
        return IngestResult(document_id=None, deduped=True, num_chunks=0, title=payload.title, url=payload.url)

    chunks = chunk_text(payload.text, target_tokens=600, overlap=80)
    chunk_ids = add_chunks_with_embeddings(db, doc_id, chunks)
    for cid, text in zip(chunk_ids, chunks):
        extract_and_link_entities(db, cid, text)

    return IngestResult(document_id=doc_id, deduped=False, num_chunks=len(chunks), title=payload.title, url=payload.url)

@app.post("/ingest_bulk", response_model=list[IngestResult])
def ingest_bulk(payload: BulkIngestRequest, db: Session = Depends(get_db)):
    results: list[IngestResult] = []
    # Stage 1: normalize/dedupe + stage chunks for embedding in one batch per doc
    staged = []  # [(doc_id, chunks, title, url, deduped)]
    for item in payload.items:
        lang = item.lang or detect_lang_optional(item.text) or "en"
        doc_id, deduped = ingest_document(db, item.title, item.url, item.text, lang=lang)
        if deduped:
            results.append(IngestResult(document_id=None, deduped=True, num_chunks=0, title=item.title, url=item.url))
            continue
        chunks = chunk_text(item.text, target_tokens=600, overlap=80)
        staged.append((doc_id, chunks, item.title, item.url))

    # Stage 2: embed + write chunks, then entity pass per doc
    # (We embed per doc to keep memory bounded; switch to global batch if needed.)
    for doc_id, chunks, title, url in staged:
        cids = add_chunks_with_embeddings(db, doc_id, chunks)
        for cid, text in zip(cids, chunks):
            extract_and_link_entities(db, cid, text)
        results.append(IngestResult(document_id=doc_id, deduped=False, num_chunks=len(chunks), title=title, url=url))

    return results

@app.get("/search", response_model=list[SearchResponseItem])
def semantic_search(q: str = Query(..., min_length=2), k: int = 10, db: Session = Depends(get_db)):
    qvec = embed_texts([q])[0]
    # cosine similarity since vectors are normalized
    sim = func.cosine_similarity(Chunk.embedding, qvec.tolist())  # pgvector >= 0.7 has cosine_similarity
    stmt = (
        select(Chunk.id, Chunk.document_id, Document.title, Chunk.text, sim.label("score"))
        .join(Document, Document.id == Chunk.document_id)
        .order_by(func.cosine_distance(Chunk.embedding, qvec.tolist()))  # smaller is better
        .limit(k)
    )
    rows = db.execute(stmt).all()
    out = []
    for cid, did, title, text, score in rows:
        snippet = text[:280].replace("\n", " ")
        out.append(SearchResponseItem(chunk_id=cid, document_id=did, title=title, snippet=snippet + ("…" if len(text) > 280 else ""), score=float(score or 0.0)))
    return out

@app.get("/entity/{entity_id}")
def get_entity(entity_id: int, db: Session = Depends(get_db)):
    ent = db.get(Entity, entity_id)
    if not ent:
        raise HTTPException(404, "entity not found")
    # naive co-mention neighbors: entities that appear in same chunks
    subq = select(EntityChunk.chunk_id).where(EntityChunk.entity_id == entity_id).subquery()
    neighbor_stmt = (
        select(Entity.id, Entity.name, func.count().label("co_count"))
        .join(EntityChunk, EntityChunk.entity_id == Entity.id)
        .where(EntityChunk.chunk_id.in_(select(subq.c.chunk_id)))
        .where(Entity.id != entity_id)
        .group_by(Entity.id, Entity.name)
        .order_by(func.count().desc())
        .limit(20)
    )
    neighbors = [{"id": i, "name": n, "weight": int(c)} for i,n,c in db.execute(neighbor_stmt).all()]
    return {
        "id": ent.id, "name": ent.name, "canonical_label": ent.canonical_label,
        "neighbors": neighbors
    }

@app.get("/graph/entity/{entity_id}")
def graph_neighbors(entity_id: int, db: Session = Depends(get_db)):
    ent = db.get(Entity, entity_id)
    if not ent:
        raise HTTPException(404, "entity not found")
    # simple 1-hop graph from co-mentions
    subq = select(EntityChunk.chunk_id).where(EntityChunk.entity_id == entity_id).subquery()
    neighs = (
        select(Entity.id, Entity.name, func.count().label("w"))
        .join(EntityChunk, EntityChunk.entity_id == Entity.id)
        .where(EntityChunk.chunk_id.in_(select(subq.c.chunk_id)))
        .where(Entity.id != entity_id)
        .group_by(Entity.id, Entity.name)
        .order_by(func.count().desc())
        .limit(20)
    )
    nodes = [{"id": f"e:{entity_id}", "label": ent.name, "type": "entity"}]
    edges = []
    for i, n, w in db.execute(neighs).all():
        nodes.append({"id": f"e:{i}", "label": n, "type": "entity"})
        edges.append({"source": f"e:{entity_id}", "target": f"e:{i}", "weight": int(w)})
    return {"nodes": nodes, "edges": edges}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/stats")
def stats(db: Session = Depends(get_db)):
    doc_count = db.scalar(select(func.count()).select_from(Document)) or 0
    chunk_count = db.scalar(select(func.count()).select_from(Chunk)) or 0
    entity_count = db.scalar(select(func.count()).select_from(Entity)) or 0
    return {"documents": doc_count, "chunks": chunk_count, "entities": entity_count}
