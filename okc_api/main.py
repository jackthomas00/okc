from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import select, func, Float as SQLFloat, cast
from okc_core.db import Base, engine, SessionLocal
from okc_core.models import Document, Chunk, Entity, Sentence, EntityMention
from okc_core.schemas import IngestRequest, BulkIngestRequest, IngestResult, SearchResponseItem, EntitySearchResult, UnifiedSearchResult
from okc_pipeline.stage_00_ingestion.ingest_utils import detect_lang_optional
from okc_pipeline.stage_00_ingestion.chunker import chunk_text
from okc_api.crud import ingest_document, add_chunks_with_embeddings, update_document_embedding
# extract_and_link_entities removed - will be reimplemented for Milestone 1
from okc_pipeline.stage_00_embeddings.embedder import embed_texts
from okc_pipeline.stage_00_embeddings.doc_embeddings import compute_doc_embedding

app = FastAPI(title="OKC API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],  # Add your frontend origins here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    try:
        doc_vec = compute_doc_embedding(payload.text)
    except ValueError:
        doc_vec = None
    doc_id, deduped = ingest_document(db, payload.title, payload.url, payload.text, lang=lang, doc_embedding=doc_vec)
    if deduped:
        return IngestResult(document_id=None, deduped=True, num_chunks=0, title=payload.title, url=payload.url)

    chunks = chunk_text(payload.text, target_tokens=600, overlap=80)
    chunk_ids, doc_vec_updated = add_chunks_with_embeddings(db, doc_id, chunks)
    if doc_vec_updated:
        update_document_embedding(db, doc_id, doc_vec_updated)
    # TODO: Extract sentences and entities (Milestone 1)
    # for cid, text in zip(chunk_ids, chunks):
    #     extract_and_link_entities(db, cid, text)

    return IngestResult(document_id=doc_id, deduped=False, num_chunks=len(chunks), title=payload.title, url=payload.url)

@app.post("/ingest_bulk", response_model=list[IngestResult])
def ingest_bulk(payload: BulkIngestRequest, db: Session = Depends(get_db)):
    results: list[IngestResult] = []
    # Stage 1: normalize/dedupe + stage chunks for embedding in one batch per doc
    staged = []  # [(doc_id, chunks, title, url)]
    for item in payload.items:
        lang = item.lang or detect_lang_optional(item.text) or "en"
        try:
            doc_vec = compute_doc_embedding(item.text)
        except ValueError:
            doc_vec = None
        doc_id, deduped = ingest_document(db, item.title, item.url, item.text, lang=lang, doc_embedding=doc_vec)
        if deduped:
            results.append(IngestResult(document_id=None, deduped=True, num_chunks=0, title=item.title, url=item.url))
            continue
        chunks = chunk_text(item.text, target_tokens=600, overlap=80)
        staged.append((doc_id, chunks, item.title, item.url))

    # Stage 2: embed + write chunks, then entity pass per doc
    # (We embed per doc to keep memory bounded; switch to global batch if needed.)
    for doc_id, chunks, title, url in staged:
        cids, doc_vec_updated = add_chunks_with_embeddings(db, doc_id, chunks)
        if doc_vec_updated:
            update_document_embedding(db, doc_id, doc_vec_updated)
        # TODO: Extract sentences and entities (Milestone 1)
        # for cid, text in zip(cids, chunks):
        #     extract_and_link_entities(db, cid, text)
        results.append(IngestResult(document_id=doc_id, deduped=False, num_chunks=len(chunks), title=title, url=url))

    return results

@app.get("/chunk/vector/search", response_model=list[SearchResponseItem])
def chunk_vector_search(
    q: str = Query(..., min_length=2),
    k: int = 10,
    db: Session = Depends(get_db),
):
    # embed and convert to plain Python list so it can be bound as a pgvector parameter
    qvec = embed_texts([q])[0].tolist()

    # cosine distance operator: <=> (smaller = more similar)
    # Cast to Float to ensure SQLAlchemy treats it as a float, not a vector
    dist_raw = Chunk.embedding.op("<=>")(qvec)
    dist_expr = cast(dist_raw, SQLFloat).label("distance")

    stmt = (
        select(
            Chunk.id,
            Chunk.document_id,
            Document.title,
            Chunk.text,
            dist_expr,
        )
        .select_from(Chunk)
        .join(Document, Document.id == Chunk.document_id)
        .order_by(dist_raw.asc())
        .limit(k)
    )

    rows = db.execute(stmt).all()
    out: list[SearchResponseItem] = []

    for cid, did, title, text, dist in rows:
        snippet = text[:280].replace("\n", " ")
        # convert distance to a similarity-ish score if you want (1 - dist, clamped)
        try:
            d = float(dist or 0.0)
        except (TypeError, ValueError):
            d = 0.0
        score = max(0.0, min(1.0, 1.0 - d))  # optional

        out.append(
            SearchResponseItem(
                chunk_id=cid,
                document_id=did,
                title=title,
                snippet=snippet + ("…" if len(text) > 280 else ""),
                score=score,
            )
        )

    return out

@app.get("/search", response_model=list[UnifiedSearchResult])
def unified_search(
    q: str = Query(..., min_length=2),
    k: int = 20,
    db: Session = Depends(get_db),
):
    """
    Unified search across entities and documents.
    Returns results ranked by semantic similarity.
    Note: Topics removed for Milestone 1.
    """
    # Embed the query for vector search
    qvec = embed_texts([q])[0].tolist()
    results: list[UnifiedSearchResult] = []
    
    # Search entities by text matching (centroid removed in Milestone 1)
    q_like = f"%{q}%"
    entity_text_stmt = (
        select(Entity.id, Entity.canonical_name, Entity.type)
        .where(Entity.canonical_name.ilike(q_like))
        .limit(k)
    )
    entity_text_rows = db.execute(entity_text_stmt).all()
    
    # Add entity results
    q_lower = q.lower()
    for eid, canonical_name, entity_type in entity_text_rows:
        name_lower = canonical_name.lower()
        if name_lower == q_lower:
            text_score = 1.0
        elif name_lower.startswith(q_lower):
            text_score = 0.8
        else:
            text_score = 0.6
        
        results.append(
            UnifiedSearchResult(
                id=eid,
                type="entity",
                title=canonical_name,
                snippet=entity_type,
                score=text_score,
            )
        )
    
    # Search documents by doc_embedding similarity
    doc_dist_raw = Document.doc_embedding.op("<=>")(qvec)
    doc_dist_expr = cast(doc_dist_raw, SQLFloat).label("distance")
    doc_stmt = (
        select(Document.id, Document.title, Document.text, doc_dist_expr)
        .where(Document.doc_embedding.isnot(None))
        .order_by(doc_dist_raw.asc())
        .limit(k)
    )
    doc_rows = db.execute(doc_stmt).all()
    for did, title, text, dist in doc_rows:
        try:
            d = float(dist or 1.0)
        except (TypeError, ValueError):
            d = 1.0
        score = max(0.0, min(1.0, 1.0 - d))
        snippet = text[:200].replace("\n", " ") if text else None
        results.append(
            UnifiedSearchResult(
                id=did,
                type="document",
                title=title or f"Document {did}",
                snippet=snippet,
                score=score,
            )
        )
    
    # Sort all results by score descending and return top k
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:k]

@app.get("/entities/search")
def entities_search(q: str, k: int = 20, db: Session = Depends(get_db)):
    """
    Text-based search for entities (centroid removed in Milestone 1).
    Returns entities ranked by text match.
    """
    # Text search: ILIKE matching
    q_like = f"%{q}%"
    text_stmt = (
        select(Entity.id, Entity.canonical_name, Entity.type)
        .where(Entity.canonical_name.ilike(q_like))
        .limit(k * 2)
    )
    text_rows = db.execute(text_stmt).all()
    
    # Process text results
    # Simple text match score: 1.0 if exact match, 0.8 if starts with, 0.6 otherwise
    q_lower = q.lower()
    results = []
    for eid, canonical_name, entity_type in text_rows:
        name_lower = canonical_name.lower()
        if name_lower == q_lower:
            text_score = 1.0
        elif name_lower.startswith(q_lower):
            text_score = 0.8
        else:
            text_score = 0.6
        
        results.append(
            EntitySearchResult(
                id=eid,
                name=canonical_name,
                canonical_label=entity_type,
                score=text_score,
            )
        )
    
    # Sort by score descending and return top k
    results.sort(key=lambda x: x.score or 0.0, reverse=True)
    return results[:k]

@app.get("/entity/{entity_id}")
def get_entity(entity_id: int, db: Session = Depends(get_db)):
    ent = db.get(Entity, entity_id)
    if not ent:
        raise HTTPException(404, "entity not found")
    
    # Co-mention neighbors: entities that appear in same sentences (Milestone 1)
    neighbor_stmt = (
        select(Entity.id, Entity.canonical_name, func.count().label("co_count"))
        .join(EntityMention, EntityMention.entity_id == Entity.id)
        .join(
            Sentence,
            Sentence.id == EntityMention.sentence_id
        )
        .where(
            Sentence.id.in_(
                select(EntityMention.sentence_id)
                .where(EntityMention.entity_id == entity_id)
            )
        )
        .where(Entity.id != entity_id)
        .group_by(Entity.id, Entity.canonical_name)
        .order_by(func.count().desc())
        .limit(20)
    )
    neighbors = [{"id": i, "name": n, "weight": int(c)} for i, n, c in db.execute(neighbor_stmt).all()]
    
    # Get sentences where this entity is mentioned
    mention_stmt = (
        select(Sentence.id, Sentence.text, Sentence.chunk_id, Chunk.document_id, Document.title, Document.source_url)
        .join(EntityMention, EntityMention.sentence_id == Sentence.id)
        .join(Chunk, Chunk.id == Sentence.chunk_id)
        .join(Document, Document.id == Chunk.document_id)
        .where(EntityMention.entity_id == entity_id)
        .limit(50)
    )
    mentions = []
    for sent_id, sent_text, chunk_id, doc_id, doc_title, doc_url in db.execute(mention_stmt).all():
        mentions.append({
            "sentence_id": sent_id,
            "text": sent_text,
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "document_title": doc_title,
            "document_url": doc_url,
        })
    
    return {
        "id": ent.id,
        "canonical_name": ent.canonical_name,
        "type": ent.type,
        "normalized_name": ent.normalized_name,
        "extra_metadata": ent.extra_metadata,
        "neighbors": neighbors,
        "mentions": mentions,
    }

# Topic endpoints removed for Milestone 1
# @app.get("/topic/{topic_id}")
# def get_topic(topic_id: int, db: Session = Depends(get_db)):
#     ...

@app.get("/graph/entity/{entity_id}")
def graph_neighbors(entity_id: int, db: Session = Depends(get_db)):
    ent = db.get(Entity, entity_id)
    if not ent:
        raise HTTPException(404, "entity not found")
    
    nodes = [{"id": f"e:{entity_id}", "label": ent.canonical_name, "type": "entity"}]
    edges = []
    node_ids = {entity_id}
    
    # Co-mention neighbors: entities that appear in same sentences (Milestone 1)
    neighs = (
        select(Entity.id, Entity.canonical_name, func.count().label("w"))
        .join(EntityMention, EntityMention.entity_id == Entity.id)
        .join(
            Sentence,
            Sentence.id == EntityMention.sentence_id
        )
        .where(
            Sentence.id.in_(
                select(EntityMention.sentence_id)
                .where(EntityMention.entity_id == entity_id)
            )
        )
        .where(Entity.id != entity_id)
        .group_by(Entity.id, Entity.canonical_name)
        .order_by(func.count().desc())
        .limit(20)
    )
    for i, n, w in db.execute(neighs).all():
        if i not in node_ids:
            nodes.append({"id": f"e:{i}", "label": n, "type": "entity"})
            node_ids.add(i)
        edges.append({
            "source": f"e:{entity_id}",
            "target": f"e:{i}",
            "type": "co-mention",
            "weight": int(w)
        })
    
    return {"nodes": nodes, "edges": edges}

# Topic graph endpoint removed for Milestone 1
# @app.get("/graph/topic/{topic_id}")
# def graph_topic(topic_id: int, db: Session = Depends(get_db)):
#     ...

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/stats")
def stats(db: Session = Depends(get_db)):
    doc_count = db.scalar(select(func.count()).select_from(Document)) or 0
    chunk_count = db.scalar(select(func.count()).select_from(Chunk)) or 0
    entity_count = db.scalar(select(func.count()).select_from(Entity)) or 0
    return {"documents": doc_count, "chunks": chunk_count, "entities": entity_count}
