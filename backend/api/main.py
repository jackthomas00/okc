from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import select, func, Float as SQLFloat, cast
from api.db import Base, engine, SessionLocal
from api.models import Document, Chunk, Entity, EntityChunk, Topic, TopicMember
from api.schemas import IngestRequest, BulkIngestRequest, IngestResult, SearchResponseItem, EntitySearchResult, UnifiedSearchResult
from pipeline.ingestion.ingest_utils import detect_lang_optional
from pipeline.utils.text_cleaning import chunk_text
from api.crud import ingest_document, add_chunks_with_embeddings, extract_and_link_entities, update_document_embedding
from pipeline.embeddings.embedder import embed_texts
from pipeline.embeddings.doc_embeddings import compute_doc_embedding

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
    for cid, text in zip(chunk_ids, chunks):
        extract_and_link_entities(db, cid, text)

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
        for cid, text in zip(cids, chunks):
            extract_and_link_entities(db, cid, text)
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
    Unified search across topics, entities, and documents.
    Returns results ranked by semantic similarity.
    """
    # Embed the query for vector search
    qvec = embed_texts([q])[0].tolist()
    results: list[UnifiedSearchResult] = []
    
    # Search topics by centroid similarity
    topic_dist_raw = Topic.centroid.op("<=>")(qvec)
    topic_dist_expr = cast(topic_dist_raw, SQLFloat).label("distance")
    topic_stmt = (
        select(Topic.id, Topic.label, Topic.summary, topic_dist_expr)
        .where(Topic.centroid.isnot(None))
        .order_by(topic_dist_raw.asc())
        .limit(k)
    )
    topic_rows = db.execute(topic_stmt).all()
    for tid, label, summary, dist in topic_rows:
        try:
            d = float(dist or 1.0)
        except (TypeError, ValueError):
            d = 1.0
        score = max(0.0, min(1.0, 1.0 - d))
        results.append(
            UnifiedSearchResult(
                id=tid,
                type="topic",
                title=label or f"Topic {tid}",
                snippet=summary[:200] if summary else None,
                score=score,
            )
        )
    
    # Search entities by centroid similarity (hybrid with text)
    entity_dist_raw = Entity.centroid.op("<=>")(qvec)
    entity_dist_expr = cast(entity_dist_raw, SQLFloat).label("distance")
    entity_vector_stmt = (
        select(Entity.id, Entity.name, Entity.canonical_label, entity_dist_expr)
        .where(Entity.centroid.isnot(None))
        .order_by(entity_dist_raw.asc())
        .limit(k)
    )
    entity_vector_rows = db.execute(entity_vector_stmt).all()
    
    # Also get text matches for entities
    q_like = f"%{q}%"
    entity_text_stmt = (
        select(Entity.id, Entity.name, Entity.canonical_label)
        .where(Entity.name.ilike(q_like))
        .limit(k)
    )
    entity_text_rows = db.execute(entity_text_stmt).all()
    
    # Combine entity results
    entity_results_dict: dict[int, dict] = {}
    for eid, name, canonical_label, dist in entity_vector_rows:
        try:
            d = float(dist or 1.0)
        except (TypeError, ValueError):
            d = 1.0
        vector_score = max(0.0, min(1.0, 1.0 - d))
        entity_results_dict[eid] = {
            "id": eid,
            "name": name,
            "canonical_label": canonical_label,
            "vector_score": vector_score,
            "text_score": 0.0,
        }
    
    q_lower = q.lower()
    for eid, name, canonical_label in entity_text_rows:
        name_lower = name.lower()
        if name_lower == q_lower:
            text_score = 1.0
        elif name_lower.startswith(q_lower):
            text_score = 0.8
        else:
            text_score = 0.6
        
        if eid not in entity_results_dict:
            entity_results_dict[eid] = {
                "id": eid,
                "name": name,
                "canonical_label": canonical_label,
                "vector_score": 0.0,
                "text_score": text_score,
            }
        else:
            entity_results_dict[eid]["text_score"] = text_score
    
    # Add entity results with hybrid score
    for eid, data in entity_results_dict.items():
        hybrid_score = 0.6 * data["vector_score"] + 0.4 * data["text_score"]
        results.append(
            UnifiedSearchResult(
                id=data["id"],
                type="entity",
                title=data["name"],
                snippet=data["canonical_label"],
                score=hybrid_score,
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
    Hybrid search combining vector similarity (centroid) and text matching (ILIKE).
    Returns entities ranked by a combination of semantic similarity and text match.
    """
    # Embed the query for vector search
    qvec = embed_texts([q])[0].tolist()
    
    # Vector search: cosine distance on Entity.centroid
    dist_raw = Entity.centroid.op("<=>")(qvec)
    dist_expr = cast(dist_raw, SQLFloat).label("distance")
    
    # Get vector search results (top k*2 to have enough candidates)
    vector_stmt = (
        select(
            Entity.id,
            Entity.name,
            Entity.canonical_label,
            dist_expr,
        )
        .where(Entity.centroid.isnot(None))
        .order_by(dist_raw.asc())
        .limit(k * 2)
    )
    vector_rows = db.execute(vector_stmt).all()
    
    # Text search: ILIKE matching
    q_like = f"%{q}%"
    text_stmt = (
        select(Entity.id, Entity.name, Entity.canonical_label)
        .where(Entity.name.ilike(q_like))
        .limit(k * 2)
    )
    text_rows = db.execute(text_stmt).all()
    
    # Combine results: create a dict keyed by entity id
    results_dict: dict[int, dict] = {}
    
    # Process vector results (weight: 0.6)
    for eid, name, canonical_label, dist in vector_rows:
        try:
            d = float(dist or 1.0)
        except (TypeError, ValueError):
            d = 1.0
        vector_score = max(0.0, min(1.0, 1.0 - d))  # convert distance to similarity
        
        if eid not in results_dict:
            results_dict[eid] = {
                "id": eid,
                "name": name,
                "canonical_label": canonical_label,
                "vector_score": vector_score,
                "text_score": 0.0,
            }
        else:
            results_dict[eid]["vector_score"] = vector_score
    
    # Process text results (weight: 0.4)
    # Simple text match score: 1.0 if exact match, 0.8 if starts with, 0.6 otherwise
    q_lower = q.lower()
    for eid, name, canonical_label in text_rows:
        name_lower = name.lower()
        if name_lower == q_lower:
            text_score = 1.0
        elif name_lower.startswith(q_lower):
            text_score = 0.8
        else:
            text_score = 0.6
        
        if eid not in results_dict:
            results_dict[eid] = {
                "id": eid,
                "name": name,
                "canonical_label": canonical_label,
                "vector_score": 0.0,
                "text_score": text_score,
            }
        else:
            results_dict[eid]["text_score"] = text_score
    
    # Calculate hybrid score: weighted combination
    # vector_weight = 0.6, text_weight = 0.4
    hybrid_results = []
    for eid, data in results_dict.items():
        hybrid_score = 0.6 * data["vector_score"] + 0.4 * data["text_score"]
        hybrid_results.append(
            EntitySearchResult(
                id=data["id"],
                name=data["name"],
                canonical_label=data["canonical_label"],
                score=hybrid_score,
            )
        )
    
    # Sort by hybrid score descending and return top k
    hybrid_results.sort(key=lambda x: x.score or 0.0, reverse=True)
    return hybrid_results[:k]

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

@app.get("/topic/{topic_id}")
def get_topic(topic_id: int, db: Session = Depends(get_db)):
    topic = db.get(Topic, topic_id)
    if not topic:
        raise HTTPException(404, "topic not found")
    
    # Get member entities (top N by score)
    member_stmt = (
        select(Entity.id, Entity.name, TopicMember.score)
        .join(TopicMember, TopicMember.entity_id == Entity.id)
        .where(TopicMember.topic_id == topic_id)
        .order_by(TopicMember.score.desc())
        .limit(30)
    )
    members = [
        {"id": eid, "name": name, "score": float(score)}
        for eid, name, score in db.execute(member_stmt)
    ]
    
    return {
        "id": topic.id,
        "label": topic.label,
        "summary": topic.summary,
        "members": members,
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

@app.get("/graph/topic/{topic_id}")
def graph_topic(topic_id: int, db: Session = Depends(get_db)):
    topic = db.get(Topic, topic_id)
    if not topic:
        raise HTTPException(404, "topic not found")

    # neighbors: similar topics by centroid similarity
    if topic.centroid is None:
        neighbors = []
    else:
        qvec = topic.centroid  # list[float]
        dist_raw = Topic.centroid.op("<=>")(qvec)  # cosine distance
        dist_expr = cast(dist_raw, SQLFloat).label("distance")

        neigh_stmt = (
            select(Topic.id, Topic.label, dist_expr)
            .where(Topic.id != topic_id)
            .order_by(dist_raw.asc())
            .limit(20)
        )
        neighbors = []
        for tid, label, distance in db.execute(neigh_stmt):
            d = float(distance or 0.0)
            sim = max(0.0, min(1.0, 1.0 - d))  # crude similarity
            neighbors.append({"id": tid, "label": label, "similarity": sim})

    # include member entities (top N by score) to let the UI show “what this topic is about”
    member_stmt = (
        select(Entity.id, Entity.name, TopicMember.score)
        .join(TopicMember, TopicMember.entity_id == Entity.id)
        .where(TopicMember.topic_id == topic_id)
        .order_by(TopicMember.score.desc())
        .limit(30)
    )
    members = [
        {"id": eid, "name": name, "score": float(score)}
        for eid, name, score in db.execute(member_stmt)
    ]

    nodes = [{"id": f"t:{topic_id}", "label": topic.label or f"topic {topic_id}", "type": "topic"}]
    edges = []

    for n in neighbors:
        nodes.append({"id": f"t:{n['id']}", "label": n["label"], "type": "topic"})
        edges.append({"source": f"t:{topic_id}", "target": f"t:{n['id']}", "weight": n["similarity"]})

    return {
        "topic": {
            "id": topic.id,
            "label": topic.label,
            "members": members,
        },
        "nodes": nodes,
        "edges": edges,
    }

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/stats")
def stats(db: Session = Depends(get_db)):
    doc_count = db.scalar(select(func.count()).select_from(Document)) or 0
    chunk_count = db.scalar(select(func.count()).select_from(Chunk)) or 0
    entity_count = db.scalar(select(func.count()).select_from(Entity)) or 0
    return {"documents": doc_count, "chunks": chunk_count, "entities": entity_count}
