import numpy as np
from sqlalchemy import select, exists, update, Float as SQLFloat, cast
from sqlalchemy.dialects.postgresql import insert as pg_upsert
from sqlalchemy.orm import Session
from okc_core.models import Document, Chunk, Entity
from okc_pipeline.utils.text_cleaning import canonicalize
from okc_pipeline.stage_02_entities.entity_normalizer import normalize_entity_name
from okc_pipeline.stage_00_embeddings.embedder import embed_texts
from okc_pipeline.stage_00_ingestion.ingest_utils import normalize_text, content_sha1, word_count
from okc_pipeline.stage_00_embeddings.doc_embeddings import aggregate_chunk_embeddings, embedding_to_list

DUPLICATE_SIMILARITY_THRESHOLD = 0.95
DUPLICATE_CANDIDATE_K = 5

def upsert_entity(session: Session, canonical_name: str, entity_type: str | None = None) -> int:
    """
    Upsert an entity with canonical_name, normalized_name, and optional type.
    
    This function checks for existing entities by normalized_name to merge duplicates
    like "apple ii" and "The Apple ii" into a single entity.
    
    Args:
        session: Database session
        canonical_name: Entity canonical name
        entity_type: Optional entity type (if provided and entity exists, updates type)
    
    Returns:
        Entity ID (may return existing entity ID if normalized_name matches)
    """
    normalized = normalize_entity_name(canonical_name)
    
    # First check if entity with exact canonical_name exists
    existing_by_canonical = session.scalar(
        select(Entity).where(Entity.canonical_name == canonical_name)
    )
    
    if existing_by_canonical:
        # Update normalized_name and type if needed
        if existing_by_canonical.normalized_name != normalized:
            existing_by_canonical.normalized_name = normalized
        if entity_type is not None and existing_by_canonical.type != entity_type:
            existing_by_canonical.type = entity_type
        session.flush()
        return existing_by_canonical.id
    
    # Check if entity with same normalized_name exists (for deduplication)
    if normalized:
        existing_by_normalized = session.scalar(
            select(Entity).where(Entity.normalized_name == normalized)
        )
        
        if existing_by_normalized:
            # Entity with same normalized name exists - return that ID
            # This merges duplicates like "apple ii" and "The Apple ii"
            # Update type if provided
            if entity_type is not None and existing_by_normalized.type != entity_type:
                existing_by_normalized.type = entity_type
            session.flush()
            return existing_by_normalized.id
    
    # Insert new entity
    stmt = pg_upsert(Entity).values(
        canonical_name=canonical_name,
        normalized_name=normalized,
        type=entity_type
    ).on_conflict_do_update(
        index_elements=[Entity.canonical_name],
        set_={"type": entity_type, "normalized_name": normalized}
    )
    session.execute(stmt)
    session.flush()
    
    # Fetch the entity ID
    entity_id = session.scalar(select(Entity.id).where(Entity.canonical_name == canonical_name))
    return entity_id

def doc_exists_by_hash(session: Session, h: str) -> bool:
    return session.scalar(select(exists().where(Document.content_hash == h))) or False

def insert_document_core(session: Session, *, title: str, url: str | None, text: str,
                         lang: str | None, content_hash: str, doc_embedding: list[float] | None) -> int:
    doc = Document(
        source_url=url, source_type="web" if url else "local",
        title=title, text=text, lang=lang, word_count=word_count(text),
        content_hash=content_hash, doc_embedding=doc_embedding
    )
    session.add(doc); session.flush()
    return doc.id

def distance_to_cosine(distance: float | None) -> float:
    if distance is None:
        return -1.0
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return -1.0
    return 1.0 - (d * d) / 2.0

def find_duplicate_by_embedding(session: Session, embedding: list[float], threshold: float = DUPLICATE_SIMILARITY_THRESHOLD,
                                k: int = DUPLICATE_CANDIDATE_K) -> tuple[int, float] | None:
    if not embedding:
        return None
    dist_raw = Document.doc_embedding.op("<=>")(embedding)
    dist_expr = cast(dist_raw, SQLFloat).label("distance")
    stmt = (
        select(Document.id, dist_expr)
        .where(Document.doc_embedding.is_not(None))
        .order_by(dist_raw.asc())
        .limit(k)
    )
    for doc_id, distance in session.execute(stmt):
        similarity = distance_to_cosine(distance)
        if similarity >= threshold:
            return (doc_id, similarity)
    return None

def ingest_document(session: Session, title: str, url: str | None, text: str, lang: str | None = "en",
                    doc_embedding: list[float] | None = None, similarity_threshold: float = DUPLICATE_SIMILARITY_THRESHOLD) -> tuple[int, bool]:
    # normalize + hash for dedupe
    norm = normalize_text(text)
    h = content_sha1(norm)
    if doc_exists_by_hash(session, h):
        # return an existing id to be nice (optional; here we just signal dedupe)
        return (-1, True)
    vector_payload = embedding_to_list(doc_embedding)
    duplicate_hit = None
    if vector_payload is not None:
        duplicate_hit = find_duplicate_by_embedding(session, vector_payload, threshold=similarity_threshold)
    if duplicate_hit:
        return (duplicate_hit[0], True)
    doc_id = insert_document_core(session, title=title, url=url, text=norm, lang=lang,
                                  content_hash=h, doc_embedding=vector_payload)
    return (doc_id, False)

def add_chunks_with_embeddings(session: Session, document_id: int, chunks: list[str],
                               embeddings: np.ndarray | None = None) -> tuple[list[int], list[float] | None]:
    if not chunks:
        return [], None
    vecs = embeddings
    if vecs is None:
        vecs = embed_texts(chunks)  # ndarray (n, d)
    if not isinstance(vecs, np.ndarray):
        vecs = np.asarray(vecs, dtype=np.float32)
    doc_vec = aggregate_chunk_embeddings(vecs)
    ids = []
    for i, (c, v) in enumerate(zip(chunks, vecs)):
        ch = Chunk(document_id=document_id, idx=i, text=c, embedding=v.tolist())
        session.add(ch); session.flush()
        ids.append(ch.id)
    return ids, doc_vec.tolist()

def update_document_embedding(session: Session, document_id: int, embedding: list[float] | np.ndarray | None):
    payload = embedding_to_list(embedding)
    if payload is None:
        return
    session.execute(
        update(Document).where(Document.id == document_id).values(doc_embedding=payload)
    )

# extract_and_link_entities removed - will be reimplemented for Milestone 1
# This function will need to:
# 1. Split chunk into sentences (Sentence model)
# 2. Extract entities and create EntityMention records linked to sentences
# def extract_and_link_entities(session: Session, chunk_id: int, chunk_text: str):
#     ...
