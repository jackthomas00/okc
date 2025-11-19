import numpy as np
from sqlalchemy import select, exists, update, Float as SQLFloat, cast
from sqlalchemy.dialects.postgresql import insert as pg_upsert
from sqlalchemy.orm import Session
from api.models import Document, Chunk, Entity
from pipeline.utils.text_cleaning import extract_candidate_entities, canonicalize
from pipeline.utils.spacy_processing import make_doc, merge_entity_candidates
from pipeline.embeddings.embedder import embed_texts
from pipeline.ingestion.ingest_utils import normalize_text, content_sha1, word_count
from pipeline.embeddings.doc_embeddings import aggregate_chunk_embeddings, embedding_to_list

DUPLICATE_SIMILARITY_THRESHOLD = 0.95
DUPLICATE_CANDIDATE_K = 5

def upsert_entity(session: Session, canonical_name: str) -> int:
    """
    Upsert an entity with canonical_name and normalized_name.
    Returns the entity ID.
    """
    normalized = canonicalize(canonical_name)
    stmt = pg_upsert(Entity).values(
        canonical_name=canonical_name,
        normalized_name=normalized
    ).on_conflict_do_nothing(index_elements=[Entity.canonical_name])
    session.execute(stmt)
    # fetch existing or just inserted
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
