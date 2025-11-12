from sqlalchemy import select, exists
from sqlalchemy.dialects.postgresql import insert as pg_upsert
from sqlalchemy.orm import Session
from models import Document, Chunk, Entity, EntityChunk
from text_utils import extract_candidate_entities, canonicalize
from embed import embed_texts
from ingest_utils import normalize_text, content_sha1, word_count

def upsert_entity(session: Session, name: str) -> int:
    canon = canonicalize(name)
    stmt = pg_upsert(Entity).values(name=name, canonical_label=canon).on_conflict_do_nothing(index_elements=[Entity.name])
    session.execute(stmt)
    # fetch existing or just inserted
    entity_id = session.scalar(select(Entity.id).where(Entity.name == name))
    return entity_id

def doc_exists_by_hash(session: Session, h: str) -> bool:
    return session.scalar(select(exists().where(Document.content_hash == h))) or False

def insert_document_core(session: Session, *, title: str, url: str | None, text: str,
                         lang: str | None, content_hash: str) -> int:
    doc = Document(
        source_url=url, source_type="web" if url else "local",
        title=title, text=text, lang=lang, word_count=word_count(text),
        content_hash=content_hash
    )
    session.add(doc); session.flush()
    return doc.id

def ingest_document(session: Session, title: str, url: str | None, text: str, lang: str | None = "en") -> tuple[int, bool]:
    # normalize + hash for dedupe
    norm = normalize_text(text)
    h = content_sha1(norm)
    if doc_exists_by_hash(session, h):
        # return an existing id to be nice (optional; here we just signal dedupe)
        return (-1, True)
    doc_id = insert_document_core(session, title=title, url=url, text=norm, lang=lang, content_hash=h)
    return (doc_id, False)

def add_chunks_with_embeddings(session: Session, document_id: int, chunks: list[str]) -> list[int]:
    vecs = embed_texts(chunks)  # ndarray (n, d)
    ids = []
    for i, (c, v) in enumerate(zip(chunks, vecs)):
        ch = Chunk(document_id=document_id, idx=i, text=c, embedding=v.tolist())
        session.add(ch); session.flush()
        ids.append(ch.id)
    return ids

def extract_and_link_entities(session: Session, chunk_id: int, chunk_text: str):
    names = extract_candidate_entities(chunk_text)
    for n in names:
        eid = upsert_entity(session, n)
        pos = chunk_text.find(n)
        ec = EntityChunk(entity_id=eid, chunk_id=chunk_id,
                         span_start=pos if pos >= 0 else None,
                         span_end=(pos + len(n)) if pos >= 0 else None)
        session.add(ec)
