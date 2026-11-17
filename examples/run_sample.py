"""Run OKC stages 1-4 (sentences, entities, claims, relations) on a text file.

Uses a throwaway SQLite database, so no Postgres or embedding model is needed.

    python examples/run_sample.py examples/sample.txt
"""

import json
import os
import sys
import tempfile
from pathlib import Path

DB_PATH = os.path.join(tempfile.mkdtemp(), "okc_sample.db")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{DB_PATH}")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from okc_api.crud import add_chunks_with_embeddings, ingest_document  # noqa: E402
from okc_core.db import Base  # noqa: E402
from okc_core.models import Chunk, ClaimSentence, Entity, Relation, RelationEvidence, Sentence  # noqa: E402
from okc_pipeline.stage_00_ingestion.chunker import chunk_text  # noqa: E402
from okc_pipeline.stage_01_sentences.sentence_splitter import split_chunk_into_sentences  # noqa: E402
from okc_pipeline.stage_02_entities.entity_extractor import extract_entities_for_sentences  # noqa: E402
from okc_pipeline.stage_03_claims.claim_detector import detect_claim_sentences  # noqa: E402
from okc_pipeline.stage_04_relations.relation_extractor import extract_relations_for_sentences  # noqa: E402


def main(path: str) -> None:
    engine = create_engine(f"sqlite:///{DB_PATH}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()

    text = Path(path).read_text()
    doc_id, _ = ingest_document(db, Path(path).stem, None, text, lang="en", doc_embedding=None)
    chunks = chunk_text(text, target_tokens=600, overlap=80)
    # Placeholder vectors: embeddings aren't needed for stages 1-4.
    chunk_ids, _ = add_chunks_with_embeddings(db, doc_id, chunks, embeddings=[[0.0] * 384 for _ in chunks])

    sentence_ids = []
    for chunk_id in chunk_ids:
        chunk = db.get(Chunk, chunk_id)
        for i, s in enumerate(split_chunk_into_sentences(chunk)):
            db.add(Sentence(chunk_id=chunk_id, text=s["text"], char_start=s["start"], char_end=s["end"], order_index=i))
        db.flush()
        sentence_ids += [s.id for s in db.query(Sentence).filter(Sentence.chunk_id == chunk_id)]

    extract_entities_for_sentences(db, sentence_ids)
    detect_claim_sentences(db, sentence_ids)
    extract_relations_for_sentences(db, sentence_ids)
    db.commit()

    relations = []
    for rel in db.query(Relation):
        evidence = db.query(RelationEvidence).filter_by(relation_id=rel.id).first()
        relations.append({
            "head": db.get(Entity, rel.head_entity_id).canonical_name,
            "relation": rel.relation_type,
            "tail": db.get(Entity, rel.tail_entity_id).canonical_name,
            "confidence": round(rel.confidence, 2),
            "evidence": db.get(Sentence, evidence.sentence_id).text if evidence else None,
        })

    print(json.dumps({
        "entities": sorted({e.canonical_name for e in db.query(Entity)}),
        "claims": [db.get(Sentence, c.sentence_id).text for c in db.query(ClaimSentence).filter(ClaimSentence.is_claim)],
        "relations": relations,
    }, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("sample.txt")))
