#!/usr/bin/env python3
"""
Quick script to check how many documents match the new stricter filter.
Helps decide whether to cleanup or do a full reset.
"""
from sqlalchemy import select, func

from api.db import SessionLocal  # noqa: E402
from api.models import Document, Chunk, Entity  # noqa: E402
from pipeline.ingestion.doc_filters import keep_doc  # noqa: E402

def main():
    db = SessionLocal()
    try:
        total_docs = db.scalar(select(func.count()).select_from(Document)) or 0
        total_chunks = db.scalar(select(func.count()).select_from(Chunk)) or 0
        total_entities = db.scalar(select(func.count()).select_from(Entity)) or 0
        
        print(f"Database stats:")
        print(f"  Documents: {total_docs}")
        print(f"  Chunks: {total_chunks}")
        print(f"  Entities: {total_entities}")
        
        if total_docs == 0:
            print("\n✓ Database is empty. Ready to run dump_ingest.")
            return
        
        # Check filter match
        matching = 0
        non_matching = 0
        
        for doc in db.scalars(select(Document)):
            if keep_doc(doc.text or "", title=doc.title or ""):
                matching += 1
            else:
                non_matching += 1
        
        print(f"\nFilter analysis (new stricter filter):")
        print(f"  Matching: {matching} ({100*matching/total_docs:.1f}%)")
        print(f"  Non-matching: {non_matching} ({100*non_matching/total_docs:.1f}%)")
        
        if non_matching == 0:
            print("\n✓ All documents match the new filter! No cleanup needed.")
        elif non_matching < total_docs * 0.1:  # Less than 10% bad
            print(f"\n→ Recommendation: Use cleanup script (only {non_matching} docs to remove)")
            print(f"  Run: python scripts/cleanup_unfiltered_docs.py --execute")
        else:
            print(f"\n→ Recommendation: Consider full reset (many documents don't match)")
            print(f"  Option 1: Cleanup: python scripts/cleanup_unfiltered_docs.py --execute")
            print(f"  Option 2: Full reset: Drop tables and rerun dump_ingest")
            
    finally:
        db.close()

if __name__ == "__main__":
    main()

