#!/usr/bin/env python3
"""
Clean up documents in the database that don't match the filter criteria.
This ensures the database only contains documents matching the focused domain.
"""
import sys
from pathlib import Path
from sqlalchemy import select, func, delete, update
from sqlalchemy.orm import Session

from api.db import SessionLocal  # noqa: E402
from api.models import Document, Chunk, Entity, EntityChunk  # noqa: E402
from pipeline.ingestion.doc_filters import keep_doc  # noqa: E402

def analyze_documents(session: Session) -> dict:
    """Analyze which documents match the filter and which don't."""
    all_docs = list(session.scalars(select(Document)))
    
    matching = []
    non_matching = []
    
    for doc in all_docs:
        if keep_doc(doc.text or "", title=doc.title or ""):
            matching.append(doc)
        else:
            non_matching.append(doc)
    
    return {
        "total": len(matching) + len(non_matching),
        "matching": len(matching),
        "non_matching": len(non_matching),
        "non_matching_docs": non_matching
    }

def count_related_entities(session: Session, doc_ids: list[int]) -> int:
    """Count unique entities that would be affected by deleting these documents."""
    if not doc_ids:
        return 0
    
    # Find chunks from these documents
    chunk_ids = session.scalars(
        select(Chunk.id).where(Chunk.document_id.in_(doc_ids))
    ).all()
    
    if not chunk_ids:
        return 0
    
    # Find entities in these chunks
    entity_ids = session.scalars(
        select(EntityChunk.entity_id)
        .where(EntityChunk.chunk_id.in_(chunk_ids))
        .distinct()
    ).all()
    
    return len(entity_ids)

def cleanup_unfiltered_docs(session: Session, dry_run: bool = True) -> dict:
    """
    Remove documents that don't match the filter criteria.
    
    Args:
        session: Database session
        dry_run: If True, only report what would be deleted without actually deleting
    
    Returns:
        Dictionary with stats about what was/would be deleted
    """
    stats = analyze_documents(session)
    
    if stats["non_matching"] == 0:
        print("✓ All documents match the filter criteria. Nothing to clean up.")
        return stats
    
    non_matching_docs = stats["non_matching_docs"]
    doc_ids_to_delete = [doc.id for doc in non_matching_docs]
    
    # Count related data
    chunk_count = session.scalar(
        select(func.count(Chunk.id)).where(Chunk.document_id.in_(doc_ids_to_delete))
    ) or 0
    
    entity_count = count_related_entities(session, doc_ids_to_delete)
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Documents to remove:")
    print(f"  - Documents: {stats['non_matching']}/{stats['total']}")
    print(f"  - Chunks: {chunk_count}")
    print(f"  - Affected entities: {entity_count}")
    
    if dry_run:
        print("\nSample documents that would be deleted:")
        for doc in non_matching_docs[:10]:
            title = doc.title or "(no title)"
            wc = len((doc.text or "").split())
            print(f"  - {title[:60]} ({wc} words)")
        if len(non_matching_docs) > 10:
            print(f"  ... and {len(non_matching_docs) - 10} more")
        print("\nRun with --execute to actually delete these documents.")
    else:
        print(f"\nDeleting {stats['non_matching']} documents...")
        # Use bulk delete by ID to avoid SQLAlchemy trying to nullify foreign keys
        # CASCADE will automatically delete related chunks and entity_chunks
        deleted_count = session.execute(
            delete(Document).where(Document.id.in_(doc_ids_to_delete))
        ).rowcount
        session.commit()
        print(f"✓ Deleted {deleted_count} documents and their related chunks/entity_chunks (via CASCADE)")
        
        # Clean up orphaned entities (entities with no remaining chunks)
        print("Cleaning up orphaned entities...")
        orphaned_entities = session.scalars(
            select(Entity.id)
            .where(
                ~select(EntityChunk.entity_id)
                .where(EntityChunk.entity_id == Entity.id)
                .exists()
            )
        ).all()
        
        if orphaned_entities:
            orphaned_count = len(orphaned_entities)
            print(f"  Found {orphaned_count} orphaned entities")
            
            # First, nullify alias_of references to entities we're about to delete
            # (entities that have alias_of pointing to orphaned entities)
            orphaned_set = set(orphaned_entities)
            entities_referencing_orphaned = session.scalars(
                select(Entity.id)
                .where(Entity.alias_of.in_(orphaned_set))
            ).all()
            
            if entities_referencing_orphaned:
                print(f"  Nullifying {len(entities_referencing_orphaned)} alias_of references...")
                session.execute(
                    update(Entity)
                    .where(Entity.alias_of.in_(orphaned_set))
                    .values(alias_of=None)
                )
                session.flush()
            
            # Now delete the orphaned entities using bulk delete
            deleted_entities = session.execute(
                delete(Entity).where(Entity.id.in_(orphaned_set))
            ).rowcount
            session.commit()
            print(f"✓ Deleted {deleted_entities} orphaned entities")
        else:
            print("  No orphaned entities found")
    
    return stats

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Clean up documents that don't match filter criteria")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete documents (default is dry-run)"
    )
    args = parser.parse_args()
    
    db = SessionLocal()
    try:
        cleanup_unfiltered_docs(db, dry_run=not args.execute)
    finally:
        db.close()

if __name__ == "__main__":
    main()

