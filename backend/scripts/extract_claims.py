# scripts/extract_claims.py
from api.db import SessionLocal  # noqa: E402
from api.models import Chunk  # noqa: E402
from pipeline.claims.claims import extract_claims_for_chunk  # noqa: E402

from sqlalchemy import select, func


def main(batch_size: int = 100, commit_interval: int = 100):
    """
    Extract claims from chunks in batches to avoid memory issues.
    
    Args:
        batch_size: Number of chunks to fetch from DB at once (default: 100)
        commit_interval: Number of chunks to process before committing (default: 100)
    """
    db = SessionLocal()
    try:
        # Count total chunks for progress tracking
        total_count = db.execute(select(func.count(Chunk.id))).scalar()
        if total_count == 0:
            print("No chunks found in database.")
            return
        print(f"Found {total_count} chunks to process.")
        
        processed = 0
        errors = 0
        offset = 0
        
        print(f"Processing chunks in batches of {batch_size}, committing every {commit_interval} chunks...")
        
        # Process chunks in batches using LIMIT/OFFSET to avoid cursor issues
        # This prevents loading all chunks into memory simultaneously
        while offset < total_count:
            # Fetch a batch of chunks
            batch = list(
                db.execute(
                    select(Chunk)
                    .order_by(Chunk.id)
                    .limit(batch_size)
                    .offset(offset)
                ).scalars()
            )
            
            if not batch:
                break
            
            # Process each chunk in the batch
            for chunk in batch:
                try:
                    extract_claims_for_chunk(db, chunk)
                    processed += 1
                    
                    # Commit periodically to avoid losing all work and improve performance
                    if processed % commit_interval == 0:
                        db.commit()
                        print(f"Processed {processed}/{total_count} chunks (committed)...")
                        
                except Exception as e:
                    errors += 1
                    print(f"Error processing chunk {chunk.id}: {e}", file=sys.stderr)
                    # Rollback only the current chunk's transaction
                    db.rollback()
                    # Continue processing other chunks
                    continue
            
            offset += batch_size
            
            # Clear the batch from memory
            del batch
        
        # Final commit for any remaining chunks
        if processed % commit_interval != 0:
            db.commit()
            print(f"Final commit: processed {processed} chunks total.")
        
        print(f"\nCompleted: {processed} chunks processed, {errors} errors encountered.")
        
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
