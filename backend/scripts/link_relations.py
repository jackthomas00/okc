# scripts/link_relations.py
from api.db import SessionLocal
from pipeline.relations.relation_inserter import link_relations_for_all_claims


def main(batch_size: int = 100, commit_interval: int = 100):
    """
    Link relations for all claims above confidence threshold.
    
    Args:
        batch_size: Number of claims to fetch from DB at once (default: 100)
        commit_interval: Number of claims to process before committing (default: 100)
    """
    db = SessionLocal()
    try:
        # Note: commit_interval commits happen inside link_relations_for_all_claims
        # No need for final commit here as it's handled internally
        link_relations_for_all_claims(
            db, 
            min_claim_confidence=0.4,
            batch_size=batch_size,
            commit_interval=commit_interval
        )
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    # Allow batch_size and commit_interval to be passed as command-line args
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    commit_interval = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    main(batch_size=batch_size, commit_interval=commit_interval)
