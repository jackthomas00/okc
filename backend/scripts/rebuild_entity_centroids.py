#!/usr/bin/env python3
"""
Rebuild entity centroids by mean-pooling embeddings of chunks where entities appear.
Also updates Entity.popularity with occurrence counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from api.db import SessionLocal  # noqa: E402
from pipeline.topics.topics import rebuild_entity_centroids  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild entity centroids by mean-pooling chunk embeddings."
    )
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=2,
        help="Minimum number of occurrences required to compute centroid (default: 2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of entities to process per batch (default: 1000). Reduce if memory issues occur.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("rebuild_centroids_checkpoint.json"),
        help="Checkpoint file to save progress and resume from (default: rebuild_centroids_checkpoint.json)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if it exists",
    )
    args = parser.parse_args()

    # Load checkpoint if resuming
    start_from = 0
    if args.resume and args.checkpoint.exists():
        try:
            checkpoint_data = json.loads(args.checkpoint.read_text())
            start_from = checkpoint_data.get("last_batch", 0)
            print(f"[rebuild_entity_centroids] Resuming from batch {start_from}")
        except Exception as e:
            print(f"[rebuild_entity_centroids] Warning: Could not load checkpoint: {e}", file=sys.stderr)
            start_from = 0

    session = SessionLocal()
    try:
        print(f"[rebuild_entity_centroids] Starting with min_occurrences={args.min_occurrences}, batch_size={args.batch_size}, start_from={start_from}")
        updated_count, skipped_count, last_batch = rebuild_entity_centroids(
            session, 
            min_occurrences=args.min_occurrences,
            batch_size=args.batch_size,
            start_from=start_from
        )
        
        # Save checkpoint on success
        checkpoint_data = {
            "last_batch": last_batch,
            "updated_count": updated_count,
            "skipped_count": skipped_count,
        }
        args.checkpoint.write_text(json.dumps(checkpoint_data, indent=2))
        
        print(f"[rebuild_entity_centroids] Completed successfully: updated {updated_count} entities, skipped {skipped_count} entities")
        print(f"[rebuild_entity_centroids] Checkpoint saved to {args.checkpoint}")
    except Exception as e:
        session.rollback()
        print(f"[rebuild_entity_centroids] Error: {e}", file=sys.stderr)
        print(f"[rebuild_entity_centroids] You can resume with: --resume", file=sys.stderr)
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()

