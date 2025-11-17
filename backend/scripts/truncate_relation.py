#!/usr/bin/env python3
"""
Truncate the relation table.
Usage: python truncate_relation.py
"""
import sys
from sqlalchemy import text
from api.db import SessionLocal

def truncate_relation_table():
    """Truncate the relation table."""
    db = SessionLocal()
    try:
        # Execute TRUNCATE command
        db.execute(text("TRUNCATE TABLE relation CASCADE"))
        db.commit()
        print("✓ Successfully truncated relation table")
    except Exception as e:
        db.rollback()
        print(f"Error truncating relation table: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        db.close()

if __name__ == "__main__":
    truncate_relation_table()

