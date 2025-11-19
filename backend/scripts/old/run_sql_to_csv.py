#!/usr/bin/env python3
"""
Run a SQL file and export results to CSV.
Usage: python run_sql_to_csv.py <sql_file> [output_csv]
"""
import sys
import csv
from pathlib import Path
from sqlalchemy import text
from api.db import SessionLocal

def run_sql_to_csv(sql_file: str, output_csv: str = None):
    """Execute SQL file and write results to CSV."""
    sql_path = Path(sql_file)
    if not sql_path.exists():
        print(f"Error: SQL file not found: {sql_file}", file=sys.stderr)
        sys.exit(1)
    
    # Read SQL file
    with open(sql_path, 'r') as f:
        sql_query = f.read()
    
    # Determine output filename
    if output_csv is None:
        output_csv = sql_path.stem + '.csv'
    
    # Execute query
    db = SessionLocal()
    try:
        result = db.execute(text(sql_query))
        
        # Get column names and fetch all rows
        columns = result.keys()
        rows = result.fetchall()
        
        # Write to CSV
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)  # Header
            writer.writerows(rows)
        
        print(f"✓ Query executed successfully")
        print(f"✓ Results written to: {output_csv}")
        print(f"✓ Exported {len(rows)} rows")
        
    except Exception as e:
        print(f"Error executing query: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        db.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_sql_to_csv.py <sql_file> [output_csv]", file=sys.stderr)
        sys.exit(1)
    
    sql_file = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    run_sql_to_csv(sql_file, output_csv)

