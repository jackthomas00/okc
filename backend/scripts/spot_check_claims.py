#!/usr/bin/env python3
"""
Spot check Claim + ClaimSource rows to verify reasonable sentences are being picked.
Shows claim text, chunk context, and highlights the sentence.

NOTE: This script is for Milestone 2 (Claims & Relations) and will not work in Milestone 1.
The Claim and ClaimSource models have been removed from the schema.
"""
from sqlalchemy import select, func
from sqlalchemy.orm import joinedload

from api.db import SessionLocal
from api.models import Claim, ClaimSource, Chunk, Document


def highlight_sentence_in_context(chunk_text: str, claim_text: str, context_chars: int = 100) -> str:
    """Find claim text in chunk and show with surrounding context."""
    claim_text = claim_text.strip()
    idx = chunk_text.find(claim_text)
    
    if idx == -1:
        # Try case-insensitive
        idx = chunk_text.lower().find(claim_text.lower())
        if idx == -1:
            return f"[Claim text not found in chunk]\nChunk: {chunk_text[:200]}..."
    
    start = max(0, idx - context_chars)
    end = min(len(chunk_text), idx + len(claim_text) + context_chars)
    
    before = chunk_text[start:idx]
    match = chunk_text[idx:idx + len(claim_text)]
    after = chunk_text[idx + len(claim_text):end]
    
    return f"...{before}>>>{match}<<<{after}..."


def main(num_samples: int = 10):
    db = SessionLocal()
    try:
        # Get total counts
        total_claims = db.scalar(select(func.count()).select_from(Claim)) or 0
        total_sources = db.scalar(select(func.count()).select_from(ClaimSource)) or 0
        
        print(f"Database stats:")
        print(f"  Total Claims: {total_claims}")
        print(f"  Total ClaimSources: {total_sources}")
        print()
        
        if total_claims == 0:
            print("No claims found in database.")
            return
        
        # Sample some claims with their sources
        claims = db.scalars(
            select(Claim)
            .order_by(func.random())
            .limit(num_samples)
        ).all()
        
        print(f"Spot checking {len(claims)} random claims:\n")
        print("=" * 80)
        
        for i, claim in enumerate(claims, 1):
            print(f"\n[{i}/{len(claims)}] Claim ID: {claim.id}")
            print(f"Confidence: {claim.confidence:.2f}")
            print(f"Claim Text: {claim.text}")
            print()
            
            # Get sources for this claim
            sources = db.scalars(
                select(ClaimSource)
                .where(ClaimSource.claim_id == claim.id)
            ).all()
            
            print(f"  Sources: {len(sources)}")
            
            for j, source in enumerate(sources, 1):
                chunk = db.get(Chunk, source.chunk_id) if source.chunk_id else None
                doc = db.get(Document, source.document_id) if source.document_id else None
                
                if chunk:
                    print(f"\n  Source {j}:")
                    if doc:
                        print(f"    Document: {doc.title or doc.id}")
                    print(f"    Chunk ID: {chunk.id}")
                    print(f"    Context:")
                    highlighted = highlight_sentence_in_context(chunk.text or "", claim.text or "")
                    print(f"    {highlighted}")
                else:
                    print(f"  Source {j}: [Chunk not found]")
            
            print("\n" + "=" * 80)
        
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(num_samples)

