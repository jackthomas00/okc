#!/usr/bin/env python3
"""
Scan the document table, backfill doc-level embeddings (mean of chunk vectors),
and emit any pairs whose cosine similarity clears a threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sqlalchemy import select, update

from api.db import SessionLocal  # noqa: E402
from api.models import Document, Chunk  # noqa: E402
from pipeline.embeddings.doc_embeddings import normalize_vector  # noqa: E402


def backfill_missing_embeddings(session, batch_size: int = 100) -> int:
    doc_ids = session.scalars(select(Document.id).where(Document.doc_embedding.is_(None))).all()
    if not doc_ids:
        return 0
    updated = 0
    for idx, doc_id in enumerate(doc_ids, start=1):
        chunk_rows = session.execute(
            select(Chunk.embedding).where(Chunk.document_id == doc_id).order_by(Chunk.idx.asc())
        ).all()
        if not chunk_rows:
            continue
        vecs = np.asarray([row[0] for row in chunk_rows], dtype=np.float32)
        if vecs.size == 0:
            continue
        doc_vec = normalize_vector(vecs.mean(axis=0))
        session.execute(
            update(Document).where(Document.id == doc_id).values(doc_embedding=doc_vec.astype(np.float32).tolist())
        )
        updated += 1
        if idx % batch_size == 0:
            session.commit()
    session.commit()
    return updated


def load_document_vectors(session) -> tuple[list[int], list[str | None], np.ndarray]:
    rows = session.execute(
        select(Document.id, Document.title, Document.doc_embedding)
        .where(Document.doc_embedding.is_not(None))
        .order_by(Document.id.asc())
    ).all()
    ids = []
    titles = []
    vectors = []
    for doc_id, title, emb in rows:
        ids.append(doc_id)
        titles.append(title)
        vectors.append(emb)
    if not vectors:
        return ids, titles, np.zeros((0, 0), dtype=np.float32)
    vecs = np.asarray(vectors, dtype=np.float32)
    # ensure unit norm (mean-pooled vectors can drift slightly)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    return ids, titles, vecs


def find_duplicate_pairs(doc_ids: list[int], titles: list[str | None], vecs: np.ndarray,
                         threshold: float, limit: int | None) -> list[dict]:
    n = len(doc_ids)
    if n == 0:
        return []
    sims = np.clip(vecs @ vecs.T, -1.0, 1.0)
    duplicates: list[dict] = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sims[i, j])
            if sim >= threshold:
                duplicates.append({
                    "doc_a": {"id": doc_ids[i], "title": titles[i]},
                    "doc_b": {"id": doc_ids[j], "title": titles[j]},
                    "similarity": sim,
                })
                if limit and len(duplicates) >= limit:
                    return duplicates
    return sorted(duplicates, key=lambda x: x["similarity"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Report near-duplicate documents by cosine similarity.")
    parser.add_argument("--threshold", type=float, default=0.95, help="Cosine similarity threshold (default: 0.95)")
    parser.add_argument("--limit", type=int, default=200, help="Maximum duplicate pairs to report (0 = no limit)")
    parser.add_argument("--output", type=Path, help="Optional JSON file to write duplicate report")
    parser.add_argument("--skip-backfill", action="store_true", help="Skip doc embedding backfill step")
    args = parser.parse_args()

    session = SessionLocal()
    try:
        if not args.skip_backfill:
            updated = backfill_missing_embeddings(session)
            if updated:
                print(f"[dedupe] backfilled doc embeddings for {updated} documents")
        ids, titles, vecs = load_document_vectors(session)
        if not ids:
            print("[dedupe] no documents with embeddings found")
            return
        limit = args.limit if args.limit and args.limit > 0 else None
        duplicates = find_duplicate_pairs(ids, titles, vecs, args.threshold, limit)
        if not duplicates:
            print(f"[dedupe] no duplicates found at threshold {args.threshold}")
        else:
            print(f"[dedupe] found {len(duplicates)} potential duplicate pairs")
            for dup in duplicates[:20]:
                print(
                    f"  doc {dup['doc_a']['id']} ↔ doc {dup['doc_b']['id']} "
                    f"(sim={dup['similarity']:.4f})"
                )
        if args.output:
            payload = {
                "threshold": args.threshold,
                "pair_count": len(duplicates),
                "pairs": duplicates,
            }
            args.output.write_text(json.dumps(payload, indent=2))
            print(f"[dedupe] wrote report to {args.output}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
