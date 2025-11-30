#!/usr/bin/env python3
"""
Inspect extracted relations alongside their head/tail entities.

Example:
    python okc_cli/inspect_relations.py --limit 40
    python okc_cli/inspect_relations.py --relation-type improves --show-sentences
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from okc_core.db import SessionLocal
from okc_core.models import Entity, Relation, RelationEvidence, Sentence


def summarize(relations: Iterable[Relation]) -> None:
    """Print quick counts by relation type and entity-type pairs."""
    relation_type_counts = Counter()
    pair_counts = Counter()

    for rel in relations:
        relation_type_counts[rel.relation_type] += 1
        head_type = rel.head_entity.type or "Unknown"
        tail_type = rel.tail_entity.type or "Unknown"
        pair_counts[(rel.relation_type, head_type, tail_type)] += 1

    if relation_type_counts:
        print("Counts by relation_type:")
        for rel_type, count in relation_type_counts.most_common():
            print(f"  {rel_type:15s} {count}")
    if pair_counts:
        print("\nCounts by relation_type + (head_type -> tail_type):")
        for (rel_type, h_type, t_type), count in pair_counts.most_common():
            print(f"  {rel_type:15s} {h_type:12s} -> {t_type:12s} : {count}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Join relations with entities to inspect what was extracted.")
    parser.add_argument("--limit", type=int, default=50, help="Max relations to display (after filtering).")
    parser.add_argument("--relation-type", type=str, help="Filter to a specific relation_type.")
    parser.add_argument(
        "--show-sentences",
        action="store_true",
        help="Also show one sentence of evidence for each relation (if available).",
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        stmt = (
            select(Relation)
            .options(
                selectinload(Relation.head_entity),
                selectinload(Relation.tail_entity),
                selectinload(Relation.evidence).selectinload(RelationEvidence.sentence),
            )
            .order_by(Relation.created_at.desc())
        )
        if args.relation_type:
            stmt = stmt.where(Relation.relation_type == args.relation_type)

        relations = list(db.scalars(stmt.limit(args.limit)))

        if not relations:
            print("No relations found for the current filters.")
            return

        summarize(relations)

        print(f"Showing {len(relations)} relation rows:")
        for rel in relations:
            head: Entity = rel.head_entity
            tail: Entity = rel.tail_entity
            head_label = f"{head.canonical_name} ({head.type or 'Unknown'})"
            tail_label = f"{tail.canonical_name} ({tail.type or 'Unknown'})"
            evidence_count = len(rel.evidence)
            print(f"- [{rel.relation_type}] {head_label} -> {tail_label}  | confidence={rel.confidence:.2f}  | evidence={evidence_count}")

            if args.show_sentences and rel.evidence:
                sentence: Sentence | None = rel.evidence[0].sentence
                if sentence:
                    text = sentence.text.replace("\n", " ").strip()
                    print(f"    e.g., {text[:280]}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
