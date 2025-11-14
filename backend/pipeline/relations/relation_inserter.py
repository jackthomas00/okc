# relations.py
from __future__ import annotations

from typing import List

from sqlalchemy import select
from sqlalchemy.orm import Session

from api.models import (
    Chunk,
    Entity,
    EntityChunk,
    Relation,
    Claim,
    ClaimSource,
)
from pipeline.claims.claims import EntityOccur
from pipeline.utils.text_cleaning import sentence_spans
from pipeline.relations.relation_typer import infer_relation_type

def _resolve_canonical_entity_id(session: Session, entity_id: int) -> int:
    """
    Follow alias_of chain to get canonical entity id.
    If alias_of is not used yet, this is just identity.
    """
    visited = set()
    current = session.get(Entity, entity_id)
    while current and current.alias_of is not None and current.alias_of not in visited:
        visited.add(current.id)
        current = session.get(Entity, current.alias_of)
    return current.id if current else entity_id


def _entities_in_sentence(
    session: Session,
    chunk_id: int,
    sent_start: int,
    sent_end: int,
) -> List[EntityOccur]:
    """
    Return all EntityChunk occurrences that fall within [sent_start, sent_end).
    """
    rows = session.execute(
        select(
            EntityChunk.span_start,
            EntityChunk.span_end,
            EntityChunk.entity_id,
            Entity.name,
        )
        .join(Entity, Entity.id == EntityChunk.entity_id)
        .where(EntityChunk.chunk_id == chunk_id)
    )

    ents: List[EntityOccur] = []
    for span_start, span_end, eid, name in rows:
        if span_start is None:
            continue
        if not (sent_start <= span_start < sent_end):
            continue
        span_end = span_end if span_end is not None else span_start + len(name or "")
        ents.append(
            EntityOccur(
                entity_id=eid,
                name=name,
                span_start=span_start,
                span_end=span_end,
            )
        )

    ents.sort(key=lambda e: e.span_start)
    return ents

def link_relations_for_claim(session: Session, claim: Claim):
    """
    For a given Claim, look at each ClaimSource (chunk), find the sentence,
    and create typed Relation triples if we can.

    Currently:
    - Uses the exact claim.text to find the sentence span.
    - Requires >= 2 entities in the sentence.
    - Picks head = first entity, tail = last entity.
    - Infers relation type from sentence text.
    - Resolves aliases to canonical entity ids.
    - Inserts/upgrades Relation with evidence_claim_id and confidence.
    """
    claim_text = (claim.text or "").strip()
    if not claim_text:
        return

    # there can be multiple sources (same sentence in multiple docs/chunks)
    src_rows = session.execute(
        select(ClaimSource.chunk_id)
        .where(ClaimSource.claim_id == claim.id)
    ).all()

    for (chunk_id,) in src_rows:
        chunk = session.get(Chunk, chunk_id)
        if not chunk or not chunk.text:
            continue

        chunk_text = chunk.text
        sent_spans = sentence_spans(chunk_text)
        if not sent_spans:
            continue

        # find the sentence containing our claim text
        idx = chunk_text.find(claim_text)
        if idx == -1:
            # fallback: try case-insensitive search
            idx = chunk_text.lower().find(claim_text.lower())
            if idx == -1:
                continue

        sent_idx = None
        for i, (s_start, s_end, _s) in enumerate(sent_spans):
            if s_start <= idx < s_end:
                sent_idx = i
                break

        if sent_idx is None:
            continue

        s_start, s_end, sent_text = sent_spans[sent_idx]
        ents = _entities_in_sentence(session, chunk.id, s_start, s_end)
        if len(ents) < 2:
            continue

        sent_lower = " " + sent_text.lower() + " "
        rel_type = infer_relation_type(sent_lower)
        if rel_type is None:
            continue  # no recognizable relation pattern

        # pick head/tail
        head = ents[0]
        tail = ents[-1]

        head_id = _resolve_canonical_entity_id(session, head.entity_id)
        tail_id = _resolve_canonical_entity_id(session, tail.entity_id)
        if head_id == tail_id:
            continue

        # make similar_to symmetric by normalizing ordering
        if rel_type == "similar_to" and head_id > tail_id:
            head_id, tail_id = tail_id, head_id

        # check if relation already exists
        existing = session.execute(
            select(Relation)
            .where(Relation.head_entity_id == head_id)
            .where(Relation.tail_entity_id == tail_id)
            .where(Relation.type == rel_type)
        ).scalar_one_or_none()

        if existing:
            # bump confidence a little if we see another supporting claim
            existing.confidence = min(1.0, float(existing.confidence or 0.5) + 0.05)
            # prefer to keep original evidence_claim_id or set if missing
            if existing.evidence_claim_id is None:
                existing.evidence_claim_id = claim.id
        else:
            rel = Relation(
                head_entity_id=head_id,
                tail_entity_id=tail_id,
                type=rel_type,
                evidence_claim_id=claim.id,
                confidence=float(claim.confidence or 0.5),
            )
            session.add(rel)

def link_relations_for_all_claims(
    session: Session,
    min_claim_confidence: float = 0.4,
):
    """
    Iterate over all claims above a confidence threshold and try to
    create/update Relation triples.
    """
    rows = session.execute(
        select(Claim)
        .where(Claim.confidence >= min_claim_confidence)
    ).scalars()

    for claim in rows:
        link_relations_for_claim(session, claim)
