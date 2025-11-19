# relations.py
# NOTE: This module is for Milestone 2 (Claims & Relations) and is not used in Milestone 1.
# The Relation, Claim, and ClaimSource models have been removed from the schema.
# This file is kept for reference but should not be imported or used until Milestone 2.

from __future__ import annotations

import re
from typing import List

from sqlalchemy import select, func
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
from pipeline.utils.spacy_processing import make_doc

_BAD_SINGLE_TOKENS = {
    "Among", "Because", "For", "An", "A",
    "He", "She", "They", "It", "This", "That",
    "These", "Those", "Here", "There",
    "However", "Although", "Though", "While",
    "When", "Whereas", "If", "So", "Then",
    "Thus", "Hence", "Therefore",
}


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

def _dedupe_overlapping_entities(ents: list[EntityOccur]) -> list[EntityOccur]:
    # sort by start, then by descending length
    ents_sorted = sorted(
        ents,
        key=lambda e: (e.span_start, -(e.span_end - e.span_start))
    )
    result: list[EntityOccur] = []
    for e in ents_sorted:
        if result and e.span_start >= result[-1].span_start and e.span_end <= result[-1].span_end:
            # e is fully contained in the last kept entity -> drop shorter fragment
            continue
        result.append(e)
    return result

def _filter_relation_entities(ents: list[EntityOccur]) -> list[EntityOccur]:
    filtered: list[EntityOccur] = []
    for e in ents:
        name = (e.name or "").strip()
        if not name:
            continue

        # Drop pure numbers (years, counts)
        if name.isdigit():
            continue

        # Require some alpha
        if not re.search(r"[A-Za-z]", name):
            continue

        tokens = name.split()
        if len(tokens) == 1:
            # Drop obvious function words / pronouns
            if name in _BAD_SINGLE_TOKENS:
                continue

        filtered.append(e)

    # If we end up with only 0–1 entities after filtering, there's nothing to relate
    return filtered

_RELATION_TRIGGERS: dict[str, list[tuple[str, bool]]] = {
    "is_a": [
        (" is a ", False),
        (" is an ", False),
        (" are a ", False),
        (" are an ", False),
    ],
    "part_of": [
        (" is part of ", False),
        (" forms part of ", False),
    ],
    "depends_on": [
        (" depends on ", False),
        (" relies on ", False),
        (" reliant on ", False),
    ],
    "uses": [
        (" uses ", False),
        (" utilizes ", False),
        (" is used by ", True),
    ],
    "improves": [
        (" improves ", False),
        (" enhances ", False),
    ],
    "influences": [
        (" influences ", False),
        (" affects ", False),
        (" impacts ", False),
        (" increases ", False),
        (" decreases ", False),
    ],
    "similar_to": [
        (" similar to ", False),
        (" akin to ", False),
    ],
}

_SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent"}
_OBJECT_DEPS = {"dobj", "obj", "pobj", "iobj", "attr", "oprd", "obl"}


def _entity_has_dep(entity: EntityOccur, dep_set: set[str]) -> bool:
    labels = getattr(entity, "dep_labels", set()) or set()
    return any(label in dep_set for label in labels)

def _pick_head_tail_from_trigger(
    ents: list[EntityOccur],
    sent_text: str,
    sent_start: int,
    rel_type: str,
) -> tuple[EntityOccur | None, EntityOccur | None]:
    """
    Find head and tail entities based on trigger position in sentence.
    
    Args:
        ents: List of entities with chunk-global span positions
        sent_text: Sentence text (sentence-local)
        sent_start: Start position of sentence in chunk (for offset conversion)
        rel_type: Relation type string
        
    Returns:
        Tuple of (head_entity, tail_entity) or (None, None) if not found
    """
    s_lower = " " + sent_text.lower() + " "
    triggers = _RELATION_TRIGGERS.get(rel_type)
    if not triggers:
        return None, None

    # Find trigger in sentence-local coordinates
    local_trigger_start = None
    local_trigger_end = None
    reverse_direction = False
    for t, reverse in triggers:
        idx = s_lower.find(t)
        if idx != -1:
            local_trigger_start = idx
            local_trigger_end = idx + len(t)
            reverse_direction = reverse
            break

    if local_trigger_start is None:
        return None, None

    # Convert to chunk-global coordinates
    # s_lower has " " + sent_text + " ", so positions in s_lower need to be adjusted
    # by subtracting 1 to get positions in sent_text (accounting for leading space)
    sent_local_trigger_start = local_trigger_start - 1
    sent_local_trigger_end = local_trigger_end - 1
    
    # Now convert from sentence-local to chunk-global coordinates
    chunk_trigger_start = sent_start + sent_local_trigger_start
    chunk_trigger_end = sent_start + sent_local_trigger_end

    # Find head: entity ending before or at trigger start
    # Find tail: entity starting at or after trigger end
    head = max(
        (e for e in ents if e.span_end <= chunk_trigger_start),
        default=None,
        key=lambda e: e.span_end
    )
    tail = min(
        (e for e in ents if e.span_start >= chunk_trigger_end),
        default=None,
        key=lambda e: e.span_start
    )
    if reverse_direction:
        head, tail = tail, head

    return head, tail


def _pick_head_tail_from_dependencies(
    ents: list[EntityOccur],
) -> tuple[EntityOccur | None, EntityOccur | None]:
    """
    Use dependency roles to guess head/tail when trigger heuristics fail.
    """
    subj = next((e for e in ents if _entity_has_dep(e, _SUBJECT_DEPS)), None)
    obj = next((e for e in ents if _entity_has_dep(e, _OBJECT_DEPS)), None)
    if subj and obj and subj != obj:
        return subj, obj
    return None, None


def _entities_in_sentence(
    session: Session,
    chunk_id: int,
    sent_start: int,
    sent_end: int,
    doc,
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
        base_start = max(int(span_start), 0)
        base_end = span_end if span_end is not None else base_start + len(name or "")
        display_name = name
        dep_labels = set()
        effective_start = base_start
        effective_end = base_end
        if doc is not None:
            doc_span = doc.char_span(base_start, base_end, alignment_mode="expand")
            if doc_span is not None:
                effective_start = doc_span.start_char
                effective_end = doc_span.end_char
                surface = doc_span.text.strip()
                if surface:
                    display_name = surface
                dep_labels = {token.dep_.lower() for token in doc_span if token.dep_}

        if not (sent_start <= effective_start < sent_end):
            continue

        ents.append(
            EntityOccur(
                entity_id=eid,
                name=display_name,
                span_start=effective_start,
                span_end=effective_end,
                dep_labels=dep_labels,
            )
        )

    ents.sort(key=lambda e: e.span_start)
    ents = _dedupe_overlapping_entities(ents)
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
        doc = make_doc(chunk_text)
        sent_spans = [
            (sent.start_char, sent.end_char, sent.text.strip())
            for sent in doc.sents
            if sent.text.strip()
        ]
        if not sent_spans:
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
        ents = _entities_in_sentence(session, chunk.id, s_start, s_end, doc)
        ents = _filter_relation_entities(ents)
        if len(ents) < 2:
            continue

        sent_lower = " " + sent_text.lower() + " "
        rel_type = infer_relation_type(sent_lower)
        if rel_type is None:
            continue  # no recognizable relation pattern

        # Use the helper function that properly handles offset conversion
        head, tail = _pick_head_tail_from_trigger(ents, sent_text, s_start, rel_type)
        if head is None or tail is None:
            head, tail = _pick_head_tail_from_dependencies(ents)
        if head is None or tail is None:
            continue  # couldn't find head/tail

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
    batch_size: int = 100,
    commit_interval: int = 100,
):
    """
    Iterate over all claims above a confidence threshold and try to
    create/update Relation triples.
    
    Processes claims in batches to avoid memory issues and long-running transactions.
    
    Args:
        session: Database session
        min_claim_confidence: Minimum confidence threshold for claims
        batch_size: Number of claims to fetch from DB at once (default: 100)
        commit_interval: Number of claims to process before committing (default: 100)
    """
    import sys
    
    # Count total claims for progress tracking
    total_count = session.execute(
        select(func.count(Claim.id))
        .where(Claim.confidence >= min_claim_confidence)
    ).scalar()
    
    if total_count == 0:
        print("No claims found matching confidence threshold.")
        return
    
    print(f"Found {total_count} claims to process (confidence >= {min_claim_confidence}).")
    print(f"Processing in batches of {batch_size}, committing every {commit_interval} claims...")
    
    processed = 0
    errors = 0
    offset = 0
    
    # Process claims in batches using LIMIT/OFFSET to avoid cursor issues
    while offset < total_count:
        # Fetch a batch of claims
        batch = list(
            session.execute(
                select(Claim)
                .where(Claim.confidence >= min_claim_confidence)
                .order_by(Claim.id)
                .limit(batch_size)
                .offset(offset)
            ).scalars()
        )
        
        if not batch:
            break
        
        # Process each claim in the batch
        for claim in batch:
            nested_txn = session.begin_nested()
            try:
                link_relations_for_claim(session, claim)
                nested_txn.commit()
                processed += 1
                
                # Commit periodically to avoid losing all work and improve performance
                if processed % commit_interval == 0:
                    session.commit()
                    print(f"Processed {processed}/{total_count} claims (committed)...")
                    
            except Exception as e:
                errors += 1
                print(f"Error processing claim {claim.id}: {e}", file=sys.stderr)
                # Roll back only the current claim's work
                nested_txn.rollback()
                # Continue processing other claims
                continue
        
        offset += batch_size
        
        # Clear the batch from memory
        del batch
    
    # Final commit for any remaining claims
    if processed % commit_interval != 0:
        session.commit()
        print(f"Final commit: processed {processed} claims total.")
    
    print(f"\nCompleted: {processed} claims processed, {errors} errors encountered.")
