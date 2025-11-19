# claims.py
# NOTE: This module is for Milestone 2 (Claims & Relations) and is not used in Milestone 1.
# The Claim and ClaimSource models have been removed from the schema.
# This file is kept for reference but should not be imported or used until Milestone 2.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Set

from sqlalchemy import select
from sqlalchemy.orm import Session

from api.models import Chunk, EntityChunk, Entity, Claim, ClaimSource
from pipeline.utils.text_cleaning import sentence_spans
from pipeline.utils.spacy_processing import make_doc


@dataclass
class EntityOccur:
    entity_id: int
    name: str
    span_start: int
    span_end: int
    dep_labels: Set[str] = field(default_factory=set)


_SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent"}
_OBJECT_DEPS = {"dobj", "obj", "pobj", "iobj", "attr", "oprd", "obl"}


def _has_dep_role(labels: Set[str], dep_set: set[str]) -> bool:
    return any(label in dep_set for label in labels)

def extract_claims_for_chunk(
    session: Session,
    chunk: Chunk,
    min_entities_in_sentence: int = 2,
):
    """
    Heuristic claim extractor:
    - Split chunk into sentences.
    - Assign entity occurrences (EntityChunk) to sentences by span.
    - For sentences with >= 2 entities and a copula / relation verb, create Claim + ClaimSource.

    We do NOT deduplicate across chunks globally yet.
    """
    text = chunk.text or ""
    if not text.strip():
        return

    doc = make_doc(text)
    sent_spans: List[tuple[int, int, str]] = []
    sent_index_map: dict[int, int] = {}
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        idx = len(sent_spans)
        sent_spans.append((sent.start_char, sent.end_char, sent_text))
        sent_index_map[id(sent)] = idx

    if not sent_spans:
        sent_spans = sentence_spans(text)
        if not sent_spans:
            return

    # load entity occurrences for this chunk
    rows = session.execute(
        select(
            EntityChunk.span_start,
            EntityChunk.span_end,
            EntityChunk.entity_id,
            Entity.name,
        )
        .join(Entity, Entity.id == EntityChunk.entity_id)
        .where(EntityChunk.chunk_id == chunk.id)
    )

    sent_entities: Dict[int, List[EntityOccur]] = {i: [] for i in range(len(sent_spans))}

    for span_start, span_end, eid, name in rows:
        if span_start is None:
            continue
        base_start = max(int(span_start), 0)
        base_end = span_end if span_end is not None else base_start + len(name or "")
        doc_span = doc.char_span(base_start, base_end, alignment_mode="expand")
        effective_start = base_start
        effective_end = base_end
        display_name = name
        dep_labels: Set[str] = set()
        sent_idx = None

        if doc_span is not None:
            effective_start = doc_span.start_char
            effective_end = doc_span.end_char
            surface = doc_span.text.strip()
            if surface:
                display_name = surface
            dep_labels = {token.dep_.lower() for token in doc_span if token.dep_}
            sent_idx = sent_index_map.get(id(doc_span.sent))

        if sent_idx is None:
            for i, (s_start, s_end, _s) in enumerate(sent_spans):
                if s_start <= effective_start < s_end:
                    sent_idx = i
                    break

        if sent_idx is None:
            continue

        sent_entities[sent_idx].append(
            EntityOccur(
                entity_id=eid,
                name=display_name,
                span_start=effective_start,
                span_end=effective_end,
                dep_labels=dep_labels,
            )
        )

    # trigger verbs for “claims”
    VERB_TRIGGERS = [
        " is ", " are ", " was ", " were ",
        " causes ", " cause ",
        " leads to ", " lead to ",
        " depends on ", " reliant on ", " relies on ",
        " improves ", " enhances ", " increases ", " decreases ",
        " similar to ", " part of ",
        " uses ", " used by ",
    ]

    for i, (s_start, s_end, sent_text) in enumerate(sent_spans):
        ents = sent_entities.get(i) or []
        if len(ents) < min_entities_in_sentence:
            continue

        sent_lower = " " + sent_text.lower() + " "
        has_dep_pair = (
            any(_has_dep_role(e.dep_labels, _SUBJECT_DEPS) for e in ents) and
            any(_has_dep_role(e.dep_labels, _OBJECT_DEPS) for e in ents)
        )

        if not any(v in sent_lower for v in VERB_TRIGGERS) and not has_dep_pair:
            continue

        # avoid duplicate claim for same chunk + sentence text
        existing = session.execute(
            select(Claim.id)
            .join(ClaimSource, ClaimSource.claim_id == Claim.id)
            .where(Claim.text == sent_text)
            .where(ClaimSource.chunk_id == chunk.id)
        ).first()
        if existing:
            continue

        claim = Claim(
            text=sent_text,
            polarity="neutral",   # keep it simple for now
            confidence=0.6,       # arbitrary “medium” confidence
        )
        session.add(claim)
        session.flush()  # get claim.id

        cs = ClaimSource(
            claim_id=claim.id,
            document_id=chunk.document_id,
            chunk_id=chunk.id,
        )
        session.add(cs)
