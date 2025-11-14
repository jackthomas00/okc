# claims.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

from sqlalchemy import select
from sqlalchemy.orm import Session

from api.models import Chunk, EntityChunk, Entity, Claim, ClaimSource
from pipeline.utils.text_cleaning import sentence_spans


@dataclass
class EntityOccur:
    entity_id: int
    name: str
    span_start: int
    span_end: int

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
    if not text:
        return

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
        for i, (s_start, s_end, _s) in enumerate(sent_spans):
            if s_start <= span_start < s_end:
                sent_entities[i].append(
                    EntityOccur(
                        entity_id=eid,
                        name=name,
                        span_start=span_start,
                        span_end=span_end if span_end is not None else span_start + len(name),
                    )
                )
                break

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

        if not any(v in sent_lower for v in VERB_TRIGGERS):
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
