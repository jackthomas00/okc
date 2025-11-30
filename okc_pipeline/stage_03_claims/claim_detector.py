"""Claim sentence detection heuristics for Stage 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sqlalchemy.orm import Session, selectinload

from okc_core.models import ClaimSentence, Sentence
from okc_pipeline.utils.spacy_processing import make_doc

# A compact set of verb lemmas that usually express a relation between
# two typed entities. Most correspond to Stage 4 relation templates.
RELATION_VERB_LEMMAS: frozenset[str] = frozenset(
    [
        "be",
        "improve",
        "increase",
        "decrease",
        "boost",
        "reduce",
        "outperform",
        "train",
        "evaluate",
        "test",
        "use",
        "depend",
        "require",
        "need",
        "base",
        "build",
        "cause",
        "enable",
        "support",
        "apply",
        "compare",
    ]
)

# Terms that down-weight confidence because they hedge the statement.
HEDGING_LEMMAS: frozenset[str] = frozenset(
    [
        "might",
        "may",
        "could",
        "possibly",
        "suggest",
        "suggested",
        "appear",
        "seem",
        "approximately",
    ]
)


@dataclass(slots=True)
class ClaimDetectionResult:
    sentence_id: int
    is_claim: bool
    score: float
    matched_verbs: tuple[str, ...]
    hedge_terms: tuple[str, ...]
    mention_count: int


def _extract_verbs_and_hedges(text: str) -> tuple[set[str], set[str]]:
    """Return lemmas for relation verbs and hedging terms found in text."""
    if not text:
        return set(), set()
    doc = make_doc(text)
    verb_hits: set[str] = set()
    hedge_hits: set[str] = set()
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in RELATION_VERB_LEMMAS and token.pos_ in {"VERB", "AUX"}:
            verb_hits.add(lemma)
        if lemma in HEDGING_LEMMAS:
            hedge_hits.add(lemma)
    return verb_hits, hedge_hits


def _score_claim(mention_count: int, matched_verbs: Sequence[str], hedge_terms: Sequence[str]) -> float:
    """Compute a coarse confidence score between 0 and 1."""
    score = 0.0
    if mention_count >= 2:
        score += 0.6
    if mention_count >= 3:
        score += 0.1
    if matched_verbs:
        score += 0.4 + 0.05 * (len(matched_verbs) - 1)
    if hedge_terms:
        score -= 0.2
    return max(0.0, min(1.0, score))


def evaluate_sentence(sentence: Sentence, mention_count: int) -> ClaimDetectionResult:
    """Evaluate a single sentence for claim-ness using deterministic rules."""
    verb_hits, hedge_hits = _extract_verbs_and_hedges(sentence.text)
    is_claim = mention_count >= 2 and bool(verb_hits)
    score = _score_claim(mention_count, tuple(verb_hits), tuple(hedge_hits))
    return ClaimDetectionResult(
        sentence_id=sentence.id,
        is_claim=is_claim,
        score=score,
        matched_verbs=tuple(sorted(verb_hits)),
        hedge_terms=tuple(sorted(hedge_hits)),
        mention_count=mention_count,
    )


def detect_claim_sentences(session: Session, sentence_ids: Sequence[int]) -> dict[str, int]:
    """Detect claim sentences and persist ClaimSentence rows.

    Args:
        session: SQLAlchemy session.
        sentence_ids: Sentence primary keys to process.

    Returns:
        Simple stats about processed sentences.
    """
    if not sentence_ids:
        return {"sentences_processed": 0, "claims_detected": 0}

    sentences = (
        session.query(Sentence)
        .options(selectinload(Sentence.mentions))
        .filter(Sentence.id.in_(sentence_ids))
        .all()
    )

    existing_claims = {
        cs.sentence_id: cs
        for cs in session.query(ClaimSentence).filter(ClaimSentence.sentence_id.in_(sentence_ids))
    }

    stats = {"sentences_processed": 0, "claims_detected": 0}

    for sentence in sentences:
        mention_count = len(sentence.mentions)
        stats["sentences_processed"] += 1
        detection = evaluate_sentence(sentence, mention_count)
        stats["claims_detected"] += int(detection.is_claim)

        existing = existing_claims.get(sentence.id)
        details = {
            "matched_verbs": list(detection.matched_verbs),
            "hedge_terms": list(detection.hedge_terms),
            "mention_count": detection.mention_count,
        }
        if existing:
            existing.is_claim = detection.is_claim
            existing.score = detection.score
            existing.details = details
        else:
            session.add(
                ClaimSentence(
                    sentence_id=sentence.id,
                    is_claim=detection.is_claim,
                    score=detection.score,
                    details=details,
                )
            )

    session.flush()
    return stats
