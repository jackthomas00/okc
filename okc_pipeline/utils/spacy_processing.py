from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import spacy
from spacy.language import Language
from spacy.tokens import Doc

SPACY_MODEL_CANDIDATES = [
    "en_core_web_sm",
    "en_core_web_md",
    "en_core_web_lg",
]


@lru_cache(maxsize=1)
def get_spacy_model() -> Language:
    """
    Load and cache a spaCy Language pipeline.
    Tries progressively larger English models, falling back to a blank model.
    """
    last_error: Exception | None = None
    for name in SPACY_MODEL_CANDIDATES:
        try:
            # We only need tagger + parser for sents/noun chunks; drop NER for perf.
            return spacy.load(name, exclude=["ner"])
        except OSError as exc:  # model not installed locally
            last_error = exc
            continue

    # Fall back to a blank English model with sentencizer if the pre-trained
    # packages are unavailable. This keeps the system usable in CI.
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


@lru_cache(maxsize=1)
def get_spacy_model_with_ner() -> Language:
    """
    Load and cache a spaCy Language pipeline WITH NER enabled.
    Tries progressively larger English models, falling back to a blank model.
    Used for Stage 2 entity extraction.
    """
    last_error: Exception | None = None
    for name in SPACY_MODEL_CANDIDATES:
        try:
            # Load with NER enabled for entity extraction
            return spacy.load(name)
        except OSError as exc:  # model not installed locally
            last_error = exc
            continue

    # Fall back to a blank English model with sentencizer if the pre-trained
    # packages are unavailable. Note: blank models don't have NER, so this is
    # a degraded fallback.
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def make_doc(text: str) -> Doc:
    """Process text with the cached spaCy model."""
    nlp = get_spacy_model()
    return nlp(text)


def make_doc_with_ner(text: str) -> Doc:
    """Process text with the cached spaCy model that includes NER."""
    nlp = get_spacy_model_with_ner()
    return nlp(text)


@dataclass(frozen=True)
class SpanInfo:
    text: str
    start: int
    end: int
    kind: str  # e.g., "noun_chunk", "proper_noun"

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


def sentence_spans_from_doc(doc: Doc) -> list[tuple[int, int, str]]:
    """Extract sentence spans (start, end, text) from a spaCy Doc."""
    spans: list[tuple[int, int, str]] = []
    for sent in doc.sents:
        text = sent.text.strip()
        if not text:
            continue
        start = sent.start_char
        end = sent.end_char
        spans.append((start, end, text))
    return spans


def noun_chunk_spans(doc: Doc) -> list[SpanInfo]:
    """
    Collect noun chunk spans if the Doc has dependency parses.
    spaCy raises ValueError for noun_chunks without parser, so guard carefully.
    """
    chunks: list[SpanInfo] = []
    try:
        for chunk in doc.noun_chunks:
            text = chunk.text.strip()
            if not text:
                continue
            chunks.append(
                SpanInfo(
                    text=text,
                    start=chunk.start_char,
                    end=chunk.end_char,
                    kind="noun_chunk",
                )
            )
    except ValueError:
        # Parser not available (likely blank model), so noun_chunks aren't usable.
        pass
    return chunks


def proper_noun_spans(doc: Doc) -> list[SpanInfo]:
    """
    Build spans for contiguous PROPN tokens (allowing hyphen connectors).
    Useful when noun_chunks miss title-case names and we still want the phrase.
    """
    spans: list[SpanInfo] = []
    start_idx: int | None = None
    end_idx: int | None = None
    last_was_propn = False

    def flush():
        nonlocal start_idx, end_idx, last_was_propn
        if start_idx is None or end_idx is None:
            return
        span = doc[start_idx:end_idx]
        text = span.text.strip()
        if text:
            spans.append(
                SpanInfo(
                    text=text,
                    start=span.start_char,
                    end=span.end_char,
                    kind="proper_noun",
                )
            )
        start_idx = None
        end_idx = None
        last_was_propn = False

    for token in doc:
        if token.pos_ == "PROPN" or (token.text in {"-", "–"} and last_was_propn):
            if start_idx is None:
                start_idx = token.i
            end_idx = token.i + 1
            last_was_propn = token.pos_ == "PROPN"
        else:
            flush()

    flush()
    return spans


def _best_covering_span(spans: Sequence[SpanInfo], start: int, end: int) -> SpanInfo | None:
    """
    Pick the smallest-span that fully covers [start, end).
    """
    covering = [
        span for span in spans
        if span.start <= start and span.end >= end
    ]
    if not covering:
        return None
    return min(covering, key=lambda s: s.length)


def merge_entity_candidates(
    text: str,
    candidates: Iterable[str],
    doc: Doc | None = None,
) -> list[SpanInfo]:
    """
    Combine heuristic entity candidates with spaCy noun chunks/proper nouns.

    - Expand regex candidates to the noun_chunk that contains them (if any).
    - Add additional proper-noun spans spaCy found but regex missed.
    """
    doc = doc or make_doc(text)
    noun_spans = noun_chunk_spans(doc)
    proper_spans = proper_noun_spans(doc)
    merged: list[SpanInfo] = []
    seen_ranges: set[tuple[int, int]] = set()

    for cand in candidates:
        if not cand:
            continue
        cand = cand.strip()
        if not cand:
            continue

        idx = text.find(cand)
        if idx == -1:
            continue
        start = idx
        end = idx + len(cand)

        expanded = _best_covering_span(noun_spans, start, end)
        if expanded is not None and expanded.length > 0:
            start = expanded.start
            end = expanded.end
            cand_text = expanded.text
        else:
            cand_text = cand

        cand_text = cand_text.strip()
        if not cand_text:
            continue

        key = (start, end)
        if key in seen_ranges:
            continue
        seen_ranges.add(key)
        merged.append(SpanInfo(text=cand_text, start=start, end=end, kind="candidate"))

    for span in proper_spans:
        key = (span.start, span.end)
        if key in seen_ranges:
            continue
        # Basic validation: drop all-lowercase phrases (likely not canonical names)
        if not span.text or span.text.islower():
            continue
        seen_ranges.add(key)
        merged.append(span)

    return merged
