# pipeline/utils/text_cleaning.py
from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import re

try:
    import spacy
except ImportError:
    spacy = None


@lru_cache(maxsize=1)
def _get_nlp():
    """
    Lazy-load spaCy model once per process.

    You need to install:
        pip install spacy
        python -m spacy download en_core_web_sm
    """
    if spacy is None:
        raise RuntimeError("spaCy is not installed. `pip install spacy` and download a model.")

    # We care mostly about sents / deps / noun_chunks.
    # Disable heavy stuff we don't need.
    return spacy.load(
        "en_core_web_sm",
        disable=["ner", "lemmatizer", "textcat"],
    )


# Fallback regex splitter if spaCy is missing or fails
_SENT_SPLIT = re.compile(r"(?<=[\.!?])\s+")
_ABBREV_ENDINGS = ("e.g.", "i.e.", "etc.", "U.S.", "U.K.", "U.N.", "Dr.", "Mr.", "Mrs.")


def _regex_sentence_spans(text: str) -> List[Tuple[int, int, str]]:
    """
    Basic regex-based fallback: (start, end, sentence_text).
    Slightly smarter about abbreviations than the original version.
    """
    raw = [s for s in _SENT_SPLIT.split(text) if s]
    if not raw:
        return []

    merged: list[str] = []
    cursor = 0
    spans: list[Tuple[int, int, str]] = []

    # First just merge tokens around abbreviations
    for s in raw:
        s_strip = s.strip()
        if merged and merged[-1].strip().endswith(_ABBREV_ENDINGS):
            merged[-1] = merged[-1] + " " + s_strip
        else:
            merged.append(s_strip)

    for s in merged:
        idx = text.find(s, cursor)
        if idx == -1:
            idx = text.find(s)
            if idx == -1:
                continue
        start = idx
        end = idx + len(s)
        spans.append((start, end, s))
        cursor = end

    return spans


def sentence_spans(text: str) -> List[Tuple[int, int, str]]:
    """
    Return sentence spans as (start_char, end_char, sentence_text).

    Primary: spaCy's sentence parser.
    Fallback: regex-based splitter if spaCy isn't available.
    """
    text = text or ""
    if not text.strip():
        return []

    # Try spaCy first
    if spacy is not None:
        try:
            nlp = _get_nlp()
            doc = nlp(text)
            spans = []
            for sent in doc.sents:
                s_text = sent.text.strip()
                if not s_text:
                    continue
                spans.append((sent.start_char, sent.end_char, s_text))
            if spans:
                return spans
        except Exception:
            # Fall back gracefully if model or parsing explodes
            pass

    # Fallback
    return _regex_sentence_spans(text)


def split_sentences(text: str) -> list[str]:
    """
    Convenience wrapper: just the texts.
    """
    return [s for _, _, s in sentence_spans(text)]


def canonicalize(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).lower()

