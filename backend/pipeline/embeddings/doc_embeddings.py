from __future__ import annotations

from typing import Sequence

import numpy as np

from pipeline.embeddings.embedder import embed_texts

DOC_EMBED_MAX_PARAGRAPHS = 6
DOC_EMBED_MIN_CHARS = 120
DOC_EMBED_CHAR_LIMIT = 2000


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def select_doc_segments(text: str) -> list[str]:
    """Pick a few representative paragraphs so we can build a doc-level vector quickly."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    slices: list[str] = []
    for para in paragraphs:
        normalized = _normalize_whitespace(para)
        if len(normalized) < DOC_EMBED_MIN_CHARS:
            continue
        slices.append(normalized[:DOC_EMBED_CHAR_LIMIT])
        if len(slices) >= DOC_EMBED_MAX_PARAGRAPHS:
            break
    if not slices:
        slices = [_normalize_whitespace(text)[:DOC_EMBED_CHAR_LIMIT]]
    return slices


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0 or not np.isfinite(norm):
        return vec
    return vec / norm


def compute_doc_embedding(text: str) -> np.ndarray:
    """Embed a few representative paragraphs and mean-pool them into a document vector."""
    segments = select_doc_segments(text)
    vecs = embed_texts(segments)
    if vecs.size == 0:
        raise ValueError("Failed to compute embedding; segments empty")
    pooled = vecs.mean(axis=0)
    return normalize_vector(pooled)


def aggregate_chunk_embeddings(chunk_embeddings: np.ndarray) -> np.ndarray:
    """Mean-pool chunk vectors so we can store an updated doc-level embedding."""
    if chunk_embeddings.size == 0:
        raise ValueError("chunk_embeddings is empty")
    pooled = chunk_embeddings.mean(axis=0)
    return normalize_vector(pooled)


def embedding_to_list(vec: np.ndarray | Sequence[float] | None) -> list[float] | None:
    if vec is None:
        return None
    if isinstance(vec, np.ndarray):
        return vec.astype(np.float32).tolist()
    return list(vec)
