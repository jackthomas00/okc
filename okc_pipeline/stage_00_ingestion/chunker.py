"""
Chunking functionality for Stage 0.

Splits documents into chunks with overlap, respecting sentence boundaries.
"""

import re

from okc_pipeline.utils.text_cleaning import split_sentences


def chunk_text(text: str, target_tokens: int = 600, overlap: int = 80) -> list[str]:
    """
    Split text into chunks with overlap, respecting sentence boundaries.
    
    Uses sentence splitting to ensure chunks don't break mid-sentence,
    which is important for downstream processing (Stage 1: sentence splitting).
    
    Args:
        text: The text to chunk
        target_tokens: Target number of tokens per chunk (default: 600)
        overlap: Number of tokens to overlap between chunks (default: 80)
        
    Returns:
        List of chunk text strings
    """
    TOKEN = re.compile(r"\w+|\S")

    sents = split_sentences(text)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        tokens = TOKEN.findall(s)
        if cur_len + len(tokens) > target_tokens and cur:
            chunks.append(" ".join(cur))
            # overlap from end of current chunk
            overlap_tokens = []
            total = 0
            for sent in reversed(cur):
                toks = TOKEN.findall(sent)
                total += len(toks)
                overlap_tokens.append(sent)
                if total >= overlap:
                    break
            cur = list(reversed(overlap_tokens))
            cur_len = sum(len(TOKEN.findall(x)) for x in cur)
        cur.append(s)
        cur_len += len(tokens)
    if cur:
        chunks.append(" ".join(cur))
    return chunks

