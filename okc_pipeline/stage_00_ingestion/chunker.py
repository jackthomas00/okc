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
    If a single sentence exceeds the target token count, it will be split
    at token boundaries to respect the target.
    
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
    
    def flush_chunk():
        """Flush current chunk and set up overlap for next chunk."""
        nonlocal cur, cur_len
        if not cur:
            return
        chunks.append(" ".join(cur))
        # Set up overlap from the end of current chunk
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
    
    for s in sents:
        tokens = TOKEN.findall(s)
        sent_token_count = len(tokens)
        
        # If a single sentence exceeds target, split it into multiple chunks
        if sent_token_count > target_tokens:
            # First, flush any accumulated sentences
            if cur:
                flush_chunk()
            
            # Split the long sentence by grouping words to respect token limits
            words = s.split()
            word_idx = 0
            while word_idx < len(words):
                # Calculate how many tokens we can take
                remaining_tokens = target_tokens - cur_len
                if remaining_tokens <= 0:
                    # Start a new chunk
                    if cur:
                        flush_chunk()
                    remaining_tokens = target_tokens - cur_len
                
                # Group words until we reach the token limit
                chunk_words = []
                chunk_token_count = 0
                while word_idx < len(words):
                    word = words[word_idx]
                    word_tokens = TOKEN.findall(word)
                    word_token_count = len(word_tokens)
                    
                    # Check if adding this word would exceed the limit
                    if chunk_token_count + word_token_count > remaining_tokens and chunk_words:
                        break
                    
                    chunk_words.append(word)
                    chunk_token_count += word_token_count
                    word_idx += 1
                    
                    # Stop if we've reached the token limit
                    if chunk_token_count >= remaining_tokens:
                        break
                
                if chunk_words:
                    chunk_text_segment = " ".join(chunk_words)
                    cur.append(chunk_text_segment)
                    cur_len += chunk_token_count
        else:
            # Normal case: sentence fits or can be added to current chunk
            if cur_len + sent_token_count > target_tokens and cur:
                flush_chunk()
            cur.append(s)
            cur_len += sent_token_count
    
    if cur:
        chunks.append(" ".join(cur))
    return chunks

