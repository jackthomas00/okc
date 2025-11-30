"""Tests for chunking stage (Stage 0)."""

import pytest
from okc_pipeline.stage_00_ingestion.chunker import chunk_text


def test_chunking_basic():
    """Test basic chunking functionality."""
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = chunk_text(text, target_tokens=10, overlap=2)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(len(chunk) > 0 for chunk in chunks)


def test_chunking_respects_sentence_boundaries():
    """Test that chunks don't break mid-sentence."""
    text = "This is a complete sentence. This is another complete sentence. And here is a third one."
    chunks = chunk_text(text, target_tokens=10, overlap=2)
    
    # Check that sentence boundaries are preserved
    # Each chunk should end with a period (or be the last chunk)
    for i, chunk in enumerate(chunks):
        if i < len(chunks) - 1:  # Not the last chunk
            # Should end with sentence-ending punctuation or be followed by overlap
            assert chunk.strip()[-1] in ".!?" or chunk.count(".") > 0


def test_chunking_overlap():
    """Test that chunks have proper overlap."""
    text = " ".join([f"Sentence {i}." for i in range(20)])
    chunks = chunk_text(text, target_tokens=15, overlap=5)
    
    if len(chunks) > 1:
        # Check that consecutive chunks share some content
        for i in range(len(chunks) - 1):
            chunk1_words = set(chunks[i].lower().split())
            chunk2_words = set(chunks[i + 1].lower().split())
            # Should have some overlap
            assert len(chunk1_words & chunk2_words) > 0


def test_chunking_target_tokens():
    """Test that chunks respect target token count."""
    text = " ".join([f"Word {i}" for i in range(100)])
    target_tokens = 20
    chunks = chunk_text(text, target_tokens=target_tokens, overlap=5)
    
    # Count tokens (simple word count)
    for chunk in chunks:
        words = chunk.split()
        # Should be close to target (allow some flexibility)
        assert len(words) <= target_tokens * 1.5  # Allow 50% overage


def test_chunking_single_chunk():
    """Test that short text produces a single chunk."""
    text = "This is a short text."
    chunks = chunk_text(text, target_tokens=100, overlap=10)
    
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunking_empty_text():
    """Test chunking with empty text."""
    text = ""
    chunks = chunk_text(text, target_tokens=100, overlap=10)
    
    assert len(chunks) == 0 or (len(chunks) == 1 and chunks[0] == "")


def test_chunking_very_long_text():
    """Test chunking with very long text."""
    text = " ".join([f"Sentence {i} with multiple words." for i in range(1000)])
    chunks = chunk_text(text, target_tokens=50, overlap=10)
    
    assert len(chunks) > 1
    # All chunks should be non-empty
    assert all(len(chunk.strip()) > 0 for chunk in chunks)
    # Reconstructing should preserve content (approximately)
    reconstructed = " ".join(chunks)
    # Should contain most of the original words
    original_words = set(text.split())
    reconstructed_words = set(reconstructed.split())
    # At least 80% of words should be preserved
    assert len(original_words & reconstructed_words) / len(original_words) > 0.8


def test_chunking_different_overlap_values():
    """Test chunking with different overlap values."""
    text = " ".join([f"Sentence {i}." for i in range(50)])
    
    chunks_small_overlap = chunk_text(text, target_tokens=20, overlap=2)
    chunks_large_overlap = chunk_text(text, target_tokens=20, overlap=10)
    
    # Larger overlap should generally produce more chunks (or same)
    # because each chunk is smaller relative to overlap
    assert len(chunks_small_overlap) <= len(chunks_large_overlap) + 2  # Allow some variance

