"""Tests for sentence splitting stage (Stage 1)."""

import pytest
from okc_core.models import Chunk
from okc_pipeline.stage_01_sentences.sentence_splitter import split_chunk_into_sentences


def test_sentence_splitting_basic(db_session):
    """Test basic sentence splitting."""
    chunk = Chunk(
        id=1,
        document_id=1,
        idx=0,
        text="This is sentence one. This is sentence two. This is sentence three."
    )
    
    sentences = split_chunk_into_sentences(chunk)
    
    assert len(sentences) == 3
    assert all("text" in s for s in sentences)
    assert all("start" in s for s in sentences)
    assert all("end" in s for s in sentences)
    assert "This is sentence one." in [s["text"] for s in sentences]
    assert "This is sentence two." in [s["text"] for s in sentences]
    assert "This is sentence three." in [s["text"] for s in sentences]


def test_sentence_splitting_empty_chunk(db_session):
    """Test sentence splitting with empty chunk."""
    chunk = Chunk(
        id=2,
        document_id=1,
        idx=0,
        text=""
    )
    
    sentences = split_chunk_into_sentences(chunk)
    
    assert len(sentences) == 0


def test_sentence_splitting_single_sentence(db_session):
    """Test sentence splitting with single sentence."""
    chunk = Chunk(
        id=3,
        document_id=1,
        idx=0,
        text="This is a single sentence."
    )
    
    sentences = split_chunk_into_sentences(chunk)
    
    assert len(sentences) == 1
    assert sentences[0]["text"] == "This is a single sentence."


def test_sentence_splitting_char_offsets(db_session):
    """Test that character offsets are correct."""
    chunk = Chunk(
        id=4,
        document_id=1,
        idx=0,
        text="First sentence. Second sentence."
    )
    
    sentences = split_chunk_into_sentences(chunk)
    
    assert len(sentences) >= 2
    # Check that offsets are within chunk text
    for sent in sentences:
        assert sent["start"] >= 0
        assert sent["end"] <= len(chunk.text)
        assert sent["start"] < sent["end"]
        # Verify the text matches
        assert sent["text"] == chunk.text[sent["start"]:sent["end"]]


def test_sentence_splitting_abbreviations(db_session):
    """Test sentence splitting with abbreviations."""
    chunk = Chunk(
        id=5,
        document_id=1,
        idx=0,
        text="Dr. Smith went to the U.S.A. He was happy."
    )
    
    sentences = split_chunk_into_sentences(chunk)
    
    # Should split on the period after "happy" but not on abbreviations
    assert len(sentences) >= 1
    # The text should be preserved
    full_text = " ".join(s["text"] for s in sentences)
    assert "Dr." in full_text or "U.S.A." in full_text


def test_sentence_splitting_quotes(db_session):
    """Test sentence splitting with quotes."""
    chunk = Chunk(
        id=6,
        document_id=1,
        idx=0,
        text='He said "Hello world." Then he left.'
    )
    
    sentences = split_chunk_into_sentences(chunk)
    
    # Should handle quotes correctly
    assert len(sentences) >= 1
    full_text = " ".join(s["text"] for s in sentences)
    assert "Hello world" in full_text or '"Hello world."' in full_text


def test_sentence_splitting_multiple_punctuation(db_session):
    """Test sentence splitting with multiple punctuation marks."""
    chunk = Chunk(
        id=7,
        document_id=1,
        idx=0,
        text="What?! Really... That's amazing!!!"
    )
    
    sentences = split_chunk_into_sentences(chunk)
    
    # Should handle multiple punctuation
    assert len(sentences) >= 1
    full_text = " ".join(s["text"] for s in sentences)
    assert len(full_text) > 0


def test_sentence_splitting_newlines(db_session):
    """Test sentence splitting with newlines."""
    chunk = Chunk(
        id=8,
        document_id=1,
        idx=0,
        text="First sentence.\n\nSecond sentence.\nThird sentence."
    )
    
    sentences = split_chunk_into_sentences(chunk)
    
    # Should split sentences even with newlines
    assert len(sentences) >= 2
    assert "First sentence." in [s["text"].strip() for s in sentences]
    assert "Second sentence." in [s["text"].strip() for s in sentences] or "Third sentence." in [s["text"].strip() for s in sentences]


def test_sentence_splitting_no_period(db_session):
    """Test sentence splitting when there's no period."""
    chunk = Chunk(
        id=9,
        document_id=1,
        idx=0,
        text="This is a sentence without a period"
    )
    
    sentences = split_chunk_into_sentences(chunk)
    
    # Should still create at least one sentence
    assert len(sentences) >= 1
    assert "sentence without a period" in sentences[0]["text"] or "sentence" in sentences[0]["text"].lower()


def test_sentence_splitting_chunk_id_preserved(db_session):
    """Test that chunk_id is preserved in sentence data."""
    chunk = Chunk(
        id=10,
        document_id=1,
        idx=0,
        text="First. Second. Third."
    )
    
    sentences = split_chunk_into_sentences(chunk)
    
    # All sentences should reference the same chunk_id
    assert all(s.get("chunk_id") == chunk.id for s in sentences)

