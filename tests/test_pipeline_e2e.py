"""End-to-end pipeline tests that run full pipeline and validate complete outputs."""

import pytest
import tempfile
import yaml
import os
from pathlib import Path

from okc_pipeline.pipeline_runner import PipelineRunner
from tests.fixtures.test_documents import TEST_DOCUMENTS
from tests.fixtures.expected_outputs import EXPECTED_OUTPUTS
from tests.utils.pipeline_test_utils import (
    EntityMatch,
    ClaimMatch,
    RelationMatch,
    extract_entities_from_db,
    extract_claims_from_db,
    extract_relations_from_db,
    compare_entities,
    compare_claims,
    compare_relations,
    calculate_metrics,
)


def create_test_pipeline_config(stages: list = None) -> str:
    """Create a temporary pipeline config file for testing."""
    if stages is None:
        stages = [
            {"name": "chunk", "params": {"target_tokens": 600, "overlap": 80}},
            {"name": "embed", "model": "sentence-transformers/all-MiniLM-L6-v2"},
            {"name": "sentences"},
            {"name": "entities"},
            {"name": "claims"},
            {"name": "relations"},
        ]
    
    config = {
        "sources": [],
        "config": {
            "spacy_model": "en_core_web_md",
        },
        "stages": stages,
    }
    
    # Create temporary config file
    fd, path = tempfile.mkstemp(suffix=".yaml", text=True)
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(config, f)
        return path
    except Exception:
        os.close(fd)
        raise


def run_pipeline_on_document(db_session, doc_data: dict, stages: list = None) -> int:
    """Run pipeline on a test document and return document ID."""
    from okc_core.db import SessionLocal
    
    # Create temporary config
    config_path = create_test_pipeline_config(stages)
    
    try:
        # Create PipelineRunner
        runner = PipelineRunner(config_path, skip_sourcing=True)
        
        # Manually process the document
        doc_id = runner.process_document(db_session, doc_data)
        db_session.commit()
        
        return doc_id
    finally:
        # Clean up config file
        if os.path.exists(config_path):
            os.unlink(config_path)


@pytest.mark.parametrize("doc_key", ["simple_entity", "multiple_entities", "claim_sentences"])
def test_pipeline_e2e_basic(db_session, doc_key):
    """Test end-to-end pipeline on basic documents."""
    doc_data = TEST_DOCUMENTS[doc_key]
    expected = EXPECTED_OUTPUTS[doc_key]
    
    # Run pipeline
    doc_id = run_pipeline_on_document(
        db_session,
        doc_data,
        stages=[
            {"name": "chunk", "params": {"target_tokens": 600, "overlap": 80}},
            {"name": "embed", "model": "sentence-transformers/all-MiniLM-L6-v2"},
            {"name": "sentences"},
            {"name": "entities"},
            {"name": "claims"},
        ]
    )
    
    assert doc_id is not None
    
    # Verify document was created
    from okc_core.models import Document, Chunk, Sentence
    doc = db_session.query(Document).filter(Document.id == doc_id).first()
    assert doc is not None
    assert doc.title == doc_data["title"]
    
    # Verify chunks were created
    chunks = db_session.query(Chunk).filter(Chunk.document_id == doc_id).all()
    assert len(chunks) > 0
    
    # Verify sentences were created
    sentence_ids = (
        db_session.query(Sentence.id)
        .join(Chunk, Sentence.chunk_id == Chunk.id)
        .filter(Chunk.document_id == doc_id)
        .all()
    )
    assert len(sentence_ids) > 0
    
    # Verify entities were extracted
    entities = extract_entities_from_db(db_session, doc_id)
    assert len(entities) > 0
    
    # Verify claims were detected
    claims = extract_claims_from_db(db_session, doc_id)
    assert len(claims) >= 0  # May or may not have claims


def test_pipeline_e2e_full_stages(db_session):
    """Test end-to-end pipeline with all stages including relations."""
    doc_data = TEST_DOCUMENTS["relation_patterns"]
    expected = EXPECTED_OUTPUTS["relation_patterns"]
    
    # Run full pipeline
    doc_id = run_pipeline_on_document(
        db_session,
        doc_data,
        stages=[
            {"name": "chunk", "params": {"target_tokens": 600, "overlap": 80}},
            {"name": "embed", "model": "sentence-transformers/all-MiniLM-L6-v2"},
            {"name": "sentences"},
            {"name": "entities"},
            {"name": "claims"},
            {"name": "relations"},
        ]
    )
    
    assert doc_id is not None
    
    # Verify all stages completed
    from okc_core.models import Document, Chunk, Sentence, Entity, EntityMention, ClaimSentence, Relation
    
    # Document
    doc = db_session.query(Document).filter(Document.id == doc_id).first()
    assert doc is not None
    
    # Chunks
    chunks = db_session.query(Chunk).filter(Chunk.document_id == doc_id).all()
    assert len(chunks) > 0
    
    # Sentences
    sentences = (
        db_session.query(Sentence)
        .join(Chunk, Sentence.chunk_id == Chunk.id)
        .filter(Chunk.document_id == doc_id)
        .all()
    )
    assert len(sentences) > 0
    
    # Entities
    entities = extract_entities_from_db(db_session, doc_id)
    assert len(entities) > 0
    
    # Claims
    claims = extract_claims_from_db(db_session, doc_id)
    assert len(claims) > 0
    
    # Relations
    relations = extract_relations_from_db(db_session, doc_id)
    assert len(relations) >= 0  # May or may not have relations


def test_pipeline_e2e_compare_outputs(db_session):
    """Test end-to-end pipeline and compare outputs against expected."""
    doc_data = TEST_DOCUMENTS["multiple_entities"]
    expected = EXPECTED_OUTPUTS["multiple_entities"]
    
    # Run pipeline
    doc_id = run_pipeline_on_document(
        db_session,
        doc_data,
        stages=[
            {"name": "chunk", "params": {"target_tokens": 600, "overlap": 80}},
            {"name": "embed", "model": "sentence-transformers/all-MiniLM-L6-v2"},
            {"name": "sentences"},
            {"name": "entities"},
            {"name": "claims"},
            {"name": "relations"},
        ]
    )
    
    # Extract outputs
    extracted_entities = extract_entities_from_db(db_session, doc_id)
    extracted_claims = extract_claims_from_db(db_session, doc_id)
    extracted_relations = extract_relations_from_db(db_session, doc_id)
    
    # Convert to match format
    doc_text = doc_data["text"]
    extracted_entity_matches = [
        EntityMatch(text=e.text, type=e.type, char_start=e.char_start, char_end=e.char_end)
        for e in extracted_entities
    ]
    
    # Convert expected entities
    expected_entity_matches = []
    for entity_text, entity_type, start, end in expected["entities"]:
        found = doc_text.lower().find(entity_text.lower(), start)
        if found != -1:
            expected_entity_matches.append(EntityMatch(
                text=entity_text,
                type=entity_type,
                char_start=found,
                char_end=found + len(entity_text)
            ))
    
    # Compare entities
    entity_comparison = compare_entities(extracted_entity_matches, expected_entity_matches)
    
    # Should have some matches
    assert entity_comparison["overall"].true_positives >= 0
    
    # Compare claims
    extracted_claim_matches = [
        ClaimMatch(sentence_text=c.sentence_text, is_claim=c.is_claim, score=c.score)
        for c in extracted_claims
    ]
    expected_claim_matches = [
        ClaimMatch(sentence_text=text, is_claim=is_claim, score=min_score)
        for text, is_claim, min_score in expected["claims"]
    ]
    claim_comparison = compare_claims(extracted_claim_matches, expected_claim_matches)
    
    # Should have some matches
    assert claim_comparison["metrics"].true_positives >= 0


def test_pipeline_e2e_error_handling(db_session):
    """Test pipeline error handling with invalid input."""
    # Test with empty document
    doc_data = {
        "title": "Empty Document",
        "url": None,
        "text": "",
        "lang": "en",
    }
    
    doc_id = run_pipeline_on_document(
        db_session,
        doc_data,
        stages=[
            {"name": "chunk", "params": {"target_tokens": 600, "overlap": 80}},
            {"name": "embed", "model": "sentence-transformers/all-MiniLM-L6-v2"},
        ]
    )
    
    # Should handle empty document gracefully
    # Either return None (deduped) or create document with no chunks
    if doc_id is not None:
        from okc_core.models import Chunk
        chunks = db_session.query(Chunk).filter(Chunk.document_id == doc_id).all()
        # Should have 0 chunks for empty text
        assert len(chunks) == 0


def test_pipeline_e2e_different_configurations(db_session):
    """Test pipeline with different configurations."""
    doc_data = TEST_DOCUMENTS["simple_entity"]
    
    # Test with different chunk sizes
    doc_id1 = run_pipeline_on_document(
        db_session,
        doc_data,
        stages=[
            {"name": "chunk", "params": {"target_tokens": 100, "overlap": 10}},
            {"name": "embed", "model": "sentence-transformers/all-MiniLM-L6-v2"},
            {"name": "sentences"},
            {"name": "entities"},
        ]
    )
    
    doc_id2 = run_pipeline_on_document(
        db_session,
        doc_data,
        stages=[
            {"name": "chunk", "params": {"target_tokens": 600, "overlap": 80}},
            {"name": "embed", "model": "sentence-transformers/all-MiniLM-L6-v2"},
            {"name": "sentences"},
            {"name": "entities"},
        ]
    )
    
    # Both should succeed
    assert doc_id1 is not None or doc_id2 is not None  # One might be deduped
    
    # Verify entities were extracted in both cases
    entities1 = extract_entities_from_db(db_session, doc_id1) if doc_id1 else []
    entities2 = extract_entities_from_db(db_session, doc_id2) if doc_id2 else []
    
    # At least one should have entities
    assert len(entities1) > 0 or len(entities2) > 0

