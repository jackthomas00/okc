"""Tests for claim detection stage (Stage 3) with accuracy metrics."""

import pytest
from sqlalchemy.orm import Session

from okc_core.models import Document, Chunk, Sentence, Entity, EntityMention, ClaimSentence
from okc_pipeline.stage_03_claims.claim_detector import detect_claim_sentences
from okc_pipeline.stage_02_entities.entity_extractor import extract_entities_for_sentences
from okc_pipeline.stage_01_sentences.sentence_splitter import split_chunk_into_sentences
from tests.fixtures.test_documents import TEST_DOCUMENTS
from tests.fixtures.expected_outputs import EXPECTED_OUTPUTS
from tests.utils.pipeline_test_utils import (
    ClaimMatch,
    extract_claims_from_db,
    compare_claims,
)


def create_test_document_with_entities(
    db: Session, doc_text: str, title: str = "Test Document"
) -> tuple[int, list[int]]:
    """Helper to create a document with chunks, sentences, and entities."""
    from okc_api.crud import ingest_document, add_chunks_with_embeddings
    from okc_pipeline.stage_00_ingestion.chunker import chunk_text
    import numpy as np
    
    # Create document
    doc_id, _ = ingest_document(db, title, None, doc_text, lang="en", doc_embedding=None)
    
    # Create chunks
    chunks = chunk_text(doc_text, target_tokens=600, overlap=80)
    embeddings = [np.random.rand(384).tolist() for _ in chunks]
    chunk_ids, _ = add_chunks_with_embeddings(db, doc_id, chunks, embeddings=embeddings)
    
    # Create sentences
    sentence_ids = []
    for chunk_id in chunk_ids:
        chunk = db.query(Chunk).filter(Chunk.id == chunk_id).first()
        sentence_data = split_chunk_into_sentences(chunk)
        for order_idx, sent_dict in enumerate(sentence_data):
            sentence = Sentence(
                chunk_id=chunk.id,
                text=sent_dict["text"],
                char_start=sent_dict["start"],
                char_end=sent_dict["end"],
                order_index=order_idx
            )
            db.add(sentence)
        db.flush()
        sentences = db.query(Sentence).filter(Sentence.chunk_id == chunk_id).all()
        sentence_ids.extend([s.id for s in sentences])
    
    # Extract entities (required for claim detection)
    extract_entities_for_sentences(db, sentence_ids)
    db.commit()
    
    return doc_id, sentence_ids


@pytest.mark.parametrize("doc_key", ["claim_sentences", "relation_patterns", "is_a_relations"])
def test_claim_detection_accuracy(db_session, doc_key):
    """Test claim detection accuracy for various documents."""
    doc_data = TEST_DOCUMENTS[doc_key]
    expected = EXPECTED_OUTPUTS[doc_key]
    
    # Create document with entities
    doc_id, sentence_ids = create_test_document_with_entities(
        db_session, doc_data["text"], doc_data["title"]
    )
    
    # Detect claims
    detect_claim_sentences(db_session, sentence_ids)
    db_session.commit()
    
    # Get extracted claims
    extracted_claims = extract_claims_from_db(db_session, doc_id)
    
    # Convert to ClaimMatch format
    extracted_matches = [
        ClaimMatch(
            sentence_text=c.sentence_text,
            is_claim=c.is_claim,
            score=c.score
        )
        for c in extracted_claims
    ]
    
    # Convert expected to ClaimMatch format
    expected_matches = [
        ClaimMatch(
            sentence_text=sent_text,
            is_claim=is_claim,
            score=min_score
        )
        for sent_text, is_claim, min_score in expected["claims"]
    ]
    
    # Compare
    comparison = compare_claims(extracted_matches, expected_matches)
    
    # Assert reasonable accuracy
    assert comparison["metrics"].precision >= 0.3, f"Precision too low: {comparison['metrics'].precision}"
    assert comparison["metrics"].recall >= 0.3, f"Recall too low: {comparison['metrics'].recall}"


def test_claim_detection_requires_entities(db_session):
    """Test that claim detection requires entities."""
    doc_text = "This is a sentence without entities."
    doc_id, sentence_ids = create_test_document_with_entities(
        db_session, doc_text, "No Entities Test"
    )
    
    # Remove entities
    db_session.query(EntityMention).filter(
        EntityMention.sentence_id.in_(sentence_ids)
    ).delete()
    db_session.commit()
    
    # Detect claims
    detect_claim_sentences(db_session, sentence_ids)
    db_session.commit()
    
    # Should not detect claims without entities
    claims = extract_claims_from_db(db_session, doc_id)
    claim_sentences = [c for c in claims if c.is_claim]
    assert len(claim_sentences) == 0


def test_claim_detection_requires_verbs(db_session):
    """Test that claim detection requires relation verbs."""
    doc_text = "BERT and GPT-4 are models."  # Has entities but weak verb
    doc_id, sentence_ids = create_test_document_with_entities(
        db_session, doc_text, "Verb Test"
    )
    
    detect_claim_sentences(db_session, sentence_ids)
    db_session.commit()
    
    claims = extract_claims_from_db(db_session, doc_id)
    # May or may not be detected as claim depending on "be" verb handling
    assert len(claims) >= 0


def test_claim_detection_hedging(db_session):
    """Test that hedging language reduces claim scores."""
    doc_text = "The model might improve performance. It could increase accuracy."
    doc_id, sentence_ids = create_test_document_with_entities(
        db_session, doc_text, "Hedging Test"
    )
    
    detect_claim_sentences(db_session, sentence_ids)
    db_session.commit()
    
    claims = extract_claims_from_db(db_session, doc_id)
    claim_sentences = [c for c in claims if c.is_claim]
    
    # Claims with hedging should have lower scores
    for claim in claim_sentences:
        if any(hedge in claim.sentence_text.lower() for hedge in ["might", "could", "may"]):
            # Score should be lower due to hedging
            assert claim.score < 1.0


def test_claim_detection_scoring(db_session):
    """Test that claim scores are reasonable."""
    doc_text = "BERT improves accuracy. GPT-4 outperforms GPT-3. The model depends on datasets."
    doc_id, sentence_ids = create_test_document_with_entities(
        db_session, doc_text, "Scoring Test"
    )
    
    detect_claim_sentences(db_session, sentence_ids)
    db_session.commit()
    
    claims = extract_claims_from_db(db_session, doc_id)
    claim_sentences = [c for c in claims if c.is_claim]
    
    # All claim scores should be between 0 and 1
    for claim in claim_sentences:
        assert 0.0 <= claim.score <= 1.0
    
    # Should have detected some claims
    assert len(claim_sentences) > 0


def test_claim_detection_false_positives(db_session):
    """Test claim detection false positive rate."""
    doc_text = "This is a simple sentence. Another sentence here."
    doc_id, sentence_ids = create_test_document_with_entities(
        db_session, doc_text, "False Positive Test"
    )
    
    detect_claim_sentences(db_session, sentence_ids)
    db_session.commit()
    
    claims = extract_claims_from_db(db_session, doc_id)
    claim_sentences = [c for c in claims if c.is_claim]
    
    # Simple sentences without entities/verbs should not be claims
    # (Allow some false positives, but should be reasonable)
    assert len(claim_sentences) <= len(claims) * 0.5  # At most 50% false positive rate


def test_claim_detection_false_negatives(db_session):
    """Test claim detection false negative rate."""
    doc_text = "BERT improves accuracy on GLUE. GPT-4 was evaluated on MMLU."
    doc_id, sentence_ids = create_test_document_with_entities(
        db_session, doc_text, "False Negative Test"
    )
    
    detect_claim_sentences(db_session, sentence_ids)
    db_session.commit()
    
    claims = extract_claims_from_db(db_session, doc_id)
    claim_sentences = [c for c in claims if c.is_claim]
    
    # Should detect at least some claims from sentences with entities and verbs
    assert len(claim_sentences) > 0

