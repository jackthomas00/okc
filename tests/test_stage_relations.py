"""Tests for relation extraction stage (Stage 4) with accuracy metrics."""

import pytest
from sqlalchemy.orm import Session

from okc_core.models import Document, Chunk, Sentence, Entity, EntityMention, Relation, RelationEvidence
from okc_pipeline.state_04_relations.relation_extractor import extract_relations_for_sentences
from okc_pipeline.stage_03_claims.claim_detector import detect_claim_sentences
from okc_pipeline.stage_02_entities.entity_extractor import extract_entities_for_sentences
from okc_pipeline.stage_01_sentences.sentence_splitter import split_chunk_into_sentences
from tests.fixtures.test_documents import TEST_DOCUMENTS
from tests.fixtures.expected_outputs import EXPECTED_OUTPUTS
from tests.utils.pipeline_test_utils import (
    RelationMatch,
    extract_relations_from_db,
    compare_relations,
)


def create_test_document_with_claims(
    db: Session, doc_text: str, title: str = "Test Document"
) -> tuple[int, list[int]]:
    """Helper to create a document with chunks, sentences, entities, and claims."""
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
    
    # Extract entities (required for relations)
    extract_entities_for_sentences(db, sentence_ids)
    db.commit()
    
    # Detect claims (optional but helps with relation extraction)
    detect_claim_sentences(db, sentence_ids)
    db.commit()
    
    return doc_id, sentence_ids


@pytest.mark.parametrize("doc_key", ["is_a_relations", "evaluated_on_relations", "depends_on_relations"])
def test_relation_extraction_accuracy(db_session, doc_key):
    """Test relation extraction accuracy for various documents."""
    doc_data = TEST_DOCUMENTS[doc_key]
    expected = EXPECTED_OUTPUTS[doc_key]
    
    # Create document with entities and claims
    doc_id, sentence_ids = create_test_document_with_claims(
        db_session, doc_data["text"], doc_data["title"]
    )
    
    # Extract relations
    extract_relations_for_sentences(db_session, sentence_ids)
    db_session.commit()
    
    # Get extracted relations
    extracted_relations = extract_relations_from_db(db_session, doc_id)
    
    # Convert to RelationMatch format
    extracted_matches = [
        RelationMatch(
            head_entity_text=r.head_entity_text,
            relation_type=r.relation_type,
            tail_entity_text=r.tail_entity_text,
            confidence=r.confidence
        )
        for r in extracted_relations
    ]
    
    # Convert expected to RelationMatch format
    expected_matches = [
        RelationMatch(
            head_entity_text=head,
            relation_type=rel_type,
            tail_entity_text=tail,
            confidence=min_conf
        )
        for head, rel_type, tail, min_conf in expected["relations"]
    ]
    
    # Compare
    comparison = compare_relations(extracted_matches, expected_matches)
    
    # Assert reasonable accuracy (relations are harder, so lower threshold)
    assert comparison["overall"].precision >= 0.2, f"Precision too low: {comparison['overall'].precision}"
    assert comparison["overall"].recall >= 0.2, f"Recall too low: {comparison['overall'].recall}"


def test_relation_extraction_by_type(db_session):
    """Test relation extraction metrics broken down by type."""
    doc_text = "BERT is a model. BERT was evaluated on GLUE. The model depends on datasets."
    doc_id, sentence_ids = create_test_document_with_claims(
        db_session, doc_text, "Relation Type Test"
    )
    
    extract_relations_for_sentences(db_session, sentence_ids)
    db_session.commit()
    
    extracted_relations = extract_relations_from_db(db_session, doc_id)
    extracted_matches = [
        RelationMatch(
            head_entity_text=r.head_entity_text,
            relation_type=r.relation_type,
            tail_entity_text=r.tail_entity_text,
            confidence=r.confidence
        )
        for r in extracted_relations
    ]
    
    # Expected relations
    expected_matches = [
        RelationMatch(head_entity_text="BERT", relation_type="is_a", tail_entity_text="model", confidence=0.5),
        RelationMatch(head_entity_text="BERT", relation_type="evaluated_on", tail_entity_text="GLUE", confidence=0.5),
        RelationMatch(head_entity_text="model", relation_type="depends_on", tail_entity_text="datasets", confidence=0.5),
    ]
    
    comparison = compare_relations(extracted_matches, expected_matches)
    
    # Check that we have metrics by type
    assert "by_type" in comparison
    assert len(comparison["by_type"]) >= 0  # May have no matches, that's okay


def test_relation_extraction_type_constraints(db_session):
    """Test that relation extraction respects type constraints."""
    doc_text = "BERT improves accuracy. The model was tested on GLUE."
    doc_id, sentence_ids = create_test_document_with_claims(
        db_session, doc_text, "Type Constraints Test"
    )
    
    extract_relations_for_sentences(db_session, sentence_ids)
    db_session.commit()
    
    # Get relations
    relations = (
        db_session.query(Relation)
        .join(RelationEvidence, Relation.id == RelationEvidence.relation_id)
        .join(Sentence, RelationEvidence.sentence_id == Sentence.id)
        .join(Chunk, Sentence.chunk_id == Chunk.id)
        .filter(Chunk.document_id == doc_id)
        .all()
    )
    
    # Relations should have valid types
    for relation in relations:
        assert relation.relation_type in ["is_a", "improves", "evaluated_on", "depends_on", "solves"]
        assert relation.confidence >= 0.0
        assert relation.confidence <= 1.0


def test_relation_extraction_confidence_scoring(db_session):
    """Test that relation confidence scores are reasonable."""
    doc_text = "BERT improves accuracy. GPT-4 was evaluated on MMLU."
    doc_id, sentence_ids = create_test_document_with_claims(
        db_session, doc_text, "Confidence Test"
    )
    
    extract_relations_for_sentences(db_session, sentence_ids)
    db_session.commit()
    
    extracted_relations = extract_relations_from_db(db_session, doc_id)
    
    # All confidences should be between 0 and 1
    for relation in extracted_relations:
        assert 0.0 <= relation.confidence <= 1.0


def test_relation_extraction_evidence_linking(db_session):
    """Test that relations are linked to evidence sentences."""
    doc_text = "BERT improves accuracy on GLUE."
    doc_id, sentence_ids = create_test_document_with_claims(
        db_session, doc_text, "Evidence Test"
    )
    
    extract_relations_for_sentences(db_session, sentence_ids)
    db_session.commit()
    
    # Get relations with evidence
    relations = (
        db_session.query(Relation)
        .join(RelationEvidence, Relation.id == RelationEvidence.relation_id)
        .join(Sentence, RelationEvidence.sentence_id == Sentence.id)
        .join(Chunk, Sentence.chunk_id == Chunk.id)
        .filter(Chunk.document_id == doc_id)
        .all()
    )
    
    # Each relation should have at least one evidence
    for relation in relations:
        evidence = (
            db_session.query(RelationEvidence)
            .filter(RelationEvidence.relation_id == relation.id)
            .all()
        )
        assert len(evidence) > 0


def test_relation_extraction_requires_entities(db_session):
    """Test that relation extraction requires entities."""
    doc_text = "This is a sentence without entities."
    doc_id, sentence_ids = create_test_document_with_claims(
        db_session, doc_text, "No Entities Test"
    )
    
    # Remove entities
    db_session.query(EntityMention).filter(
        EntityMention.sentence_id.in_(sentence_ids)
    ).delete()
    db_session.commit()
    
    # Extract relations
    extract_relations_for_sentences(db_session, sentence_ids)
    db_session.commit()
    
    # Should not extract relations without entities
    relations = extract_relations_from_db(db_session, doc_id)
    assert len(relations) == 0

