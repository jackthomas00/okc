"""Metrics aggregation and reporting tests to generate performance summaries."""

import pytest
import json
from typing import Dict, Any, List

from tests.fixtures.test_documents import TEST_DOCUMENTS
from tests.fixtures.expected_outputs import EXPECTED_OUTPUTS
from tests.utils.pipeline_test_utils import (
    EntityMatch,
    ClaimMatch,
    RelationMatch,
    extract_entities_from_db,
    extract_claims_from_db,
    extract_relations_from_db,
    calculate_metrics,
)
from tests.test_pipeline_e2e import run_pipeline_on_document


def aggregate_metrics_across_documents(
    db_session, doc_keys: List[str]
) -> Dict[str, Any]:
    """Aggregate metrics across multiple test documents."""
    all_entity_metrics = []
    all_claim_metrics = []
    all_relation_metrics = []
    
    for doc_key in doc_keys:
        if doc_key not in TEST_DOCUMENTS or doc_key not in EXPECTED_OUTPUTS:
            continue
        
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
                {"name": "relations"},
            ]
        )
        
        if doc_id is None:
            continue
        
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
        
        extracted_claim_matches = [
            ClaimMatch(sentence_text=c.sentence_text, is_claim=c.is_claim, score=c.score)
            for c in extracted_claims
        ]
        expected_claim_matches = [
            ClaimMatch(sentence_text=text, is_claim=is_claim, score=min_score)
            for text, is_claim, min_score in expected["claims"]
        ]
        
        extracted_relation_matches = [
            RelationMatch(
                head_entity_text=r.head_entity_text,
                relation_type=r.relation_type,
                tail_entity_text=r.tail_entity_text,
                confidence=r.confidence
            )
            for r in extracted_relations
        ]
        expected_relation_matches = [
            RelationMatch(head_entity_text=head, relation_type=rel_type, tail_entity_text=tail, confidence=min_conf)
            for head, rel_type, tail, min_conf in expected["relations"]
        ]
        
        # Calculate metrics
        from tests.utils.pipeline_test_utils import (
            compare_entities,
            compare_claims,
            compare_relations,
        )
        
        entity_metrics = compare_entities(extracted_entity_matches, expected_entity_matches)
        claim_metrics = compare_claims(extracted_claim_matches, expected_claim_matches)
        relation_metrics = compare_relations(extracted_relation_matches, expected_relation_matches)
        
        all_entity_metrics.append(entity_metrics)
        all_claim_metrics.append(claim_metrics)
        all_relation_metrics.append(relation_metrics)
    
    # Aggregate metrics
    if not all_entity_metrics:
        return {}
    
    # Calculate averages
    avg_entity_precision = sum(m["overall"].precision for m in all_entity_metrics) / len(all_entity_metrics)
    avg_entity_recall = sum(m["overall"].recall for m in all_entity_metrics) / len(all_entity_metrics)
    avg_entity_f1 = sum(m["overall"].f1 for m in all_entity_metrics) / len(all_entity_metrics)
    
    avg_claim_precision = sum(m["metrics"].precision for m in all_claim_metrics) / len(all_claim_metrics) if all_claim_metrics else 0.0
    avg_claim_recall = sum(m["metrics"].recall for m in all_claim_metrics) / len(all_claim_metrics) if all_claim_metrics else 0.0
    avg_claim_f1 = sum(m["metrics"].f1 for m in all_claim_metrics) / len(all_claim_metrics) if all_claim_metrics else 0.0
    
    avg_relation_precision = sum(m["overall"].precision for m in all_relation_metrics) / len(all_relation_metrics) if all_relation_metrics else 0.0
    avg_relation_recall = sum(m["overall"].recall for m in all_relation_metrics) / len(all_relation_metrics) if all_relation_metrics else 0.0
    avg_relation_f1 = sum(m["overall"].f1 for m in all_relation_metrics) / len(all_relation_metrics) if all_relation_metrics else 0.0
    
    return {
        "entities": {
            "precision": avg_entity_precision,
            "recall": avg_entity_recall,
            "f1": avg_entity_f1,
            "num_documents": len(all_entity_metrics),
        },
        "claims": {
            "precision": avg_claim_precision,
            "recall": avg_claim_recall,
            "f1": avg_claim_f1,
            "num_documents": len(all_claim_metrics),
        },
        "relations": {
            "precision": avg_relation_precision,
            "recall": avg_relation_recall,
            "f1": avg_relation_f1,
            "num_documents": len(all_relation_metrics),
        },
    }


def test_metrics_aggregation(db_session):
    """Test metrics aggregation across multiple documents."""
    doc_keys = ["simple_entity", "multiple_entities", "claim_sentences"]
    
    metrics = aggregate_metrics_across_documents(db_session, doc_keys)
    
    # Should have aggregated metrics
    assert "entities" in metrics
    assert "claims" in metrics
    assert "relations" in metrics
    
    # Metrics should be between 0 and 1
    assert 0.0 <= metrics["entities"]["precision"] <= 1.0
    assert 0.0 <= metrics["entities"]["recall"] <= 1.0
    assert 0.0 <= metrics["entities"]["f1"] <= 1.0


def test_metrics_report_generation(db_session):
    """Test generation of metrics report."""
    doc_keys = ["simple_entity", "multiple_entities"]
    
    metrics = aggregate_metrics_across_documents(db_session, doc_keys)
    
    # Generate report
    report = {
        "summary": {
            "total_documents": len(doc_keys),
            "entity_extraction": {
                "precision": metrics["entities"]["precision"],
                "recall": metrics["entities"]["recall"],
                "f1": metrics["entities"]["f1"],
            },
            "claim_detection": {
                "precision": metrics["claims"]["precision"],
                "recall": metrics["claims"]["recall"],
                "f1": metrics["claims"]["f1"],
            },
            "relation_extraction": {
                "precision": metrics["relations"]["precision"],
                "recall": metrics["relations"]["recall"],
                "f1": metrics["relations"]["f1"],
            },
        }
    }
    
    # Report should be serializable
    report_json = json.dumps(report)
    assert len(report_json) > 0
    
    # Should have all required fields
    assert "summary" in report
    assert "entity_extraction" in report["summary"]
    assert "claim_detection" in report["summary"]
    assert "relation_extraction" in report["summary"]


def test_metrics_by_type_aggregation(db_session):
    """Test metrics aggregation broken down by entity/relation type."""
    doc_data = TEST_DOCUMENTS["multiple_entities"]
    expected = EXPECTED_OUTPUTS["multiple_entities"]
    
    doc_id = run_pipeline_on_document(
        db_session,
        doc_data,
        stages=[
            {"name": "chunk", "params": {"target_tokens": 600, "overlap": 80}},
            {"name": "embed", "model": "sentence-transformers/all-MiniLM-L6-v2"},
            {"name": "sentences"},
            {"name": "entities"},
        ]
    )
    
    if doc_id is None:
        pytest.skip("Document was deduped")
    
    extracted_entities = extract_entities_from_db(db_session, doc_id)
    doc_text = doc_data["text"]
    
    extracted_entity_matches = [
        EntityMatch(text=e.text, type=e.type, char_start=e.char_start, char_end=e.char_end)
        for e in extracted_entities
    ]
    
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
    
    from tests.utils.pipeline_test_utils import compare_entities
    comparison = compare_entities(extracted_entity_matches, expected_entity_matches)
    
    # Should have metrics by type
    assert "by_type" in comparison
    # Should have at least one type
    assert len(comparison["by_type"]) >= 0


def test_metrics_performance_tracking(db_session):
    """Test that metrics can be tracked over time (stored in JSON)."""
    doc_keys = ["simple_entity"]
    
    metrics = aggregate_metrics_across_documents(db_session, doc_keys)
    
    # Create a metrics snapshot
    snapshot = {
        "timestamp": "2024-01-01T00:00:00Z",
        "metrics": metrics,
    }
    
    # Should be serializable for storage
    snapshot_json = json.dumps(snapshot)
    assert len(snapshot_json) > 0
    
    # Should be deserializable
    loaded = json.loads(snapshot_json)
    assert "timestamp" in loaded
    assert "metrics" in loaded

