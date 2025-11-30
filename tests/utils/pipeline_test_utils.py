"""Utility functions for pipeline testing and metrics calculation."""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

from sqlalchemy.orm import Session

from okc_core.models import (
    Document, Chunk, Sentence, Entity, EntityMention,
    ClaimSentence, Relation, RelationEvidence
)


@dataclass
class EntityMatch:
    """Represents a matched entity for comparison."""
    text: str
    type: str
    char_start: int
    char_end: int
    normalized_name: Optional[str] = None


@dataclass
class ClaimMatch:
    """Represents a matched claim for comparison."""
    sentence_text: str
    is_claim: bool
    score: float


@dataclass
class RelationMatch:
    """Represents a matched relation for comparison."""
    head_entity_text: str
    relation_type: str
    tail_entity_text: str
    confidence: float


@dataclass
class Metrics:
    """Precision, recall, and F1 metrics."""
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def normalize_text_for_comparison(text: str) -> str:
    """Normalize text for comparison (lowercase, strip)."""
    return text.lower().strip()


def _entity_key(entity: EntityMatch) -> Tuple[str, str]:
    """Create a key for entity matching."""
    return (normalize_text_for_comparison(entity.text), entity.type)


def _relation_key(relation: RelationMatch) -> Tuple[str, str, str]:
    """Create a key for relation matching."""
    return (
        normalize_text_for_comparison(relation.head_entity_text),
        relation.relation_type,
        normalize_text_for_comparison(relation.tail_entity_text)
    )


def compare_entities(
    extracted: List[EntityMatch],
    expected: List[EntityMatch],
    fuzzy_match: bool = True
) -> Dict[str, Any]:
    """
    Compare extracted entities against expected entities.
    
    Args:
        extracted: List of extracted entities
        expected: List of expected entities
        fuzzy_match: If True, allow fuzzy text matching (case-insensitive, partial)
    
    Returns:
        Dictionary with:
        - overall: Metrics for all entities
        - by_type: Metrics broken down by entity type
        - matches: List of matched entities
        - false_positives: Entities extracted but not expected
        - false_negatives: Entities expected but not extracted
    """
    # Convert to sets for matching
    extracted_set = {_entity_key(e) for e in extracted}
    expected_set = {_entity_key(e) for e in expected}
    
    # Find matches
    matches = extracted_set & expected_set
    false_positives = extracted_set - expected_set
    false_negatives = expected_set - extracted_set
    
    # Calculate overall metrics
    tp = len(matches)
    fp = len(false_positives)
    fn = len(false_negatives)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    overall = Metrics(precision, recall, f1, tp, fp, fn)
    
    # Calculate metrics by type
    by_type: Dict[str, Metrics] = {}
    for entity_type in set(e.type for e in extracted + expected):
        type_extracted = {_entity_key(e) for e in extracted if e.type == entity_type}
        type_expected = {_entity_key(e) for e in expected if e.type == entity_type}
        
        type_matches = type_extracted & type_expected
        type_fp = type_extracted - type_expected
        type_fn = type_expected - type_extracted
        
        type_tp = len(type_matches)
        type_fp_count = len(type_fp)
        type_fn_count = len(type_fn)
        
        type_precision = type_tp / (type_tp + type_fp_count) if (type_tp + type_fp_count) > 0 else 0.0
        type_recall = type_tp / (type_tp + type_fn_count) if (type_tp + type_fn_count) > 0 else 0.0
        type_f1 = 2 * type_precision * type_recall / (type_precision + type_recall) if (type_precision + type_recall) > 0 else 0.0
        
        by_type[entity_type] = Metrics(type_precision, type_recall, type_f1, type_tp, type_fp_count, type_fn_count)
    
    return {
        "overall": overall,
        "by_type": by_type,
        "matches": list(matches),
        "false_positives": list(false_positives),
        "false_negatives": list(false_negatives),
    }


def compare_claims(
    extracted: List[ClaimMatch],
    expected: List[ClaimMatch],
    score_tolerance: float = 0.2
) -> Dict[str, Any]:
    """
    Compare extracted claims against expected claims.
    
    Args:
        extracted: List of extracted claims
        expected: List of expected claims
        score_tolerance: Tolerance for score comparison
    
    Returns:
        Dictionary with metrics and comparison details
    """
    # Normalize sentence texts for matching
    extracted_by_text = {
        normalize_text_for_comparison(c.sentence_text): c
        for c in extracted
    }
    expected_by_text = {
        normalize_text_for_comparison(c.sentence_text): c
        for c in expected
    }
    
    # Find matches
    all_texts = set(extracted_by_text.keys()) | set(expected_by_text.keys())
    
    tp = 0  # True positives: correctly identified as claims
    fp = 0  # False positives: identified as claims but shouldn't be
    fn = 0  # False negatives: should be claims but weren't identified
    
    matches = []
    false_positives = []
    false_negatives = []
    
    for text in all_texts:
        ext = extracted_by_text.get(text)
        exp = expected_by_text.get(text)
        
        if ext and exp:
            # Both exist - check if claim status matches
            if ext.is_claim == exp.is_claim:
                if ext.is_claim:
                    tp += 1
                    matches.append((text, ext, exp))
            else:
                if ext.is_claim:
                    fp += 1
                    false_positives.append((text, ext, exp))
                else:
                    fn += 1
                    false_negatives.append((text, ext, exp))
        elif ext and not exp:
            # Extracted but not expected
            if ext.is_claim:
                fp += 1
                false_positives.append((text, ext, None))
        elif not ext and exp:
            # Expected but not extracted
            if exp.is_claim:
                fn += 1
                false_negatives.append((text, None, exp))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "metrics": Metrics(precision, recall, f1, tp, fp, fn),
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def compare_relations(
    extracted: List[RelationMatch],
    expected: List[RelationMatch],
    fuzzy_match: bool = True
) -> Dict[str, Any]:
    """
    Compare extracted relations against expected relations.
    
    Args:
        extracted: List of extracted relations
        expected: List of expected relations
        fuzzy_match: If True, allow fuzzy text matching
    
    Returns:
        Dictionary with:
        - overall: Metrics for all relations
        - by_type: Metrics broken down by relation type
        - matches: List of matched relations
        - false_positives: Relations extracted but not expected
        - false_negatives: Relations expected but not extracted
    """
    # Convert to sets for matching
    extracted_set = {_relation_key(r) for r in extracted}
    expected_set = {_relation_key(r) for r in expected}
    
    # Find matches
    matches = extracted_set & expected_set
    false_positives = extracted_set - expected_set
    false_negatives = expected_set - extracted_set
    
    # Calculate overall metrics
    tp = len(matches)
    fp = len(false_positives)
    fn = len(false_negatives)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    overall = Metrics(precision, recall, f1, tp, fp, fn)
    
    # Calculate metrics by type
    by_type: Dict[str, Metrics] = {}
    for relation_type in set(r.relation_type for r in extracted + expected):
        type_extracted = {_relation_key(r) for r in extracted if r.relation_type == relation_type}
        type_expected = {_relation_key(r) for r in expected if r.relation_type == relation_type}
        
        type_matches = type_extracted & type_expected
        type_fp = type_extracted - type_expected
        type_fn = type_expected - type_extracted
        
        type_tp = len(type_matches)
        type_fp_count = len(type_fp)
        type_fn_count = len(type_fn)
        
        type_precision = type_tp / (type_tp + type_fp_count) if (type_tp + type_fp_count) > 0 else 0.0
        type_recall = type_tp / (type_tp + type_fn_count) if (type_tp + type_fn_count) > 0 else 0.0
        type_f1 = 2 * type_precision * type_recall / (type_precision + type_recall) if (type_precision + type_recall) > 0 else 0.0
        
        by_type[relation_type] = Metrics(type_precision, type_recall, type_f1, type_tp, type_fp_count, type_fn_count)
    
    return {
        "overall": overall,
        "by_type": by_type,
        "matches": list(matches),
        "false_positives": list(false_positives),
        "false_negatives": list(false_negatives),
    }


def extract_entities_from_db(session: Session, document_id: int) -> List[EntityMatch]:
    """Extract entities from database for a document."""
    entities = (
        session.query(EntityMention, Entity)
        .join(Entity, EntityMention.entity_id == Entity.id)
        .join(Sentence, EntityMention.sentence_id == Sentence.id)
        .join(Chunk, Sentence.chunk_id == Chunk.id)
        .filter(Chunk.document_id == document_id)
        .all()
    )
    
    result = []
    for mention, entity in entities:
        result.append(EntityMatch(
            text=mention.surface_text,
            type=entity.type or "Concept",
            char_start=mention.char_start,
            char_end=mention.char_end,
            normalized_name=entity.normalized_name
        ))
    
    return result


def extract_claims_from_db(session: Session, document_id: int) -> List[ClaimMatch]:
    """Extract claims from database for a document."""
    claims = (
        session.query(ClaimSentence, Sentence)
        .join(Sentence, ClaimSentence.sentence_id == Sentence.id)
        .join(Chunk, Sentence.chunk_id == Chunk.id)
        .filter(Chunk.document_id == document_id)
        .all()
    )
    
    result = []
    for claim, sentence in claims:
        result.append(ClaimMatch(
            sentence_text=sentence.text,
            is_claim=claim.is_claim,
            score=claim.score or 0.0
        ))
    
    return result


def extract_relations_from_db(session: Session, document_id: int) -> List[RelationMatch]:
    """Extract relations from database for a document."""
    from sqlalchemy.orm import aliased
    
    # Use aliases for the two Entity joins
    HeadEntity = aliased(Entity)
    TailEntity = aliased(Entity)
    
    relations = (
        session.query(Relation, HeadEntity, TailEntity)
        .join(HeadEntity, Relation.head_entity_id == HeadEntity.id)
        .join(TailEntity, Relation.tail_entity_id == TailEntity.id)
        .join(RelationEvidence, Relation.id == RelationEvidence.relation_id)
        .join(Sentence, RelationEvidence.sentence_id == Sentence.id)
        .join(Chunk, Sentence.chunk_id == Chunk.id)
        .filter(Chunk.document_id == document_id)
        .distinct()
        .all()
    )
    
    result = []
    for relation, head_entity, tail_entity in relations:
        result.append(RelationMatch(
            head_entity_text=head_entity.canonical_name,
            relation_type=relation.relation_type,
            tail_entity_text=tail_entity.canonical_name,
            confidence=relation.confidence
        ))
    
    return result


def calculate_metrics(
    extracted_entities: List[EntityMatch],
    expected_entities: List[EntityMatch],
    extracted_claims: List[ClaimMatch],
    expected_claims: List[ClaimMatch],
    extracted_relations: List[RelationMatch],
    expected_relations: List[RelationMatch],
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for all pipeline outputs.
    
    Returns:
        Dictionary with metrics for entities, claims, and relations
    """
    entity_metrics = compare_entities(extracted_entities, expected_entities)
    claim_metrics = compare_claims(extracted_claims, expected_claims)
    relation_metrics = compare_relations(extracted_relations, expected_relations)
    
    return {
        "entities": entity_metrics,
        "claims": claim_metrics,
        "relations": relation_metrics,
    }

