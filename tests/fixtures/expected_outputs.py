"""Expected outputs (ground truth) for test documents."""

from typing import Dict, List, Any, Optional

# Expected entity outputs: (text, type, char_start, char_end)
# char_start/char_end are relative to the document text
ExpectedEntity = tuple[str, str, int, int]

# Expected claim: (sentence_text, is_claim, min_score)
ExpectedClaim = tuple[str, bool, float]

# Expected relation: (head_entity_text, relation_type, tail_entity_text, min_confidence)
ExpectedRelation = tuple[str, str, str, float]


def _find_entity_in_text(text: str, entity_text: str, start_search: int = 0) -> Optional[tuple[int, int]]:
    """Find entity text in document text, returning (start, end) char offsets."""
    idx = text.lower().find(entity_text.lower(), start_search)
    if idx == -1:
        return None
    return (idx, idx + len(entity_text))


EXPECTED_OUTPUTS: Dict[str, Dict[str, Any]] = {
    "simple_entity": {
        "entities": [
            ("GPT-4", "TechnicalArtifact", 0, 5),
            ("OpenAI", "Organization", 44, 51),
        ],
        "claims": [
            ("GPT-4 is a language model developed by OpenAI.", True, 0.5),
            ("It improves accuracy on various benchmarks.", True, 0.5),
        ],
        "relations": [
            ("GPT-4", "is_a", "language model", 0.5),
            ("GPT-4", "improves", "accuracy", 0.5),
        ],
    },
    
    "multiple_entities": {
        "entities": [
            ("BERT", "TechnicalArtifact", 0, 4),
            ("Google", "Organization", 44, 50),
            ("GLUE", "Dataset", 75, 79),
            ("BERT", "TechnicalArtifact", 100, 104),
            ("accuracy", "Metric", 125, 133),
        ],
        "claims": [
            ("BERT is a transformer model developed by Google.", True, 0.5),
            ("It was evaluated on the GLUE benchmark dataset.", True, 0.5),
            ("BERT improves accuracy on natural language understanding tasks.", True, 0.5),
        ],
        "relations": [
            ("BERT", "is_a", "transformer model", 0.5),
            ("BERT", "evaluated_on", "GLUE", 0.5),
            ("BERT", "improves", "accuracy", 0.5),
        ],
    },
    
    "claim_sentences": {
        "entities": [
            ("GPT-4", "TechnicalArtifact", 0, 5),
            ("GPT-3", "TechnicalArtifact", 30, 35),
            ("BERT", "TechnicalArtifact", 50, 54),
            ("GLUE", "Dataset", 88, 92),
            ("model", "TechnicalArtifact", 98, 103),
            ("datasets", "Dataset", 125, 133),
        ],
        "claims": [
            ("GPT-4 outperforms GPT-3 on most tasks.", True, 0.5),
            ("BERT increases performance on GLUE.", True, 0.5),
            ("The model depends on large datasets for training.", True, 0.5),
        ],
        "relations": [
            ("GPT-4", "improves", "performance", 0.5),
            ("BERT", "improves", "performance", 0.5),
            ("model", "depends_on", "datasets", 0.5),
        ],
    },
    
    "relation_patterns": {
        "entities": [
            ("Transformers", "TechnicalArtifact", 0, 12),
            ("neural network", "Concept", 30, 44),
            ("BERT", "TechnicalArtifact", 60, 64),
            ("GLUE", "Dataset", 88, 92),
            ("Attention mechanisms", "Concept", 108, 130),
            ("model", "TechnicalArtifact", 145, 150),
            ("method", "TechnicalArtifact", 168, 174),
            ("GPU", "Tool", 193, 196),
        ],
        "claims": [
            ("Transformers are a type of neural network architecture.", True, 0.5),
            ("BERT is evaluated on GLUE.", True, 0.5),
            ("Attention mechanisms improve model performance.", True, 0.5),
            ("The method requires GPU acceleration.", True, 0.5),
        ],
        "relations": [
            ("Transformers", "is_a", "neural network architecture", 0.5),
            ("BERT", "evaluated_on", "GLUE", 0.5),
            ("Attention mechanisms", "improves", "model performance", 0.5),
            ("method", "depends_on", "GPU", 0.5),
        ],
    },
    
    "complex_entities": {
        "entities": [
            ("Stanford University", "Organization", 30, 48),
            ("ImageNet", "Dataset", 88, 96),
            ("accuracy", "Metric", 125, 133),
            ("MIT", "Organization", 170, 173),
        ],
        "claims": [
            ("The paper by researchers at Stanford University presents a new approach.", True, 0.5),
            ("The method was tested on ImageNet dataset.", True, 0.5),
            ("Results show improvements in accuracy metrics.", True, 0.5),
        ],
        "relations": [
            ("method", "evaluated_on", "ImageNet", 0.5),
            ("method", "improves", "accuracy", 0.5),
        ],
    },
    
    "hedging_language": {
        "entities": [
            ("model", "TechnicalArtifact", 4, 9),
            ("performance", "Metric", 25, 36),
            ("accuracy", "Metric", 60, 68),
            ("method", "TechnicalArtifact", 95, 101),
        ],
        "claims": [
            ("The model might improve performance.", True, 0.3),  # Lower score due to hedging
            ("It could potentially increase accuracy.", True, 0.3),
            ("The results suggest that the method may work better.", True, 0.3),
        ],
        "relations": [],  # Hedging makes relations less reliable
    },
    
    "is_a_relations": {
        "entities": [
            ("transformer", "TechnicalArtifact", 2, 13),
            ("neural network", "Concept", 30, 44),
            ("BERT", "TechnicalArtifact", 50, 54),
            ("language model", "Concept", 60, 74),
            ("GPT-4", "TechnicalArtifact", 80, 85),
            ("AI system", "Concept", 89, 98),
            ("PyTorch", "Tool", 104, 111),
            ("machine learning framework", "Concept", 115, 143),
        ],
        "claims": [
            ("A transformer is a neural network architecture.", True, 0.5),
            ("BERT is a language model.", True, 0.5),
            ("GPT-4 is an AI system.", True, 0.5),
            ("PyTorch is a machine learning framework.", True, 0.5),
        ],
        "relations": [
            ("transformer", "is_a", "neural network architecture", 0.5),
            ("BERT", "is_a", "language model", 0.5),
            ("GPT-4", "is_a", "AI system", 0.5),
            ("PyTorch", "is_a", "machine learning framework", 0.5),
        ],
    },
    
    "evaluated_on_relations": {
        "entities": [
            ("BERT", "TechnicalArtifact", 0, 4),
            ("GLUE", "Dataset", 25, 29),
            ("GPT-4", "TechnicalArtifact", 40, 45),
            ("MMLU", "Dataset", 65, 69),
            ("ResNet", "TechnicalArtifact", 80, 86),
            ("ImageNet", "Dataset", 105, 113),
            ("model", "TechnicalArtifact", 125, 130),
            ("datasets", "Dataset", 150, 158),
        ],
        "claims": [
            ("BERT was evaluated on GLUE.", True, 0.5),
            ("GPT-4 was tested on MMLU.", True, 0.5),
            ("ResNet was evaluated on ImageNet.", True, 0.5),
            ("The model was tested on multiple datasets.", True, 0.5),
        ],
        "relations": [
            ("BERT", "evaluated_on", "GLUE", 0.5),
            ("GPT-4", "evaluated_on", "MMLU", 0.5),
            ("ResNet", "evaluated_on", "ImageNet", 0.5),
            ("model", "evaluated_on", "datasets", 0.5),
        ],
    },
    
    "depends_on_relations": {
        "entities": [
            ("model", "TechnicalArtifact", 4, 9),
            ("datasets", "Dataset", 25, 33),
            ("Training", "Task", 35, 43),
            ("GPU", "Tool", 58, 61),
            ("method", "TechnicalArtifact", 68, 74),
            ("system", "TechnicalArtifact", 95, 101),
            ("APIs", "Tool", 120, 124),
        ],
        "claims": [
            ("The model depends on large datasets.", True, 0.5),
            ("Training requires GPU resources.", True, 0.5),
            ("The method needs preprocessing steps.", True, 0.5),
            ("The system depends on external APIs.", True, 0.5),
        ],
        "relations": [
            ("model", "depends_on", "datasets", 0.5),
            ("Training", "depends_on", "GPU", 0.5),
            ("method", "depends_on", "preprocessing steps", 0.5),
            ("system", "depends_on", "APIs", 0.5),
        ],
    },
    
    "mixed_content": {
        "entities": [
            ("OpenAI", "Organization", 0, 6),
            ("GPT-4", "TechnicalArtifact", 16, 21),
            ("accuracy", "Metric", 50, 58),
            ("GPT-4", "TechnicalArtifact", 75, 80),
            ("MMLU", "Dataset", 110, 114),
            ("GLUE", "Dataset", 120, 124),
            ("system", "TechnicalArtifact", 135, 141),
            ("transformer", "TechnicalArtifact", 150, 161),
            ("Transformers", "TechnicalArtifact", 175, 186),
            ("neural network", "Concept", 194, 208),
            ("research", "Concept", 220, 228),
            ("San Francisco", "Location", 245, 258),
            ("California", "Location", 260, 270),
        ],
        "claims": [
            ("OpenAI developed GPT-4, a large language model.", True, 0.5),
            ("The model improves accuracy on natural language tasks.", True, 0.5),
            ("GPT-4 was evaluated on various benchmarks including MMLU and GLUE.", True, 0.5),
            ("The system depends on transformer architecture.", True, 0.5),
            ("Transformers are a type of neural network.", True, 0.5),
        ],
        "relations": [
            ("GPT-4", "is_a", "language model", 0.5),
            ("model", "improves", "accuracy", 0.5),
            ("GPT-4", "evaluated_on", "MMLU", 0.5),
            ("GPT-4", "evaluated_on", "GLUE", 0.5),
            ("system", "depends_on", "transformer architecture", 0.5),
            ("Transformers", "is_a", "neural network", 0.5),
        ],
    },
}

