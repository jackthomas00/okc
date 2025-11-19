"""
Entity type mapping from spaCy NER labels to ontology types.

Maps spaCy's NER labels to our 11 entity types:
- Person, Organization, TechnicalArtifact, Task, Metric, Dataset,
  Tool, Concept, Location, Event, Product
"""

import re
from typing import Optional

# Entity type constants
PERSON = "Person"
ORGANIZATION = "Organization"
TECHNICAL_ARTIFACT = "TechnicalArtifact"
TASK = "Task"
METRIC = "Metric"
DATASET = "Dataset"
TOOL = "Tool"
CONCEPT = "Concept"
LOCATION = "Location"
EVENT = "Event"
PRODUCT = "Product"

# All entity types
ENTITY_TYPES = [
    PERSON,
    ORGANIZATION,
    TECHNICAL_ARTIFACT,
    TASK,
    METRIC,
    DATASET,
    TOOL,
    CONCEPT,
    LOCATION,
    EVENT,
    PRODUCT,
]

# Keywords for context-aware typing
DATASET_KEYWORDS = [
    "dataset", "datasets", "corpus", "corpora", "benchmark", "benchmarks",
    "collection", "collection", "data set", "training data", "test data",
    "validation data", "evaluation data"
]

TECHNICAL_ARTIFACT_KEYWORDS = [
    "model", "models", "algorithm", "algorithms", "method", "methods",
    "architecture", "architectures", "approach", "approaches", "technique",
    "techniques", "system", "systems", "network", "networks", "framework"
]

TOOL_KEYWORDS = [
    "library", "libraries", "framework", "frameworks", "toolkit", "toolkits",
    "api", "apis", "package", "packages", "tool", "tools", "software",
    "platform", "platforms", "suite", "suites"
]


def _contains_keywords(text: str, keywords: list[str]) -> bool:
    """Check if text contains any of the keywords (case-insensitive)."""
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)


def map_ner_label_to_type(ner_label: str, entity_text: str, context: Optional[str] = None) -> str:
    """
    Map spaCy NER label to ontology entity type.
    
    Args:
        ner_label: spaCy NER label (e.g., "PERSON", "ORG", "PRODUCT")
        entity_text: The entity text itself (for context-aware typing)
        context: Optional surrounding context (sentence text)
    
    Returns:
        Entity type string (one of the ENTITY_TYPES constants)
    """
    ner_label_upper = ner_label.upper()
    
    # Direct mappings
    if ner_label_upper == "PERSON":
        return PERSON
    
    if ner_label_upper == "ORG":
        return ORGANIZATION
    
    if ner_label_upper in ("GPE", "LOC"):
        return LOCATION
    
    if ner_label_upper == "EVENT":
        return EVENT
    
    # Numeric/metric-like entities
    if ner_label_upper in ("MONEY", "PERCENT", "QUANTITY"):
        return METRIC
    
    # Context-aware typing for PRODUCT and WORK_OF_ART
    if ner_label_upper == "PRODUCT":
        # Check if it's actually a dataset, technical artifact, or tool
        check_text = entity_text
        if context:
            check_text = f"{entity_text} {context}"
        
        if _contains_keywords(check_text, DATASET_KEYWORDS):
            return DATASET
        if _contains_keywords(check_text, TECHNICAL_ARTIFACT_KEYWORDS):
            return TECHNICAL_ARTIFACT
        if _contains_keywords(check_text, TOOL_KEYWORDS):
            return TOOL
        # Default PRODUCT
        return PRODUCT
    
    if ner_label_upper == "WORK_OF_ART":
        # In ML context, WORK_OF_ART often refers to models/methods
        check_text = entity_text
        if context:
            check_text = f"{entity_text} {context}"
        
        if _contains_keywords(check_text, DATASET_KEYWORDS):
            return DATASET
        if _contains_keywords(check_text, TECHNICAL_ARTIFACT_KEYWORDS):
            return TECHNICAL_ARTIFACT
        if _contains_keywords(check_text, TOOL_KEYWORDS):
            return TOOL
        # Default to TechnicalArtifact for ML context, fallback to Concept
        return TECHNICAL_ARTIFACT
    
    # Everything else defaults to Concept
    return CONCEPT

