"""
NER processor for extracting entities from sentences using spaCy.

Extracts entity mentions from sentences, maps NER labels to ontology types,
and normalizes entity names.
"""

from typing import TYPE_CHECKING, Optional

from okc_pipeline.stage_02_entities.entity_normalizer import normalize_entity_name
from okc_pipeline.stage_02_entities.entity_typing import map_ner_label_to_type
from okc_pipeline.utils.spacy_processing import make_doc_with_ner, make_doc, noun_chunk_spans

if TYPE_CHECKING:
    from okc_core.models import Sentence


def extract_entities_from_sentence(sentence: "Sentence", sentence_text: Optional[str] = None) -> list[dict]:
    """
    Extract entities from a sentence using spaCy NER.
    
    Args:
        sentence: Sentence object
        sentence_text: Optional sentence text (if not provided, uses sentence.text)
    
    Returns:
        List of entity mention dictionaries with keys:
        - text: Entity surface text
        - start: Character start offset (relative to sentence text)
        - end: Character end offset (relative to sentence text)
        - type: Entity type (from ontology)
        - normalized_name: Normalized entity name
        - ner_label: Original spaCy NER label
    """
    if sentence_text is None:
        sentence_text = sentence.text
    
    if not sentence_text or not sentence_text.strip():
        return []
    
    try:
        # Process sentence with spaCy NER
        doc = make_doc_with_ner(sentence_text)
        
        ner_entities = []
        for ent in doc.ents:
            # Skip empty entities
            if not ent.text or not ent.text.strip():
                continue
            
            # Get character offsets relative to sentence text
            start = ent.start_char
            end = ent.end_char
            
            # Map NER label to ontology type
            entity_type = map_ner_label_to_type(
                ent.label_,
                ent.text,
                context=sentence_text
            )
            
            # Normalize entity name
            normalized_name = normalize_entity_name(ent.text)
            
            ner_entities.append({
                "text": ent.text,
                "start": start,
                "end": end,
                "type": entity_type,
                "normalized_name": normalized_name,
                "ner_label": ent.label_,
            })

        # -----------------------------------------
        # NEW: fallback concept entities from noun chunks
        # -----------------------------------------
        noun_doc = make_doc(sentence_text)  # faster; parser-only
        noun_chunks = noun_chunk_spans(noun_doc)

        fallback_entities = []
        ner_ranges = {(e["start"], e["end"]) for e in ner_entities}

        for span in noun_chunks:
            rng = (span.start, span.end)
            if rng in ner_ranges:
                continue

            fallback_entities.append({
                "text": span.text,
                "start": span.start,
                "end": span.end,
                "type": "Concept",               # fallback ontology type
                "normalized_name": normalize_entity_name(span.text),
                "ner_label": None,
            })

        return ner_entities + fallback_entities
    
    except Exception:
        # Handle spaCy errors gracefully - return empty list
        # This allows processing to continue for other sentences
        return []

