"""
Main entity extractor for Stage 2.

Processes sentences in batches, extracts entities using NER,
and persists them to the database.
"""

from typing import TYPE_CHECKING

from okc_core.models import Sentence
from okc_pipeline.stage_02_entities.entity_persister import process_sentence_entities
from okc_pipeline.stage_02_entities.ner_processor import extract_entities_from_sentence

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


def extract_entities_for_sentences(
    session: "Session",
    sentence_ids: list[int],
    batch_size: int = 100,
) -> dict[str, int]:
    """
    Extract entities from sentences and persist them to the database.
    
    Args:
        session: Database session
        sentence_ids: List of sentence IDs to process
        batch_size: Number of sentences to process in each batch
    
    Returns:
        Dictionary with statistics:
        - sentences_processed: Number of sentences processed
        - total_mentions: Total entity mentions created
        - sentences_with_entities: Number of sentences that had entities
    """
    total_mentions = 0
    sentences_with_entities = 0
    
    # Process sentences in batches
    for i in range(0, len(sentence_ids), batch_size):
        batch_ids = sentence_ids[i:i + batch_size]
        
        # Fetch sentences for this batch
        sentences = session.query(Sentence).filter(Sentence.id.in_(batch_ids)).all()
        
        for sentence in sentences:
            # Extract entities from sentence
            entity_mentions = extract_entities_from_sentence(sentence, sentence.text)
            
            if entity_mentions:
                # Process and persist entities
                _, mentions_count = process_sentence_entities(
                    session,
                    sentence,
                    entity_mentions,
                )
                total_mentions += mentions_count
                sentences_with_entities += 1
    
    return {
        "sentences_processed": len(sentence_ids),
        "total_mentions": total_mentions,
        "sentences_with_entities": sentences_with_entities,
    }

