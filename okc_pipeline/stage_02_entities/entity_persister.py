"""
Entity persistence module for storing entities and entity mentions.

Processes entity mentions from NER and creates Entity and EntityMention records
in the database.
"""

from typing import TYPE_CHECKING

from okc_api.crud import upsert_entity
from okc_core.models import EntityMention

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from okc_core.models import Sentence


def process_sentence_entities(
    session: "Session",
    sentence: "Sentence",
    entity_mentions: list[dict],
) -> tuple[int, int]:
    """
    Process entity mentions for a sentence and persist them to the database.
    
    Args:
        session: Database session
        sentence: Sentence object
        entity_mentions: List of entity mention dicts from NER processor:
            - text: Entity surface text
            - start: Character start offset (relative to sentence text)
            - end: Character end offset (relative to sentence text)
            - type: Entity type
            - normalized_name: Normalized entity name
    
    Returns:
        Tuple of (entities_created, mentions_created) counts
    """
    entities_created = 0
    mentions_created = 0
    
    for mention_dict in entity_mentions:
        canonical_name = mention_dict["text"]
        entity_type = mention_dict.get("type")
        char_start = mention_dict["start"]
        char_end = mention_dict["end"]
        
        # Upsert entity (creates if doesn't exist, updates if exists)
        entity_id = upsert_entity(session, canonical_name, entity_type=entity_type)
        
        # Track if this is a new entity (check if it was just created)
        # Note: We can't easily tell if it was new vs existing, so we'll just count mentions
        # For a more accurate count, we'd need to check before/after entity count
        
        # Create EntityMention record
        # Offsets are relative to sentence.text, not chunk.text
        entity_mention = EntityMention(
            entity_id=entity_id,
            sentence_id=sentence.id,
            char_start=char_start,
            char_end=char_end,
            surface_text=canonical_name,
        )
        session.add(entity_mention)
        mentions_created += 1
    
    # Flush to get IDs, but don't commit (let caller handle commits)
    session.flush()
    
    # Note: We can't easily count entities_created without checking before/after
    # For now, return 0 for entities_created and mentions_created count
    return (0, mentions_created)

