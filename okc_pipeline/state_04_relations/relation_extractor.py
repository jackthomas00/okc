"""Relation extraction for Stage 4."""

from __future__ import annotations

from typing import Sequence

from sqlalchemy.orm import Session, selectinload

from okc_core.models import EntityMention, Relation, RelationEvidence, Sentence
from okc_pipeline.state_04_relations.relation_rules import RELATION_PATTERNS
from okc_pipeline.utils.spacy_processing import make_doc


def extract_relations_from_sentence(
    sentence: Sentence, mentions: Sequence[EntityMention]
) -> list[dict]:
    """Extract relation candidates from a sentence with its entity mentions.
    
    Args:
        sentence: Sentence object with text.
        mentions: List of EntityMention objects for this sentence.
    
    Returns:
        List of relation dicts with keys: relation_type, head_entity_id, tail_entity_id, confidence
    """
    if len(mentions) < 2:
        return []
    
    doc = make_doc(sentence.text)
    relations = []
    
    # Build a map of entity mentions by their character spans for quick lookup
    # Also create token-to-entity mapping for better matching
    mention_by_span = {}
    for mention in mentions:
        # Store mention by its character span
        mention_by_span[(mention.char_start, mention.char_end)] = mention
    
    # Create a map from token indices to entity mentions
    token_to_mention = {}
    for mention in mentions:
        # Find tokens that overlap with this entity's character span
        for token in doc:
            token_start = token.idx
            token_end = token.idx + len(token.text)
            # Check if token overlaps with entity span
            if (token_start < mention.char_end and token_end > mention.char_start):
                if token.i not in token_to_mention:
                    token_to_mention[token.i] = []
                token_to_mention[token.i].append(mention)
    
    # Simple verb-head extraction
    for token in doc:
        for pattern in RELATION_PATTERNS:
            if token.lemma_.lower() in pattern["verbs"]:
                # Subject and object via dependencies
                # Look for subject (can be in various positions)
                subj = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                # Look for object (direct object, attribute, or prepositional object)
                obj = [child for child in token.children if child.dep_ in ("dobj", "attr", "pobj")]
                
                # Also check for prepositional phrases (e.g., "evaluated on GLUE")
                if not obj:
                    # Look for prep + pobj pattern
                    for prep_child in token.children:
                        if prep_child.dep_ == "prep":
                            pobj = [gc for gc in prep_child.children if gc.dep_ == "pobj"]
                            if pobj:
                                obj = pobj

                if not subj or not obj:
                    continue

                subj_token = subj[0]
                obj_token = obj[0]

                # Match entities to tokens - check if tokens are part of entity spans
                head_mention = None
                tail_mention = None

                # Use token_to_mention map for direct lookup, or fall back to span checking
                # Check subject token
                if subj_token.i in token_to_mention:
                    # Token is directly part of an entity
                    head_mention = token_to_mention[subj_token.i][0]  # Take first match
                else:
                    # Fall back to span overlap check
                    for mention in mentions:
                        if (subj_token.idx < mention.char_end and 
                            subj_token.idx + len(subj_token.text) > mention.char_start):
                            head_mention = mention
                            break
                
                # Check object token
                if obj_token.i in token_to_mention:
                    tail_mention = token_to_mention[obj_token.i][0]  # Take first match
                else:
                    # Fall back to span overlap check
                    for mention in mentions:
                        if (obj_token.idx < mention.char_end and 
                            obj_token.idx + len(obj_token.text) > mention.char_start):
                            tail_mention = mention
                            break

                if not head_mention or not tail_mention:
                    continue
                
                if head_mention.entity_id == tail_mention.entity_id:
                    continue  # Skip self-relations

                # Type constraints - need to load entity types
                head_entity = head_mention.entity
                tail_entity = tail_mention.entity
                
                # If entities don't have types, skip type constraint check but still allow relation
                # (this is more permissive - we can tighten later)
                if head_entity.type and tail_entity.type:
                    allowed = pattern["allowed"]
                    h_type = head_entity.type
                    t_type = tail_entity.type

                    if allowed != "any" and (h_type, t_type) not in allowed:
                        continue

                # Calculate confidence (simple heuristic)
                confidence = 1.0
                # Lower confidence if types don't match or are missing
                if not head_entity.type or not tail_entity.type:
                    confidence = 0.7
                elif pattern["allowed"] != "any":
                    # Type constraint passed, so higher confidence
                    confidence = 1.0
                else:
                    confidence = 0.9  # "any" type relation

                relations.append({
                    "relation_type": pattern["relation_type"],
                    "head_entity_id": head_entity.id,
                    "tail_entity_id": tail_entity.id,
                    "confidence": confidence,
                })
    
    # Fallback: if we found a relation verb but no structured relation,
    # try to match entities based on proximity to the verb
    # This is less precise but catches more relations
    if not relations and len(mentions) >= 2:
        for token in doc:
            for pattern in RELATION_PATTERNS:
                if token.lemma_.lower() in pattern["verbs"]:
                    # Find entities that appear before and after the verb
                    entities_before = []
                    entities_after = []
                    
                    for mention in mentions:
                        mention_center = (mention.char_start + mention.char_end) // 2
                        token_start = token.idx
                        
                        if mention_center < token_start:
                            entities_before.append(mention)
                        elif mention_center > token_start:
                            entities_after.append(mention)
                    
                    # Try to form relations from entities before and after the verb
                    if entities_before and entities_after:
                        # Take the closest entity before and after
                        head_mention = max(entities_before, key=lambda m: (m.char_start + m.char_end) // 2)
                        tail_mention = min(entities_after, key=lambda m: (m.char_start + m.char_end) // 2)
                        
                        if head_mention.entity_id != tail_mention.entity_id:
                            head_entity = head_mention.entity
                            tail_entity = tail_mention.entity
                            
                            # Check type constraints if both have types
                            skip = False
                            if head_entity.type and tail_entity.type:
                                allowed = pattern["allowed"]
                                if allowed != "any" and (head_entity.type, tail_entity.type) not in allowed:
                                    skip = True
                            
                            if not skip:
                                # Lower confidence for fallback matches
                                confidence = 0.5
                                if head_entity.type and tail_entity.type:
                                    confidence = 0.6
                                
                                relations.append({
                                    "relation_type": pattern["relation_type"],
                                    "head_entity_id": head_entity.id,
                                    "tail_entity_id": tail_entity.id,
                                    "confidence": confidence,
                                })
                                break  # Only add one relation per verb to avoid duplicates

    return relations


def extract_relations_for_sentences(session: Session, sentence_ids: Sequence[int]) -> dict[str, int]:
    """Extract relations from sentences and persist Relation and RelationEvidence rows.
    
    Args:
        session: SQLAlchemy session.
        sentence_ids: Sentence primary keys to process.
    
    Returns:
        Simple stats about processed sentences and extracted relations.
    """
    if not sentence_ids:
        return {"sentences_processed": 0, "relations_extracted": 0}
    
    # Load sentences with their mentions and entities
    sentences = (
        session.query(Sentence)
        .options(
            selectinload(Sentence.mentions).selectinload(EntityMention.entity)
        )
        .filter(Sentence.id.in_(sentence_ids))
        .all()
    )
    
    stats = {"sentences_processed": 0, "relations_extracted": 0}
    
    # Track relations we've seen in this batch to avoid duplicates
    seen_relations = set()
    
    for sentence in sentences:
        mentions = sentence.mentions
        if len(mentions) < 2:
            continue
        
        stats["sentences_processed"] += 1
        
        # Extract relation candidates
        relation_candidates = extract_relations_from_sentence(sentence, mentions)
        
        for rel_data in relation_candidates:
            # Create a unique key for this relation
            rel_key = (
                rel_data["head_entity_id"],
                rel_data["tail_entity_id"],
                rel_data["relation_type"]
            )
            
            # Check if we've already processed this relation in this batch
            if rel_key in seen_relations:
                # Find the relation we just created
                existing_relation = (
                    session.query(Relation)
                    .filter(
                        Relation.head_entity_id == rel_data["head_entity_id"],
                        Relation.tail_entity_id == rel_data["tail_entity_id"],
                        Relation.relation_type == rel_data["relation_type"]
                    )
                    .first()
                )
                if existing_relation:
                    # Check if evidence already exists for this sentence
                    existing_evidence = (
                        session.query(RelationEvidence)
                        .filter(
                            RelationEvidence.relation_id == existing_relation.id,
                            RelationEvidence.sentence_id == sentence.id
                        )
                        .first()
                    )
                    if not existing_evidence:
                        session.add(
                            RelationEvidence(
                                relation_id=existing_relation.id,
                                sentence_id=sentence.id,
                                explanation=f"Extracted via pattern: {rel_data['relation_type']}"
                            )
                        )
                continue
            
            # Check if relation already exists in database
            existing_relation = (
                session.query(Relation)
                .filter(
                    Relation.head_entity_id == rel_data["head_entity_id"],
                    Relation.tail_entity_id == rel_data["tail_entity_id"],
                    Relation.relation_type == rel_data["relation_type"]
                )
                .first()
            )
            
            if existing_relation:
                # Relation exists, just add evidence if not already present
                existing_evidence = (
                    session.query(RelationEvidence)
                    .filter(
                        RelationEvidence.relation_id == existing_relation.id,
                        RelationEvidence.sentence_id == sentence.id
                    )
                    .first()
                )
                if not existing_evidence:
                    session.add(
                        RelationEvidence(
                            relation_id=existing_relation.id,
                            sentence_id=sentence.id,
                            explanation=f"Extracted via pattern: {rel_data['relation_type']}"
                        )
                    )
                seen_relations.add(rel_key)
                continue
            
            # Create new relation
            seen_relations.add(rel_key)
            stats["relations_extracted"] += 1
            
            relation = Relation(
                head_entity_id=rel_data["head_entity_id"],
                tail_entity_id=rel_data["tail_entity_id"],
                relation_type=rel_data["relation_type"],
                confidence=rel_data["confidence"]
            )
            session.add(relation)
            session.flush()  # Flush to get the relation ID
            
            # Add evidence
            session.add(
                RelationEvidence(
                    relation_id=relation.id,
                    sentence_id=sentence.id,
                    explanation=f"Extracted via pattern: {rel_data['relation_type']}"
                )
            )
    
    session.flush()
    return stats
