#!/usr/bin/env python3
"""
Migration script to fix duplicate entities caused by poor canonicalization.

This script:
1. Re-normalizes all existing entities using the updated normalization logic
2. Identifies duplicate entities (same normalized_name)
3. Merges duplicates by:
   - Choosing the canonical entity (most mentions, or first alphabetically)
   - Moving all EntityMention records from duplicates to canonical
   - Deleting duplicate entities

Usage:
    python -m okc_cli.fix_duplicate_entities [--dry-run] [--verbose]
"""

import sys
from collections import defaultdict
from sqlalchemy import select, func, update, delete
from sqlalchemy.orm import Session

from okc_core.db import SessionLocal
from okc_core.models import Entity, EntityMention
from okc_pipeline.stage_02_entities.entity_normalizer import normalize_entity_name


def renormalize_all_entities(session: Session, dry_run: bool = False, verbose: bool = False) -> int:
    """
    Re-normalize all entities using the updated normalization function.
    
    Returns:
        Number of entities updated
    """
    entities = session.scalars(select(Entity)).all()
    updated_count = 0
    
    for entity in entities:
        new_normalized = normalize_entity_name(entity.canonical_name)
        if entity.normalized_name != new_normalized:
            if verbose:
                print(f"  Updating {entity.canonical_name}: '{entity.normalized_name}' -> '{new_normalized}'")
            if not dry_run:
                entity.normalized_name = new_normalized
            updated_count += 1
    
    if not dry_run:
        session.commit()
    
    return updated_count


def find_duplicate_groups(session: Session) -> dict[str, list[Entity]]:
    """
    Find groups of entities with the same normalized_name.
    
    Returns:
        Dictionary mapping normalized_name -> list of Entity objects
    """
    # Get all entities with their mention counts
    entities_with_counts = session.execute(
        select(
            Entity,
            func.count(EntityMention.id).label('mention_count')
        )
        .outerjoin(EntityMention, EntityMention.entity_id == Entity.id)
        .where(Entity.normalized_name.is_not(None))
        .group_by(Entity.id)
    ).all()
    
    # Group by normalized_name
    groups = defaultdict(list)
    for entity, mention_count in entities_with_counts:
        groups[entity.normalized_name].append((entity, mention_count))
    
    # Filter to only groups with duplicates
    duplicates = {}
    for normalized_name, entity_list in groups.items():
        if len(entity_list) > 1:
            # Sort by mention count (descending), then by canonical_name (ascending)
            entity_list.sort(key=lambda x: (-x[1], x[0].canonical_name))
            duplicates[normalized_name] = [e for e, _ in entity_list]
    
    return duplicates


def merge_duplicate_entities(
    session: Session,
    canonical_entity: Entity,
    duplicate_entities: list[Entity],
    dry_run: bool = False,
    verbose: bool = False
) -> tuple[int, int]:
    """
    Merge duplicate entities into the canonical one.
    
    Args:
        session: Database session
        canonical_entity: The entity to keep (most mentions)
        duplicate_entities: List of entities to merge into canonical
        dry_run: If True, don't actually make changes
        verbose: If True, print detailed information
    
    Returns:
        Tuple of (mentions_moved, entities_deleted)
    """
    mentions_moved = 0
    entities_deleted = 0
    
    for duplicate in duplicate_entities:
        # Count mentions for this duplicate
        mention_count = session.scalar(
            select(func.count(EntityMention.id))
            .where(EntityMention.entity_id == duplicate.id)
        )
        
        if verbose:
            print(f"    Merging '{duplicate.canonical_name}' (id={duplicate.id}, {mention_count} mentions) "
                  f"into '{canonical_entity.canonical_name}' (id={canonical_entity.id})")
        
        if not dry_run:
            # Move all mentions from duplicate to canonical
            if mention_count > 0:
                session.execute(
                    update(EntityMention)
                    .where(EntityMention.entity_id == duplicate.id)
                    .values(entity_id=canonical_entity.id)
                )
                mentions_moved += mention_count
            
            # Delete the duplicate entity
            # (EntityMention records are CASCADE deleted, but we've already moved them)
            session.execute(
                delete(Entity)
                .where(Entity.id == duplicate.id)
            )
            entities_deleted += 1
    
    if not dry_run:
        session.commit()
    
    return mentions_moved, entities_deleted


def main(dry_run: bool = False, verbose: bool = False):
    """
    Main migration function.
    
    Args:
        dry_run: If True, show what would be done without making changes
        verbose: If True, print detailed information
    """
    session = SessionLocal()
    
    try:
        print("=" * 60)
        print("Entity Deduplication Migration")
        print("=" * 60)
        if dry_run:
            print("DRY RUN MODE - No changes will be made")
        print()
        
        # Step 1: Re-normalize all entities
        print("Step 1: Re-normalizing all entities...")
        updated_count = renormalize_all_entities(session, dry_run=dry_run, verbose=verbose)
        print(f"  Updated {updated_count} entities with new normalized_name values")
        print()
        
        # Step 2: Find duplicate groups
        print("Step 2: Finding duplicate entity groups...")
        duplicate_groups = find_duplicate_groups(session)
        print(f"  Found {len(duplicate_groups)} groups of duplicate entities")
        
        if verbose and duplicate_groups:
            for normalized_name, entities in duplicate_groups.items():
                print(f"    '{normalized_name}': {len(entities)} entities")
                for entity in entities:
                    mention_count = session.scalar(
                        select(func.count(EntityMention.id))
                        .where(EntityMention.entity_id == entity.id)
                    )
                    print(f"      - {entity.canonical_name} (id={entity.id}, {mention_count} mentions)")
        print()
        
        # Step 3: Merge duplicates
        if duplicate_groups:
            print("Step 3: Merging duplicate entities...")
            total_mentions_moved = 0
            total_entities_deleted = 0
            
            for normalized_name, entities in duplicate_groups.items():
                # First entity is the canonical one (sorted by mention count, then name)
                canonical = entities[0]
                duplicates = entities[1:]
                
                if verbose:
                    print(f"  Processing group '{normalized_name}':")
                    print(f"    Canonical: '{canonical.canonical_name}' (id={canonical.id})")
                
                mentions_moved, entities_deleted = merge_duplicate_entities(
                    session,
                    canonical,
                    duplicates,
                    dry_run=dry_run,
                    verbose=verbose
                )
                
                total_mentions_moved += mentions_moved
                total_entities_deleted += entities_deleted
            
            print(f"  Moved {total_mentions_moved} mentions")
            print(f"  Deleted {total_entities_deleted} duplicate entities")
        else:
            print("Step 3: No duplicates found - nothing to merge")
        
        print()
        print("=" * 60)
        print("Migration complete!")
        print("=" * 60)
        
    except Exception as e:
        session.rollback()
        print(f"ERROR: {e}", file=sys.stderr)
        raise
    finally:
        session.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fix duplicate entities in the database"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information"
    )
    
    args = parser.parse_args()
    main(dry_run=args.dry_run, verbose=args.verbose)

