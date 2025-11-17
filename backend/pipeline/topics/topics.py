# topics.py
# NOTE: This module is not part of Milestone 1 and is not used.
# The Topic and TopicMember models have been removed from the schema.
# This file is kept for reference but should not be imported or used.

from __future__ import annotations
from collections import defaultdict
from typing import List, Tuple, Optional

import numpy as np
from sqlalchemy import select, delete, func, or_
from sqlalchemy.orm import Session

from api.models import Entity, EntityChunk, Chunk, Document, Topic, TopicMember

try:
    from sklearn.cluster import MiniBatchKMeans
except ImportError:
    MiniBatchKMeans = None

def rebuild_topics(
    session: Session,
    n_topics: int = 300,
    min_popularity: float = 3.0,
    document_filter_keywords: Optional[List[str]] = None,
):
    """
    Rebuild topics by clustering entities.
    
    Args:
        session: Database session
        n_topics: Number of topics to create
        min_popularity: Minimum entity popularity (occurrence count) to include
        document_filter_keywords: If provided, only include entities from documents
            that contain any of these keywords (case-insensitive). This ensures
            topics only include entities from filtered documents.
    """
    if MiniBatchKMeans is None:
        raise RuntimeError("sklearn is required for rebuild_topics (pip install scikit-learn)")

    # Build entity query with optional document filtering
    entity_query = (
        select(Entity.id, Entity.popularity, Entity.centroid)
        .where(Entity.centroid.is_not(None))
        .where(Entity.popularity.is_not(None))
        .where(Entity.popularity >= min_popularity)
    )
    
    # If document filter is provided, only include entities that appear in filtered documents
    if document_filter_keywords:
        print(f"[rebuild_topics] Filtering entities to documents containing: {document_filter_keywords[:5]}...")
        # Find document IDs that match the filter
        # Match if any keyword appears in the document text (case-insensitive)
        doc_conditions = [
            func.lower(Document.text).contains(keyword.lower())
            for keyword in document_filter_keywords
        ]
        filtered_doc_ids = select(Document.id).where(or_(*doc_conditions)).distinct()
        
        # Find chunks from filtered documents
        filtered_chunk_ids = select(Chunk.id).where(Chunk.document_id.in_(filtered_doc_ids))
        
        # Find entities that appear in filtered chunks
        filtered_entity_ids = (
            select(EntityChunk.entity_id)
            .where(EntityChunk.chunk_id.in_(filtered_chunk_ids))
            .distinct()
        )
        
        # Filter entities to only those in filtered documents
        entity_query = entity_query.where(Entity.id.in_(filtered_entity_ids))
    
    # fetch candidate entities
    rows: List[Tuple[int, float, list[float]]] = list(
        session.execute(entity_query)
    )

    if not rows:
        filter_msg = " (after document filtering)" if document_filter_keywords else ""
        print(f"No entities with centroids{filter_msg}; run rebuild_entity_centroids first.")
        return
    
    print(f"[rebuild_topics] Found {len(rows)} entities to cluster (min_popularity={min_popularity})")

    entity_ids: List[int] = []
    X: List[np.ndarray] = []

    for eid, pop, cent in rows:
        entity_ids.append(eid)
        v = np.asarray(cent, dtype=np.float32)
        # assume already normalized, but just in case
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v = v / norm
        X.append(v)

    X_arr = np.vstack(X)
    n_clusters = min(n_topics, X_arr.shape[0])

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=2048,
        max_iter=100,
        n_init="auto",
        verbose=0,
    )
    labels = kmeans.fit_predict(X_arr)
    centers = kmeans.cluster_centers_

    # wipe old topics
    session.execute(delete(TopicMember))
    session.execute(delete(Topic))

    # build topics
    # group entity indices by cluster
    cluster_members: dict[int, List[int]] = {}
    for idx, lbl in enumerate(labels):
        cluster_members.setdefault(int(lbl), []).append(idx)

    for cluster_id, member_indices in cluster_members.items():
        if len(member_indices) < 3:
            # tiny clusters are mostly noise; skip or keep as micro-topics
            continue

        # create Topic
        centroid_vec = centers[cluster_id]
        t = Topic(
            label="",  # fill later
            summary=None,
            centroid=centroid_vec.astype(np.float32).tolist(),
        )
        session.add(t)
        session.flush()  # get id

        # find a simple label: the most popular entity in the cluster
        best_ent_id = None
        best_pop = -1.0

        for idx in member_indices:
            eid = entity_ids[idx]
            ent = session.get(Entity, eid)
            if not ent:
                continue
            pop = ent.popularity or 0.0
            if pop > best_pop:
                best_pop = pop
                best_ent_id = eid

        if best_ent_id is not None:
            ent = session.get(Entity, best_ent_id)
            t.label = ent.name

        # add TopicMembers with scores = cosine similarity to topic centroid
        centroid_norm = float(np.linalg.norm(centroid_vec))
        for idx in member_indices:
            eid = entity_ids[idx]
            v = X_arr[idx]
            # cosine similarity
            dot = float(np.dot(v, centroid_vec))
            if centroid_norm > 0:
                score = dot / centroid_norm
            else:
                score = dot

            tm = TopicMember(
                topic_id=t.id,
                entity_id=eid,
                score=score,
            )
            session.add(tm)

def rebuild_entity_centroids(
    session: Session,
    min_occurrences: int = 2,
    batch_size: int = 1000,
    start_from: int = 0,
):
    """
    For each entity, mean-pool the embeddings of chunks it appears in.
    Also store occurrence count in Entity.popularity.
    
    Processes entities in batches to avoid memory issues.
    
    Args:
        start_from: Batch index to start from (for resuming after crashes)
    """
    # Get all entity IDs that have chunks
    entity_ids = session.scalars(
        select(EntityChunk.entity_id).distinct()
    ).all()
    
    total_entities = len(entity_ids)
    updated_count = 0
    skipped_count = 0
    last_batch = start_from
    
    # Process entities in batches
    for batch_start in range(start_from * batch_size, total_entities, batch_size):
        batch_entity_ids = entity_ids[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size
        
        # Get all (entity_id, embedding) pairs for this batch of entities
        rows = session.execute(
            select(EntityChunk.entity_id, Chunk.embedding)
            .join(Chunk, Chunk.id == EntityChunk.chunk_id)
            .where(EntityChunk.entity_id.in_(batch_entity_ids))
        )
        
        # Accumulate embeddings per entity (only for this batch)
        sums: dict[int, np.ndarray] = {}
        counts: dict[int, int] = defaultdict(int)
        
        for eid, emb in rows:
            if emb is None:
                continue
            v = np.asarray(emb, dtype=np.float32)
            if eid not in sums:
                sums[eid] = v.copy()
            else:
                sums[eid] += v
            counts[eid] += 1
        
        # Update entities in this batch
        for eid in batch_entity_ids:
            if eid not in sums:
                continue
                
            c = counts[eid]
            if c < min_occurrences:
                skipped_count += 1
                continue
            
            vec = sums[eid] / float(c)
            # normalize
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            
            ent = session.get(Entity, eid)
            if not ent:
                continue
            
            ent.centroid = vec.astype(np.float32).tolist()
            ent.popularity = float(c)
            updated_count += 1
        
        # Commit after each batch to free memory
        session.commit()
        last_batch = batch_num + 1  # Track last completed batch (1-indexed for next resume)
        
        if (batch_num + 1) % 10 == 0 or batch_start + len(batch_entity_ids) >= total_entities:
            print(f"[rebuild_entity_centroids] Processed {batch_start + len(batch_entity_ids)}/{total_entities} entities, updated {updated_count}, skipped {skipped_count}")
    
    return updated_count, skipped_count, last_batch


def merge_aliases(session: Session):
    """
    Group entities by canonical_label and set alias_of for non-canonical entities.
    
    Within each group sharing the same canonical_label, picks the most popular
    entity (by popularity) as canonical and sets all others' alias_of to that id.
    
    Only processes entities that don't already have alias_of set.
    """
    # Get all entities with canonical_label that aren't already aliases
    entities = session.scalars(
        select(Entity)
        .where(Entity.canonical_label.is_not(None))
        .where(Entity.alias_of.is_(None))
    ).all()
    
    # Group by canonical_label
    groups: dict[str, list[Entity]] = defaultdict(list)
    for entity in entities:
        if entity.canonical_label:
            groups[entity.canonical_label].append(entity)
    
    updated_count = 0
    
    # Process each group
    for canonical_label, group_entities in groups.items():
        if len(group_entities) <= 1:
            # No aliases to merge
            continue
        
        # Find the most popular entity (canonical)
        # Sort by popularity descending, then by id ascending as tiebreaker
        canonical_entity = max(
            group_entities,
            key=lambda e: (e.popularity if e.popularity is not None else -1.0, -e.id)
        )
        
        # Set alias_of for all others
        for entity in group_entities:
            if entity.id != canonical_entity.id:
                entity.alias_of = canonical_entity.id
                updated_count += 1
    
    session.commit()
    return updated_count
