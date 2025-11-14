#!/usr/bin/env python3
"""
Analyze topics to see if they include entities from unfiltered documents.
This helps identify if topics need to be rebuilt with the new filter.
"""
import sys
from pathlib import Path
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from api.db import SessionLocal  # noqa: E402
from api.models import Document, Chunk, Entity, EntityChunk, Topic, TopicMember  # noqa: E402
from pipeline.ingestion.doc_filters import keep_doc  # noqa: E402

def main():
    db = SessionLocal()
    try:
        # Get all topics
        topics = list(db.scalars(select(Topic)))
        total_topics = len(topics)
        
        if total_topics == 0:
            print("No topics found in database.")
            return
        
        print(f"Found {total_topics} topics\n")
        
        # Find unfiltered document IDs
        unfiltered_doc_ids = []
        for doc in db.scalars(select(Document)):
            if not keep_doc(doc.text or "", title=doc.title or ""):
                unfiltered_doc_ids.append(doc.id)
        
        unfiltered_doc_set = set(unfiltered_doc_ids)
        print(f"Unfiltered documents: {len(unfiltered_doc_ids)}\n")
        
        if not unfiltered_doc_ids:
            print("✓ All documents match filter. Topics should be clean.")
            return
        
        # Find chunks from unfiltered documents
        unfiltered_chunk_ids = set(
            db.scalars(
                select(Chunk.id).where(Chunk.document_id.in_(unfiltered_doc_ids))
            ).all()
        )
        
        # Find entities that appear in unfiltered chunks
        unfiltered_entity_ids = set(
            db.scalars(
                select(EntityChunk.entity_id)
                .where(EntityChunk.chunk_id.in_(unfiltered_chunk_ids))
                .distinct()
            ).all()
        )
        
        print(f"Entities from unfiltered docs: {len(unfiltered_entity_ids)}\n")
        
        # Analyze each topic
        topics_with_unfiltered = []
        topics_clean = []
        
        for topic in topics:
            # Get all entities in this topic
            member_entities = db.scalars(
                select(TopicMember.entity_id).where(TopicMember.topic_id == topic.id)
            ).all()
            
            # Check if any are from unfiltered docs
            has_unfiltered = any(eid in unfiltered_entity_ids for eid in member_entities)
            
            if has_unfiltered:
                unfiltered_count = sum(1 for eid in member_entities if eid in unfiltered_entity_ids)
                topics_with_unfiltered.append({
                    "topic": topic,
                    "total_entities": len(member_entities),
                    "unfiltered_entities": unfiltered_count,
                    "pct_unfiltered": 100 * unfiltered_count / len(member_entities) if member_entities else 0
                })
            else:
                topics_clean.append(topic)
        
        print(f"Topics analysis:")
        print(f"  Clean topics (all entities from filtered docs): {len(topics_clean)}")
        print(f"  Topics with unfiltered entities: {len(topics_with_unfiltered)}")
        
        if topics_with_unfiltered:
            print(f"\n⚠️  Topics containing entities from unfiltered documents:")
            print(f"\nTop 10 topics with most unfiltered entities:")
            sorted_topics = sorted(topics_with_unfiltered, key=lambda x: x["unfiltered_entities"], reverse=True)
            for i, info in enumerate(sorted_topics[:10], 1):
                topic = info["topic"]
                print(f"  {i}. Topic {topic.id}: '{topic.label}'")
                print(f"     {info['unfiltered_entities']}/{info['total_entities']} entities ({info['pct_unfiltered']:.1f}%) from unfiltered docs")
            
            print(f"\n→ Recommendation: Rebuild topics with filter")
            print(f"  1. Clean up unfiltered docs: python scripts/cleanup_unfiltered_docs.py --execute")
            print(f"  2. Rebuild entity centroids: python scripts/rebuild_entity_centroids.py")
            print(f"  3. Rebuild topics: python scripts/rebuild_topics.py")
        else:
            print(f"\n✓ All topics are clean - all entities come from filtered documents!")
            
    finally:
        db.close()

if __name__ == "__main__":
    main()

