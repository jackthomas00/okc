# scripts/rebuild_topics.py
from api.db import SessionLocal  # noqa: E402
from pipeline.topics.topics import rebuild_entity_centroids, rebuild_topics  # noqa: E402
from pipeline.ingestion.doc_filters import DOCUMENT_FILTER_KEYWORDS  # noqa: E402

def main():
    db = SessionLocal()
    try:
        rebuild_entity_centroids(db, min_occurrences=2)
        rebuild_topics(
            db, 
            n_topics=300, 
            min_popularity=3.0,
            document_filter_keywords=DOCUMENT_FILTER_KEYWORDS
        )
        db.commit()
    finally:
        db.close()

if __name__ == "__main__":
    main()
