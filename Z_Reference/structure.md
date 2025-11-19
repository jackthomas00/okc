backend/
  pipeline/
    __init__.py

    ingestion/
      ingest_docs.py
      chunker.py

    embeddings/
      embedder.py
      vector_store.py

    entities/
      ner.py
      term_mining.py
      alias_map.py
      canonicalize.py
      entity_linker.py

    claims/
      claim_sentence_detector.py
      claim_extractor.py
      dependency_rules.py
      claim_inserter.py

    relations/
      relation_typer.py
      relation_inserter.py

    utils/
      text_cleaning.py
      scoring.py
      graph_building.py
