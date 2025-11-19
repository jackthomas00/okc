"""
Sentence splitting per chunk (Stage 1).

For each Chunk:
- Run a robust sentence splitter (e.g., spaCy doc.sents)
- Store sentences in a Sentence table
"""

from okc_pipeline.stage_01_sentences.sentence_splitter import split_chunk_into_sentences

__all__ = ["split_chunk_into_sentences"]

