"""
Sentence splitting functionality for Stage 1.

Uses spaCy's sentence segmentation to split chunks into clean sentences.
"""

from typing import TYPE_CHECKING

from okc_pipeline.utils.spacy_processing import make_doc, sentence_spans_from_doc

if TYPE_CHECKING:
    from okc_core.models import Chunk


def split_chunk_into_sentences(chunk: "Chunk") -> list[dict]:
    """
    Split a chunk into sentences using spaCy's sentence segmentation.
    
    Args:
        chunk: The Chunk object to split
        
    Returns:
        List of sentence dictionaries with:
        - text: The sentence text
        - start: Character offset start within chunk
        - end: Character offset end within chunk
        - chunk_id: Reference to the source chunk
    """
    if not chunk.text:
        return []
    
    doc = make_doc(chunk.text)
    spans = sentence_spans_from_doc(doc)
    
    sentences = []
    for start, end, text in spans:
        sentences.append({
            "text": text,
            "start": start,
            "end": end,
            "chunk_id": chunk.id,
        })
    
    return sentences

