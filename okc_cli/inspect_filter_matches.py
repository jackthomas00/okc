#!/usr/bin/env python3
"""
Inspect which documents are matching the filter and why.
Shows sample documents and what keywords they matched.
"""
from sqlalchemy import select
from okc_core.db import SessionLocal
from okc_core.models import Document
from okc_pipeline.stage_00_ingestion.doc_filters import (
    keep_doc, PHRASE_FILTERS, WORD_FILTERS, TITLE_PHRASE_FILTERS, TITLE_WORD_FILTERS,
    _matches_phrase_filter, _matches_word_filter
)

def find_matching_keywords(text: str, title: str = "") -> dict:
    """Find which keywords matched for a document."""
    text_lower = text.lower()
    title_lower = title.lower() if title else ""
    
    matched_phrases = [p for p in PHRASE_FILTERS if _matches_phrase_filter(text_lower, p)]
    matched_words = [w for w in WORD_FILTERS if _matches_word_filter(text_lower, w)]
    matched_title_phrases = [p for p in TITLE_PHRASE_FILTERS if _matches_phrase_filter(title_lower, p)] if title else []
    matched_title_words = [w for w in TITLE_WORD_FILTERS if _matches_word_filter(title_lower, w)] if title else []
    
    return {
        "phrases": matched_phrases,
        "words": matched_words,
        "title_phrases": matched_title_phrases,
        "title_words": matched_title_words,
    }

def main():
    db = SessionLocal()
    try:
        docs = list[Document](db.scalars(select(Document).limit(20)))
        
        print(f"Analyzing {len(docs)} documents...\n")
        
        for i, doc in enumerate(docs, 1):
            title = doc.title or ""
            text = doc.text or ""
            word_count = len(text.split())
            
            matches = find_matching_keywords(text, title)
            passes = keep_doc(text, title=title)
            
            print(f"{'='*80}")
            print(f"Document {i}: {title[:60]}")
            print(f"  Word count: {word_count}")
            print(f"  Passes filter: {passes}")
            
            if matches["phrases"]:
                print(f"  Matched phrases: {', '.join(matches['phrases'][:3])}")
            if matches["words"]:
                print(f"  Matched words: {', '.join(matches['words'][:5])}")
            if matches["title_phrases"]:
                print(f"  Matched title phrases: {', '.join(matches['title_phrases'])}")
            if matches["title_words"]:
                print(f"  Matched title words: {', '.join(matches['title_words'][:3])}")
            
            if not any([matches["phrases"], matches["words"], matches["title_phrases"], matches["title_words"]]):
                print(f"  ⚠️  NO MATCHES FOUND - but filter returned True!")
                print(f"  Text preview: {text[:200]}...")
            
            print()
            
    finally:
        db.close()

if __name__ == "__main__":
    main()

