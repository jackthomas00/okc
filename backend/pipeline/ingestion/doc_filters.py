"""
Shared document filtering logic for ML/AI focused content.
Used by ingestion scripts and cleanup tools to ensure consistency.
"""

import re

# Word count bounds
MIN_WORDS = 200
MAX_WORDS = 25000

# Text content keywords - documents must contain at least one of these
# Multi-word phrases (matched as substrings)
PHRASE_FILTERS = [
    # Core ML/AI terms and phrases
    "machine learning", "deep learning", "neural network", "neural networks",
    "self-attention", "multi-head attention",
    "word embedding", "vector embedding",
    "information retrieval", "vector search", "semantic search",
    "knowledge graph", "graph database", "natural language processing",
    "large language model", "language model",
    "diffusion model", "diffusion models", "reinforcement learning",
    "gradient descent", "backpropagation", "tokenization", "tokenizer",
    "retrieval-augmented", "fine-tuning", "transfer learning",
    "supervised learning", "unsupervised learning", "semi-supervised",
    "few-shot", "zero-shot", "prompt engineering", "chain of thought",
    "attention mechanism", "encoder-decoder", "seq2seq", "sequence to sequence"
]

# Single-word keywords (matched with word boundaries to avoid false positives)
# Only include very specific ML/AI terms that are unlikely to appear in non-ML contexts
WORD_FILTERS = [
    # Very specific model names/acronyms (unlikely to appear in non-ML contexts)
    "bert", "gpt", "llm", "openai", "chatgpt", "gemini", "t5", "roberta",
    "alexnet", "resnet", "vgg", "yolo", "rnn", "lstm", "gru",
    "gan", "vae", "autoencoder", "cnn",
    # Note: Removed ambiguous words that appear in non-ML contexts:
    # - "palm" (matches "Palmer" names)
    # - "inception" (common English word)
    # - "transformer" (matches album names, electrical transformers)
    # - "claude" (matches "Claude Debussy", "Claude Monet", etc.)
    # - "neural" (matches "neural pathways" in biology/medicine)
    # - "embedding" (can appear in non-ML contexts)
    # Use phrases instead for these ambiguous terms
]

# Title keywords - if title contains these, accept even if text doesn't match
# (e.g., "BERT (language model)" might not say "bert" in the text)
# Use word boundaries for single words to avoid false positives
TITLE_WORD_FILTERS = [
    # Only very specific terms that are unlikely to appear in non-ML titles
    "bert", "gpt", "gan", "cnn", "rnn", "lstm", "yolo", "vae"
]

TITLE_PHRASE_FILTERS = [
    "machine learning", "deep learning", "artificial intelligence",
    "knowledge graph", "graph database", "neural network", "transformer",
    "language model", "reinforcement learning", "diffusion model"
]


def _matches_word_filter(text: str, word: str) -> bool:
    """Check if text contains a word with word boundaries."""
    # Always use word boundaries to avoid false positives (e.g., "yolo" in "embryology")
    pattern = r'\b' + re.escape(word) + r'\b'
    return bool(re.search(pattern, text, re.IGNORECASE))


def _matches_phrase_filter(text: str, phrase: str) -> bool:
    """Check if text contains a phrase (substring match)."""
    return phrase.lower() in text.lower()


def keep_doc(text: str, title: str = "") -> bool:
    """
    Determine if a document should be kept based on ML/AI filter criteria.
    
    Args:
        text: Document text content
        title: Document title (optional, used as fallback if text doesn't match)
    
    Returns:
        True if document matches filter criteria, False otherwise
    """
    # Quick word bounds check
    wc = len(text.split())
    if wc < MIN_WORDS or wc > MAX_WORDS:
        return False
    
    text_lower = text.lower()
    
    # Check phrase filters first (more specific, less false positives)
    if any(_matches_phrase_filter(text_lower, phrase) for phrase in PHRASE_FILTERS):
        return True
    
    # Check word filters with word boundaries
    if any(_matches_word_filter(text_lower, word) for word in WORD_FILTERS):
        return True
    
    # Also check title - if title suggests ML/AI, accept even if text doesn't have explicit keywords
    # (e.g., "BERT (language model)" might not say "bert" in the text)
    if title:
        title_lower = title.lower()
        # Check title phrases
        if any(_matches_phrase_filter(title_lower, phrase) for phrase in TITLE_PHRASE_FILTERS):
            return True
        # Check title words with word boundaries
        if any(_matches_word_filter(title_lower, word) for word in TITLE_WORD_FILTERS):
            return True
    
    return False


# For SQL queries - list of keywords to match against document text
# Used by rebuild_topics.py for filtering entities by source documents
# Combine phrase and word filters for backward compatibility
DOCUMENT_FILTER_KEYWORDS = PHRASE_FILTERS + WORD_FILTERS
