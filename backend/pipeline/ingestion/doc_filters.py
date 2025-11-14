"""
Shared document filtering logic for ML/AI focused content.
Used by ingestion scripts and cleanup tools to ensure consistency.
"""

# Word count bounds
MIN_WORDS = 200
MAX_WORDS = 25000

# Text content keywords - documents must contain at least one of these
LOWER_FILTERS = [
    # Core ML/AI terms and phrases
    "machine learning", "deep learning", "neural network", "neural networks",
    "transformer", "transformers", "self-attention", "multi-head attention",
    "word embedding", "embeddings", "vector embedding",
    "information retrieval", "vector search", "semantic search",
    "knowledge graph", "graph database", "natural language processing",
    "bert", "gpt", "llm", "large language model", "language model",
    "diffusion model", "diffusion models", "reinforcement learning",
    "gradient descent", "backpropagation", "tokenization", "tokenizer",
    "retrieval-augmented", "rag", "fine-tuning", "transfer learning",
    # Model names (check both text and title)
    "openai", "chatgpt", "claude", "gemini", "palm", "t5", "roberta",
    "alexnet", "resnet", "vgg", "inception", "yolo", "rnn", "lstm", "gru",
    "gan", "vae", "autoencoder", "cnn", "convolutional", "residual",
    # Techniques and concepts
    "supervised learning", "unsupervised learning", "semi-supervised",
    "few-shot", "zero-shot", "prompt engineering", "chain of thought",
    "attention mechanism", "encoder-decoder", "seq2seq", "sequence to sequence"
]

# Title keywords - if title contains these, accept even if text doesn't match
# (e.g., "BERT (language model)" might not say "bert" in the text)
TITLE_FILTERS = [
    "bert", "gpt", "transformer", "neural", "machine learning", "deep learning",
    "embedding", "language model", "nlp", "ai", "artificial intelligence",
    "reinforcement", "diffusion", "gan", "cnn", "rnn", "lstm", "attention",
    "retrieval", "vector", "semantic", "knowledge graph", "graph database"
]


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
    
    # Check text content first
    lt = text.lower()
    if any(k in lt for k in LOWER_FILTERS):
        return True
    
    # Also check title - if title suggests ML/AI, accept even if text doesn't have explicit keywords
    # (e.g., "BERT (language model)" might not say "bert" in the text)
    if title:
        title_lower = title.lower()
        if any(k in title_lower for k in TITLE_FILTERS):
            return True
    
    return False


# For SQL queries - list of keywords to match against document text
# Used by rebuild_topics.py for filtering entities by source documents
DOCUMENT_FILTER_KEYWORDS = LOWER_FILTERS.copy()

