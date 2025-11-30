"""Test document fixtures with known expected outputs for pipeline testing."""

from typing import Dict, Any

# Test documents covering various scenarios
TEST_DOCUMENTS: Dict[str, Dict[str, Any]] = {
    "simple_entity": {
        "title": "GPT-4 Introduction",
        "url": "https://example.com/gpt4",
        "text": "GPT-4 is a language model developed by OpenAI. It improves accuracy on various benchmarks.",
        "lang": "en",
    },
    
    "multiple_entities": {
        "title": "BERT Evaluation",
        "url": "https://example.com/bert",
        "text": "BERT is a transformer model developed by Google. It was evaluated on the GLUE benchmark dataset. BERT improves accuracy on natural language understanding tasks.",
        "lang": "en",
    },
    
    "claim_sentences": {
        "title": "Model Comparison",
        "url": "https://example.com/comparison",
        "text": "GPT-4 outperforms GPT-3 on most tasks. BERT increases performance on GLUE. The model depends on large datasets for training.",
        "lang": "en",
    },
    
    "relation_patterns": {
        "title": "Machine Learning Methods",
        "url": "https://example.com/ml",
        "text": "Transformers are a type of neural network architecture. BERT is evaluated on GLUE. Attention mechanisms improve model performance. The method requires GPU acceleration.",
        "lang": "en",
    },
    
    "complex_entities": {
        "title": "Research Paper Summary",
        "url": "https://example.com/paper",
        "text": "The paper by researchers at Stanford University presents a new approach. The method was tested on ImageNet dataset. Results show improvements in accuracy metrics. The work builds on previous research from MIT.",
        "lang": "en",
    },
    
    "hedging_language": {
        "title": "Tentative Claims",
        "url": "https://example.com/hedging",
        "text": "The model might improve performance. It could potentially increase accuracy. The results suggest that the method may work better.",
        "lang": "en",
    },
    
    "is_a_relations": {
        "title": "Definitions",
        "url": "https://example.com/definitions",
        "text": "A transformer is a neural network architecture. BERT is a language model. GPT-4 is an AI system. PyTorch is a machine learning framework.",
        "lang": "en",
    },
    
    "evaluated_on_relations": {
        "title": "Model Evaluations",
        "url": "https://example.com/evaluations",
        "text": "BERT was evaluated on GLUE. GPT-4 was tested on MMLU. ResNet was evaluated on ImageNet. The model was tested on multiple datasets.",
        "lang": "en",
    },
    
    "depends_on_relations": {
        "title": "Dependencies",
        "url": "https://example.com/dependencies",
        "text": "The model depends on large datasets. Training requires GPU resources. The method needs preprocessing steps. The system depends on external APIs.",
        "lang": "en",
    },
    
    "mixed_content": {
        "title": "Comprehensive Example",
        "url": "https://example.com/comprehensive",
        "text": "OpenAI developed GPT-4, a large language model. The model improves accuracy on natural language tasks. GPT-4 was evaluated on various benchmarks including MMLU and GLUE. The system depends on transformer architecture. Transformers are a type of neural network. The research was conducted in San Francisco, California.",
        "lang": "en",
    },
}

