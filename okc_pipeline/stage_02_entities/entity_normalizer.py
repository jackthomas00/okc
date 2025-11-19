"""
Entity name normalization for deduplication and merging.

Normalizes entity names by lowercasing, stripping punctuation (except hyphens/underscores),
and normalizing whitespace.
"""

import re
from okc_pipeline.utils.text_cleaning import canonicalize


def normalize_entity_name(name: str) -> str:
    """
    Normalize an entity name for deduplication and merging.
    
    Normalization steps:
    - Lowercase
    - Strip leading articles (the, a, an)
    - Strip punctuation (keep hyphens and underscores for names like "GPT-4", "ImageNet")
    - Preserve numeric formatting (numbers, currency, decimals, commas in numbers)
    - Normalize whitespace
    
    Args:
        name: Entity name to normalize
    
    Returns:
        Normalized entity name
    """
    if not name:
        return ""
    
    # Use existing canonicalize for basic normalization
    normalized = canonicalize(name)
    
    # Remove leading articles (the, a, an) followed by whitespace
    # This handles cases like "The Apple ii" -> "apple ii"
    normalized = re.sub(r'^(the|a|an)\s+', '', normalized, flags=re.IGNORECASE)
    
    # Check if the entity contains numbers
    # If so, preserve numeric formatting to avoid merging different numeric entities
    has_numbers = bool(re.search(r'\d', normalized))
    
    if has_numbers:
        # For entities with numbers, preserve numeric formatting:
        # - Keep digits, decimal points, commas (for number formatting)
        # - Keep currency symbols ($, €, £, ¥)
        # - Keep hyphens and underscores (for names like "GPT-4")
        # - Remove other punctuation but preserve the numeric structure
        
        # Pattern to match and protect numeric values:
        # - Currency symbols followed by numbers: $7.5, €100, £1,234.56
        # - Formatted numbers: 1,234.56, 6,500, 25.00
        # - Simple numbers: 2500, 75
        
        # Split into tokens and process each
        # Protect numeric patterns, clean other parts
        tokens = re.split(r'(\s+)', normalized)  # Split on whitespace but keep it
        result_tokens = []
        
        for token in tokens:
            if not token.strip():  # Whitespace
                result_tokens.append(token)
                continue
            
            # Check if token looks like a number (with optional currency, commas, decimals)
            # Pattern allows: currency? digits (with optional comma groups) (optional decimal part)
            # Examples: $7.5, 1,234.56, 1995, 19.95, 59, 5.9
            if re.match(r'^[\$€£¥]?\s*\d+(?:,\d{3})*(?:\.\d+)?$', token, re.IGNORECASE):
                # It's a number - preserve it exactly (but lowercase currency if present)
                protected = token.lower()
                result_tokens.append(protected)
            elif re.search(r'\d', token):
                # Token contains digits but isn't a pure number (e.g., "$7.5 million")
                # Preserve digits and numeric punctuation, remove other punctuation
                # Keep: digits, decimal points, commas, currency symbols, letters, hyphens, underscores
                cleaned = re.sub(r'[^\w\s\.,\$€£¥\-_]', '', token)
                result_tokens.append(cleaned.lower())
            else:
                # No numbers - remove punctuation except hyphens and underscores
                cleaned = re.sub(r'[^\w\s\-_]', '', token)
                if cleaned:
                    result_tokens.append(cleaned.lower())
        
        normalized = ''.join(result_tokens)
    else:
        # For non-numeric entities, remove punctuation except hyphens and underscores
        # This preserves names like "GPT-4", "ImageNet", "BERT-base"
        normalized = re.sub(r'[^\w\s\-_]', '', normalized)
    
    # Normalize whitespace again after punctuation removal
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

