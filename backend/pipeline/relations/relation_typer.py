from __future__ import annotations
from typing import Optional

def infer_relation_type(sentence_lower: str) -> Optional[str]:
    """
    Map surface patterns to your RelationEnum values:
    is_a, part_of, influences, similar_to, uses, depends_on, improves

    You can extend/tune this over time.
    """
    # ORDER MATTERS – more specific patterns first

    # part_of
    if " is part of " in sentence_lower or " forms part of " in sentence_lower:
        return "part_of"

    # is_a (taxonomy / definition)
    if " is a " in sentence_lower or " is an " in sentence_lower:
        return "is_a"
    if " are a " in sentence_lower or " are an " in sentence_lower:
        return "is_a"

    # depends_on
    if " depends on " in sentence_lower or " reliant on " in sentence_lower or " relies on " in sentence_lower:
        return "depends_on"

    # uses
    if " uses " in sentence_lower or " utilizes " in sentence_lower or " is used by " in sentence_lower:
        return "uses"

    # improves
    if " improves " in sentence_lower or " enhances " in sentence_lower:
        return "improves"
    if " increases " in sentence_lower or " decreases " in sentence_lower:
        return "influences"  # directional influence

    # influences
    if " influences " in sentence_lower or " affects " in sentence_lower or " impacts " in sentence_lower:
        return "influences"

    # similar_to
    if " similar to " in sentence_lower or " akin to " in sentence_lower:
        return "similar_to"

    return None

