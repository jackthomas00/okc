from __future__ import annotations
import re
from typing import Iterable

_SENT_SPLIT = re.compile(r"(?<=[\.!?])\s+")
_TOKEN = re.compile(r"\w+|\S")

def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]

def chunk_text(text: str, target_tokens: int = 600, overlap: int = 80) -> list[str]:
    sents = split_sentences(text)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        tokens = _TOKEN.findall(s)
        if cur_len + len(tokens) > target_tokens and cur:
            chunks.append(" ".join(cur))
            # overlap last few sentences
            cur = cur[-3:] if len(cur) > 3 else cur
            cur_len = len(_TOKEN.findall(" ".join(cur)))
        cur.append(s)
        cur_len += len(tokens)
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# ultra-simple entity extractor (NER-lite): proper nouns + capitalized multi-words + code-ish tokens
_CAP_PHRASE = re.compile(r"\b([A-Z][a-zA-Z0-9\-\_]+(?:\s+[A-Z][a-zA-Z0-9\-\_]+){0,3})\b")
_CODEISH = re.compile(r"\b([A-Za-z]+[A-Z][A-Za-z0-9]+|[A-Za-z0-9\-\_]{3,})\b")

def extract_candidate_entities(chunk: str) -> list[str]:
    cands = set()
    for m in _CAP_PHRASE.finditer(chunk):
        phrase = m.group(1).strip()
        if len(phrase.split()) <= 4:
            cands.add(phrase)
    # add obvious tech terms that appear in lower case around code-ish tokens
    for tok in set(_CODEISH.findall(chunk)):
        if len(tok) > 2 and not tok.islower():
            cands.add(tok)
    # normalize basic noise
    cands = {c.strip(".,:;()[]") for c in cands if c.lower() not in {"the","and","of","for","with","from"}}
    return sorted(cands)

def canonicalize(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).lower()
