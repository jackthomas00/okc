import re, hashlib
from typing import Optional

_WS = re.compile(r"\s+")

def normalize_text(t: str) -> str:
    # collapse whitespace, strip control chars
    t = t.replace("\u00A0", " ")
    t = _WS.sub(" ", t).strip()
    return t

def content_sha1(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8", errors="ignore")).hexdigest()

def word_count(t: str) -> int:
    return len([w for w in re.findall(r"\b\w+\b", t)])

# Optional: if you want language detection, add langdetect to requirements and uncomment.
def detect_lang_optional(t: str) -> Optional[str]:
    try:
        from langdetect import detect  # pip install langdetect
        return detect(t)
    except Exception:
        return None
