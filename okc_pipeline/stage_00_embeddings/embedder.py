import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer
from okc_core.config import settings

_model = None

def get_model():
    global _model
    if _model is None:
        # Explicitly set device to avoid meta tensor issues
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            # Try loading with explicit device parameter
            _model = SentenceTransformer(settings.EMBEDDING_MODEL, device=device)
        except (NotImplementedError, RuntimeError) as e:
            error_msg = str(e).lower()
            # Check if this is the meta tensor error
            if "meta tensor" in error_msg or "no data" in error_msg or "cannot copy out" in error_msg:
                # Try loading without explicit device (let SentenceTransformer handle device placement)
                # This can help if there's a device placement issue
                try:
                    _model = SentenceTransformer(settings.EMBEDDING_MODEL)
                except Exception as e2:
                    # If that also fails, the model cache is likely corrupted
                    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
                    raise RuntimeError(
                        f"Failed to load model {settings.EMBEDDING_MODEL}. "
                        f"This is likely due to a corrupted model cache. "
                        f"To fix: clear the cache at {cache_dir}/hub/ "
                        f"or set HF_HOME environment variable to a different location. "
                        f"Original error: {e}, Fallback error: {e2}"
                    ) from e2
            else:
                # Re-raise if it's a different error
                raise
    return _model

def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_model()
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    # ensure 2D
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    return vecs.astype(np.float32)
