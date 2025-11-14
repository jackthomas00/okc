import numpy as np
from sentence_transformers import SentenceTransformer
from api.config import settings

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model

def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_model()
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    # ensure 2D
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    return vecs.astype(np.float32)
