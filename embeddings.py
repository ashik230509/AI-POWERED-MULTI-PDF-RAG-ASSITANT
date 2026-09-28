from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


class EmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = load_embedding_model(model_name)

    def embed(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(384, dtype=np.float32)
        vector = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=False)
        return np.asarray(vector, dtype=np.float32).reshape(-1)

    def embed_many(self, texts: Iterable[str]) -> np.ndarray:
        text_list = list(texts)
        if not text_list:
            return np.empty((0, 384), dtype=np.float32)

        vectors = self.model.encode(text_list, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)
