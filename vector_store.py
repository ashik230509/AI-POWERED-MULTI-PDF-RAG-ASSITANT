from __future__ import annotations

from typing import Any

import faiss
import numpy as np

from embeddings import EmbeddingModel


class VectorStore:
    def __init__(self, embedding_model: EmbeddingModel | None = None):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.chunks: list[dict[str, Any]] = []
        self.index: faiss.Index | None = None

    def build_from_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            raise ValueError("No chunks available to build a vector index.")

        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.embed_many(texts)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        normed = embeddings.astype(np.float32)
        faiss.normalize_L2(normed)
        self.index.add(normed)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.chunks or self.index is None:
            return []

        query_vector = self.embedding_model.embed(query).astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        limit = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_vector, limit)

        results: list[dict[str, Any]] = []
        for index_position, score in zip(indices[0], scores[0]):
            if index_position < 0:
                continue
            chunk = self.chunks[int(index_position)]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "document_name": chunk["document_name"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
                "score": float(score),
            })
        return results
