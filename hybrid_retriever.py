from __future__ import annotations

from typing import Any


class HybridRetriever:
    def __init__(self, vector_hits: list[dict[str, Any]] | None = None, bm25_hits: list[dict[str, Any]] | None = None):
        self.vector_hits = vector_hits or []
        self.bm25_hits = bm25_hits or []

    @staticmethod
    def _rrf(results_a: list[dict[str, Any]], results_b: list[dict[str, Any]], k: int = 60) -> list[dict[str, Any]]:
        combined: dict[str, dict[str, Any]] = {}

        for rank, result in enumerate(results_a, start=1):
            chunk_id = result["chunk_id"]
            combined.setdefault(chunk_id, {**result, "score": 0.0})
            combined[chunk_id]["score"] += 1.0 / (k + rank)

        for rank, result in enumerate(results_b, start=1):
            chunk_id = result["chunk_id"]
            combined.setdefault(chunk_id, {**result, "score": 0.0})
            combined[chunk_id]["score"] += 1.0 / (k + rank)

        return sorted(combined.values(), key=lambda item: item["score"], reverse=True)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.vector_hits and not self.bm25_hits:
            return []
        results = self._rrf(self.vector_hits, self.bm25_hits)
        return results[:top_k]
