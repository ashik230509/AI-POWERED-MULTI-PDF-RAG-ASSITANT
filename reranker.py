from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL


@lru_cache(maxsize=1)
def load_reranker_model(model_name: str = RERANKER_MODEL) -> CrossEncoder:
    try:
        return CrossEncoder(model_name)
    except Exception as exc:  # pragma: no cover - runtime model availability guard
        raise RuntimeError(f"Unable to load reranker model '{model_name}': {exc}") from exc


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self.model = load_reranker_model(model_name)

    def rerank(self, candidates: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
        if not candidates:
            return []

        pairs = [[query, candidate["text"]] for candidate in candidates]
        try:
            scores = self.model.predict(pairs)
        except Exception:
            return sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)

        if isinstance(scores, np.ndarray):
            score_values = scores.reshape(-1)
        elif isinstance(scores, list):
            score_values = np.asarray(scores, dtype=float).reshape(-1)
        else:
            score_values = np.asarray([scores], dtype=float).reshape(-1)

        reranked: list[dict[str, Any]] = []
        for candidate, score in zip(candidates, score_values):
                    reranked: list[dict[str, Any]] = []

        for candidate, score in zip(candidates, score_values):
            item = dict(candidate)

            # Preserve the original Hybrid/RRF retrieval score.
            item["retrieval_score"] = float(candidate.get("score", 0.0))

            # Store the CrossEncoder score separately.
            item["rerank_score"] = float(score)

            # Keep "score" as the reranker score for existing code compatibility.
            item["score"] = float(score)

            reranked.append(item)

        return sorted(
            reranked,
            key=lambda item: float(item.get("rerank_score", 0.0)),
            reverse=True,
        )
        
        reranked.append(item)
        return sorted(reranked, key=lambda item: float(item.get("score", 0.0)), reverse=True)
