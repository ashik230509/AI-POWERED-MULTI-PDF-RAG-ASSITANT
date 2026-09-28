from __future__ import annotations

import re
from difflib import get_close_matches
from typing import Any

from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks

        self.tokenized_chunks = [
            self._tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_chunks)

        # Build vocabulary from the uploaded documents.
        self.vocabulary = {
            token
            for tokens in self.tokenized_chunks
            for token in tokens
            if len(token) >= 4
        }

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _expand_query(self, query: str) -> list[str]:
        """
        Keep the original query words and add likely corrections
        for words that do not exist in the document vocabulary.

        Example:
            notise -> notice
        """

        original_tokens = self._tokenize(query)
        expanded_tokens = list(original_tokens)

        for token in original_tokens:
            # Ignore short words.
            if len(token) < 4:
                continue

            # Word already exists in the documents.
            if token in self.vocabulary:
                continue

            # Find close vocabulary matches.
            matches = get_close_matches(
                token,
                self.vocabulary,
                n=1,
                cutoff=0.75,
            )

            if matches:
                corrected = matches[0]

                # Keep both original + corrected word.
                expanded_tokens.append(corrected)

        return expanded_tokens

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        query_tokens = self._expand_query(query)

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results = []

        for index in ranked_indices:
            chunk = self.chunks[index]

            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "document_name": chunk["document_name"],
                    "page_number": chunk["page_number"],
                    "text": chunk["text"],
                    "score": float(scores[index]),
                }
            )

        return results