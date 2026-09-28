from __future__ import annotations

from typing import Any

from bm25_retriever import BM25Retriever
from chunker import chunk_pages
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_CONTEXT_CHUNKS,
    TOP_K,
)
from embeddings import EmbeddingModel
from hybrid_retriever import HybridRetriever
from llm import OllamaClient
from pdf_processor import extract_documents
from reranker import CrossEncoderReranker
from vector_store import VectorStore


class RAGPipeline:
    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
        bm25_retriever: BM25Retriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        llm_client: OllamaClient | None = None,
    ) -> None:

        self.embedding_model = (
            embedding_model
            or EmbeddingModel()
        )

        self.vector_store = (
            vector_store
            or VectorStore(
                self.embedding_model
            )
        )

        self.bm25_retriever = (
            bm25_retriever
        )

        self.reranker = (
            reranker
            or CrossEncoderReranker()
        )

        self.llm_client = (
            llm_client
            or OllamaClient()
        )

        self.chunks: list[
            dict[str, Any]
        ] = []

    # =========================================================
    # PROCESS DOCUMENTS
    # =========================================================

    def process_documents(
        self,
        uploaded_files: list[Any],
    ) -> int:

        pages = extract_documents(
            uploaded_files
        )

        self.chunks = chunk_pages(
            pages,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        if not self.chunks:
            raise ValueError(
                "No usable content was produced "
                "from the uploaded PDFs."
            )

        # Rebuild vector store
        # for the current documents.
        self.vector_store = VectorStore(
            self.embedding_model
        )

        self.vector_store.build_from_chunks(
            self.chunks
        )

        # Rebuild BM25 index.
        self.bm25_retriever = BM25Retriever(
            self.chunks
        )

        return len(self.chunks)

    # =========================================================
    # FOLLOW-UP QUESTION RESOLUTION
    # =========================================================

    def _resolve_retrieval_question(
        self,
        question: str,
        chat_history: list[
            dict[str, str]
        ],
    ) -> str:

        if not chat_history:
            return question

        normalized = (
            question
            .strip()
            .lower()
        )

        follow_up_patterns = (
            "what about",
            "how about",
            "and what about",
            "what else",
            "what is it",
            "how much is it",
            "when is it",
            "where is it",
            "who is it",
            "tell me more",
        )

        is_follow_up = (
            normalized.startswith(
                follow_up_patterns
            )
            or normalized in {
                "and?",
                "what?",
                "what about it?",
                "what about that?",
                "how about it?",
                "how about that?",
            }
        )

        if not is_follow_up:
            return question

        try:

            resolved = (
                self.llm_client.resolve_follow_up(
                    chat_history,
                    question,
                )
            )

        except RuntimeError:

            return question

        resolved = resolved.strip()

        return (
            resolved
            if resolved
            else question
        )

    # =========================================================
    # MAIN QUESTION ANSWER
    # =========================================================

    def answer_question(
        self,
        question: str,
        chat_history: list[
            dict[str, str]
        ] | None = None,
    ) -> dict[str, Any]:

        chat_history = (
            chat_history
            or []
        )

        # -----------------------------------------------------
        # Empty question
        # -----------------------------------------------------

        if not question or not question.strip():

            return {
                "answer": (
                    "Please ask a question "
                    "about the uploaded documents."
                ),
                "citations": [],
            }

        # -----------------------------------------------------
        # Greeting
        # -----------------------------------------------------

        if self._is_greeting(question):

            return {
                "answer": (
                    "Hi! I can help you with your "
                    "uploaded documents. Upload a PDF, "
                    "then ask me anything about it."
                ),
                "citations": [],
            }

        # -----------------------------------------------------
        # General conversation
        # -----------------------------------------------------

        if self._is_general_chat(question):

            return {
                "answer": (
                    "I'm here to help with your uploaded "
                    "documents. Upload a PDF and ask me "
                    "a question about it."
                ),
                "citations": [],
            }

        # -----------------------------------------------------
        # Make sure documents exist
        # -----------------------------------------------------

        if not self.chunks:

            return {
                "answer": (
                    "Please upload and process one or "
                    "more PDFs before asking a question."
                ),
                "citations": [],
            }

        # =====================================================
        # RESOLVE FOLLOW-UP
        # =====================================================

        retrieval_question = (
            self._resolve_retrieval_question(
                question,
                chat_history,
            )
        )

        print(
            "\n"
            + "=" * 60
        )

        print(
            "RETRIEVAL QUESTION:"
        )

        print(
            retrieval_question
        )

        print(
            "=" * 60
        )

        # =====================================================
        # MULTI-PART QUESTION PATH
        # =====================================================
        # Keep the existing single-question retrieval path unchanged.
        # Only questions containing multiple independent requests use
        # this path.
        # =====================================================

        if self._is_multi_part_question(retrieval_question):
            return self._answer_multi_part_question(
                original_question=question,
                retrieval_question=retrieval_question,
                chat_history=chat_history,
            )

        # =====================================================
        # VECTOR SEARCH
        # =====================================================

        vector_hits = (
            self.vector_store.search(
                retrieval_question,
                top_k=TOP_K * 4,
            )
        )

        # =====================================================
        # BM25 SEARCH
        # =====================================================

        if self.bm25_retriever is None:

            self.bm25_retriever = (
                BM25Retriever(
                    self.chunks
                )
            )

        bm25_hits = (
            self.bm25_retriever.search(
                retrieval_question,
                top_k=TOP_K * 4,
            )
        )

        # =====================================================
        # HYBRID RRF
        # =====================================================

        hybrid_hits = (
            HybridRetriever(
                vector_hits,
                bm25_hits,
            ).search(
                retrieval_question,
                top_k=TOP_K * 4,
            )
        )

        if not hybrid_hits:

            return {
                "answer": (
                    "I couldn't find sufficient information "
                    "to answer this question from the "
                    "uploaded documents."
                ),
                "citations": [],
            }

        # =====================================================
        # DEBUG HYBRID RESULTS
        # =====================================================

        print(
            "\n"
            + "=" * 60
        )

        print(
            "HYBRID RETRIEVAL RESULTS"
        )

        print(
            "=" * 60
        )

        for index, hit in enumerate(
            hybrid_hits,
            start=1,
        ):

            print(
                f"\nResult {index}"
            )

            print(
                "Chunk:",
                hit.get(
                    "chunk_id"
                ),
            )

            print(
                "Document:",
                hit.get(
                    "document_name"
                ),
            )

            print(
                "Page:",
                hit.get(
                    "page_number"
                ),
            )

            print(
                "RRF:",
                hit.get(
                    "rrf_score"
                ),
            )

            print(
                "Text:",
                str(
                    hit.get(
                        "text",
                        "",
                    )
                )[:300],
            )

        # =====================================================
        # CROSS ENCODER RERANKING
        # =====================================================

        reranked_hits = (
            self.reranker.rerank(
                hybrid_hits,
                retrieval_question,
            )
        )

        if not reranked_hits:

            return {
                "answer": (
                    "I couldn't find sufficient information "
                    "to answer this question from the "
                    "uploaded documents."
                ),
                "citations": [],
            }

        # =====================================================
        # DEBUG RERANKED RESULTS
        # =====================================================

        print(
            "\n"
            + "=" * 60
        )

        print(
            "RERANKED RESULTS"
        )

        print(
            "=" * 60
        )

        for index, hit in enumerate(
            reranked_hits,
            start=1,
        ):

            print(
                f"\nResult {index}"
            )

            print(
                "Chunk:",
                hit.get(
                    "chunk_id"
                ),
            )

            print(
                "Document:",
                hit.get(
                    "document_name"
                ),
            )

            print(
                "Page:",
                hit.get(
                    "page_number"
                ),
            )

            print(
                "Rerank:",
                hit.get(
                    "rerank_score"
                ),
            )

            print(
                "RRF:",
                hit.get(
                    "rrf_score"
                ),
            )

            print(
                "Text:",
                str(
                    hit.get(
                        "text",
                        "",
                    )
                )[:400],
            )

        # =====================================================
        # CANDIDATE POOL
        # =====================================================

        candidate_hits = (
            reranked_hits[
                : TOP_K * 4
            ]
        )

        if not candidate_hits:

            return {
                "answer": (
                    "I couldn't find sufficient information "
                    "to answer this question from the "
                    "uploaded documents."
                ),
                "citations": [],
            }

        # =====================================================
        # EVIDENCE VERIFICATION
        # =====================================================
        #
        # IMPORTANT:
        #
        # The verifier is NOT used to remove chunks.
        #
        # It is only used as a diagnostic signal.
        #
        # This prevents the verifier from accidentally
        # removing valid chunks in multi-part questions.
        #
        # Example:
        #
        # "What are all 4 machine learning tasks?"
        #
        # A verifier may identify only 3 of the 4 chunks.
        # We do NOT want that to destroy the retrieval result.
        # =====================================================

        try:

            verification = (
                self.llm_client.verify_evidence(
                    retrieval_question,
                    candidate_hits,
                )
            )

        except Exception as exc:

            verification = {
                "supported": False,
                "chunk_ids": [],
                "reason": (
                    f"Verification failed: {exc}"
                ),
            }

        print(
            "\n"
            + "=" * 60
        )

        print(
            "EVIDENCE VERIFICATION"
        )

        print(
            "=" * 60
        )

        print(
            "Supported:",
            verification.get(
                "supported",
                False,
            ),
        )

        print(
            "Verified chunk IDs:",
            verification.get(
                "chunk_ids",
                [],
            ),
        )

        print(
            "Reason:",
            verification.get(
                "reason",
                "",
            ),
        )

        # =====================================================
        # FINAL CONTEXT SELECTION
        # =====================================================
        #
        # Use CrossEncoder ranking as the actual selection
        # mechanism.
        #
        # The verifier does NOT filter these chunks.
        # =====================================================

        selected_hits = (
            candidate_hits[
                :MAX_CONTEXT_CHUNKS
            ]
        )

        if not selected_hits:

            return {
                "answer": (
                    "I couldn't find sufficient information "
                    "to answer this question from the "
                    "uploaded documents."
                ),
                "citations": [],
            }

        # =====================================================
        # DEBUG FINAL CONTEXT
        # =====================================================

        print(
            "\n"
            + "=" * 60
        )

        print(
            "FINAL SELECTED CONTEXT"
        )

        print(
            "=" * 60
        )

        for index, hit in enumerate(
            selected_hits,
            start=1,
        ):

            print(
                f"\nContext {index}"
            )

            print(
                "Chunk:",
                hit.get(
                    "chunk_id"
                ),
            )

            print(
                "Document:",
                hit.get(
                    "document_name"
                ),
            )

            print(
                "Page:",
                hit.get(
                    "page_number"
                ),
            )

            print(
                "Text:",
                str(
                    hit.get(
                        "text",
                        "",
                    )
                )[:500],
            )

        # =====================================================
        # BUILD CONTEXT
        # =====================================================

        context_parts = []

        for hit in selected_hits:

            document_name = (
                hit.get(
                    "document_name"
                )
                or "Unknown document"
            )

            page_number = (
                hit.get(
                    "page_number"
                )
                or 0
            )

            text = (
                hit.get(
                    "text"
                )
                or hit.get(
                    "content",
                    "",
                )
            )

            context_parts.append(
                (
                    f"Document: "
                    f"{document_name}\n"
                    f"Page: "
                    f"{page_number}\n"
                    f"Content:\n"
                    f"{text}"
                )
            )

        context = (
            "\n\n"
            + ("\n\n" + "-" * 50 + "\n\n").join(
                context_parts
            )
        )

        # =====================================================
        # BUILD LLM MESSAGES
        # =====================================================

        messages = (
            self._build_messages(
                current_question=question,
                chat_history=chat_history,
                context=context,
            )
        )

        # =====================================================
        # FINAL LLM GENERATION
        # =====================================================

        try:

            answer_text = (
                self.llm_client.generate(
                    messages,
                    temperature=0.1,
                )
            )

        except RuntimeError as exc:

            return {
                "answer": str(exc),
                "citations": (
                    self._citation_metadata(
                        selected_hits
                    )
                ),
            }

        answer = (
            answer_text.strip()
        )

        if not answer:

            answer = (
                "I couldn't find sufficient information "
                "to answer this question from the "
                "uploaded documents."
            )

        # =====================================================
        # CITATIONS
        # =====================================================

        citations = (
            self._citation_metadata(
                selected_hits
            )
        )

        return {
            "answer": answer,
            "citations": citations,
        }

    # =========================================================
    # MULTI-PART QUESTION SUPPORT
    # =========================================================

    @staticmethod
    def _is_multi_part_question(question: str) -> bool:
        """Detect questions containing multiple independent requests."""
        normalized = question.strip().lower()

        if not normalized:
            return False

        # A conjunction is treated as multi-part only when both sides
        # look like independent requests. This keeps ordinary questions
        # such as "What is Task 1 and its objective?" useful as well.
        if " and " in normalized:
            left, right = normalized.split(" and ", 1)
            return bool(left.strip()) and bool(right.strip())

        return False

    @staticmethod
    def _split_multi_part_question(question: str) -> list[str]:
        """Split a multi-part question without using document-specific rules."""
        parts = []

        for part in question.split(" and "):
            cleaned = part.strip(" ,;\n\t")
            if cleaned:
                # Preserve question intent for retrieval.
                if not cleaned.endswith("?"):
                    cleaned += "?"
                parts.append(cleaned)

        return parts

    def _retrieve_and_rerank(
        self,
        query: str,
    ) -> list[dict[str, Any]]:
        """Run the normal vector + BM25 + RRF + CrossEncoder pipeline."""

        vector_hits = self.vector_store.search(
            query,
            top_k=TOP_K * 4,
        )

        if self.bm25_retriever is None:
            self.bm25_retriever = BM25Retriever(self.chunks)

        bm25_hits = self.bm25_retriever.search(
            query,
            top_k=TOP_K * 4,
        )

        hybrid_hits = HybridRetriever(
            vector_hits,
            bm25_hits,
        ).search(
            query,
            top_k=TOP_K * 4,
        )

        if not hybrid_hits:
            return []

        return self.reranker.rerank(
            hybrid_hits,
            query,
        )

    @staticmethod
    def _merge_multi_part_results(
        result_sets: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Merge results from independent question parts by chunk ID."""

        merged: dict[str, dict[str, Any]] = {}

        for results in result_sets:
            for hit in results:
                chunk_id = hit.get("chunk_id")
                if not chunk_id:
                    continue

                existing = merged.get(chunk_id)

                if existing is None:
                    merged[chunk_id] = dict(hit)
                    continue

                current_score = float(
                    existing.get("rerank_score", float("-inf"))
                    if existing.get("rerank_score") is not None
                    else float("-inf")
                )
                new_score = float(
                    hit.get("rerank_score", float("-inf"))
                    if hit.get("rerank_score") is not None
                    else float("-inf")
                )

                if new_score > current_score:
                    merged[chunk_id] = dict(hit)

        return sorted(
            merged.values(),
            key=lambda item: float(
                item.get("rerank_score", float("-inf"))
                if item.get("rerank_score") is not None
                else float("-inf")
            ),
            reverse=True,
        )

    def _answer_multi_part_question(
        self,
        original_question: str,
        retrieval_question: str,
        chat_history: list[dict[str, str]],
    ) -> dict[str, Any]:
        """
        Handle only multi-part questions.

        Each part is retrieved independently so one part cannot push the
        evidence for another part out of the final context.
        """

        parts = self._split_multi_part_question(retrieval_question)

        if len(parts) < 2:
            # Safety fallback to the original single-question path.
            return self._answer_single_question_path(
                question=original_question,
                retrieval_question=retrieval_question,
                chat_history=chat_history,
            )

        print("\n" + "=" * 60)
        print("MULTI-PART QUESTIONS")
        print("=" * 60)
        for index, part in enumerate(parts, start=1):
            print(f"Part {index}: {part}")

        result_sets = []
        part_best_hits = []

        for part in parts:
            reranked = self._retrieve_and_rerank(part)

            if reranked:
                result_sets.append(reranked[: TOP_K * 4])
                part_best_hits.append(reranked[:MAX_CONTEXT_CHUNKS])

        if not result_sets:
            return {
                "answer": (
                    "I couldn't find sufficient information to answer "
                    "this question from the uploaded documents."
                ),
                "citations": [],
            }

        merged_hits = self._merge_multi_part_results(result_sets)

        # Guarantee that every independently retrieved question part has
        # representation in the context before filling remaining slots by
        # overall reranker score. No document-specific terms are used.
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()

        for hits in part_best_hits:
            if not hits:
                continue

            best = hits[0]
            chunk_id = best.get("chunk_id")
            if chunk_id and chunk_id not in selected_ids:
                selected.append(best)
                selected_ids.add(chunk_id)

        for hit in merged_hits:
            if len(selected) >= MAX_CONTEXT_CHUNKS:
                break

            chunk_id = hit.get("chunk_id")
            if not chunk_id or chunk_id in selected_ids:
                continue

            selected.append(hit)
            selected_ids.add(chunk_id)

        if not selected:
            return {
                "answer": (
                    "I couldn't find sufficient information to answer "
                    "this question from the uploaded documents."
                ),
                "citations": [],
            }

        print("\n" + "=" * 60)
        print("MULTI-PART FINAL CONTEXT")
        print("=" * 60)

        for index, hit in enumerate(selected, start=1):
            print(f"\nContext {index}")
            print("Chunk:", hit.get("chunk_id"))
            print("Document:", hit.get("document_name"))
            print("Page:", hit.get("page_number"))
            print("Text:", str(hit.get("text", ""))[:500])

        context_parts = []

        for hit in selected:
            document_name = hit.get("document_name") or "Unknown document"
            page_number = hit.get("page_number") or 0
            text = hit.get("text") or hit.get("content", "")

            context_parts.append(
                f"Document: {document_name}\n"
                f"Page: {page_number}\n"
                f"Content:\n{text}"
            )

        context = "\n\n" + ("\n\n" + "-" * 50 + "\n\n").join(context_parts)

        messages = self._build_messages(
            current_question=original_question,
            chat_history=chat_history,
            context=context,
        )

        try:
            answer_text = self.llm_client.generate(
                messages,
                temperature=0.1,
            )
        except RuntimeError as exc:
            return {
                "answer": str(exc),
                "citations": self._citation_metadata(selected),
            }

        answer = answer_text.strip()

        if not answer:
            answer = (
                "I couldn't find sufficient information to answer "
                "this question from the uploaded documents."
            )

        return {
            "answer": answer,
            "citations": self._citation_metadata(selected),
        }

    def _answer_single_question_path(
        self,
        question: str,
        retrieval_question: str,
        chat_history: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Fallback helper; the normal path remains unchanged in answer_question."""

        reranked_hits = self._retrieve_and_rerank(retrieval_question)

        if not reranked_hits:
            return {
                "answer": (
                    "I couldn't find sufficient information to answer "
                    "this question from the uploaded documents."
                ),
                "citations": [],
            }

        selected_hits = reranked_hits[:MAX_CONTEXT_CHUNKS]

        context_parts = []
        for hit in selected_hits:
            context_parts.append(
                f"Document: {hit.get('document_name', 'Unknown document')}\n"
                f"Page: {hit.get('page_number', 0)}\n"
                f"Content:\n{hit.get('text', '')}"
            )

        context = "\n\n" + ("\n\n" + "-" * 50 + "\n\n").join(context_parts)

        messages = self._build_messages(
            current_question=question,
            chat_history=chat_history,
            context=context,
        )

        try:
            answer_text = self.llm_client.generate(
                messages,
                temperature=0.1,
            )
        except RuntimeError as exc:
            return {
                "answer": str(exc),
                "citations": self._citation_metadata(selected_hits),
            }

        return {
            "answer": answer_text.strip() or (
                "I couldn't find sufficient information to answer "
                "this question from the uploaded documents."
            ),
            "citations": self._citation_metadata(selected_hits),
        }

    # =========================================================
    # CITATION METADATA
    # =========================================================

    @staticmethod
    def _citation_metadata(
        chunks: list[
            dict[str, Any]
        ],
    ) -> list[
        dict[str, Any]
    ]:

        seen: set[
            tuple[str, int]
        ] = set()

        citations: list[
            dict[str, Any]
        ] = []

        for chunk in chunks:

            document_name = str(
                chunk.get(
                    "document_name"
                )
                or "Unknown document"
            )

            try:

                page_number = int(
                    chunk.get(
                        "page_number"
                    )
                    or 0
                )

            except (
                TypeError,
                ValueError,
            ):

                page_number = 0

            key = (
                document_name,
                page_number,
            )

            if key in seen:
                continue

            seen.add(key)

            citations.append(
                {
                    "chunk_id": chunk.get(
                        "chunk_id"
                    ),
                    "document_name":
                        document_name,
                    "page_number":
                        page_number,
                    "text":
                        chunk.get(
                            "text"
                        )
                        or chunk.get(
                            "content",
                            "",
                        ),
                }
            )

        return citations

    # =========================================================
    # BUILD LLM MESSAGES
    # =========================================================

    def _build_messages(
        self,
        current_question: str,
        chat_history: list[
            dict[str, str]
        ],
        context: str,
    ) -> list[
        dict[str, str]
    ]:

        messages: list[
            dict[str, str]
        ] = []

        # -----------------------------------------------------
        # SYSTEM PROMPT
        # -----------------------------------------------------

        system_prompt = (
            self.llm_client.build_system_prompt()
            + "\n\n"
            "Additional instruction:\n"
            "Use the conversation history when the "
            "question is a follow-up. However, the final "
            "answer must remain grounded in the supplied "
            "document context."
        )

        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

        # -----------------------------------------------------
        # CHAT HISTORY
        # -----------------------------------------------------

        for item in chat_history[-6:]:

            role = item.get(
                "role",
                "",
            )

            content = str(
                item.get(
                    "content",
                    "",
                )
            ).strip()

            if role not in {
                "user",
                "assistant",
            }:

                continue

            if not content:
                continue

            messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        # -----------------------------------------------------
        # CURRENT QUESTION + CONTEXT
        # -----------------------------------------------------

        messages.append(
            {
                "role": "user",
                "content": (
                    f"Question:\n"
                    f"{current_question}\n\n"
                    f"Document Context:\n"
                    f"{context}"
                ),
            }
        )

        return messages

    # =========================================================
    # GREETING DETECTION
    # =========================================================

    @staticmethod
    def _is_greeting(
        question: str,
    ) -> bool:

        normalized = (
            question
            .strip()
            .lower()
        )

        greetings = {
            "hi",
            "hello",
            "hey",
            "hi there",
            "hello there",
            "good morning",
            "good afternoon",
            "good evening",
        }

        if normalized in greetings:
            return True

        greeting_starts = (
            "hi ",
            "hello ",
            "hey ",
            "good morning ",
            "good afternoon ",
            "good evening ",
        )

        return normalized.startswith(
            greeting_starts
        )

    # =========================================================
    # GENERAL CHAT DETECTION
    # =========================================================

    @staticmethod
    def _is_general_chat(
        question: str,
    ) -> bool:

        normalized = (
            question
            .strip()
            .lower()
        )

        general_phrases = {
            "how are you",
            "how are you doing",
            "what's up",
            "what are you doing",
            "who are you",
            "what is your name",
            "thanks",
            "thank you",
            "goodbye",
            "bye",
            "see you",
            "ok",
            "okay",
        }

        if normalized in general_phrases:
            return True

        general_starts = (
            "how are you",
            "what's up",
            "what are you doing",
            "who are you",
            "what is your name",
            "thanks",
            "thank you",
        )

        return normalized.startswith(
            general_starts
        )


__all__ = [
    "RAGPipeline"
]