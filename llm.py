from __future__ import annotations

import json
import re
from typing import Any

import requests

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)


class OllamaClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:

        self.base_url = (
            base_url
            or OLLAMA_BASE_URL
        ).rstrip("/")

        self.model = (
            model
            or OLLAMA_MODEL
        )

        self.chat_url = (
            f"{self.base_url}/api/chat"
        )

    # =========================================================
    # OLLAMA AVAILABILITY
    # =========================================================

    def check_available(self) -> bool:

        try:

            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )

            response.raise_for_status()

            return True

        except requests.RequestException:

            return False

    # =========================================================
    # CHECK MODEL
    # =========================================================

    def check_model_available(self) -> bool:

        try:

            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )

            response.raise_for_status()

            data = response.json()

            models = data.get(
                "models",
                [],
            )

            available_models = {
                str(
                    model.get(
                        "name",
                        "",
                    )
                )
                for model in models
            }

            if self.model in available_models:
                return True

            # Handle cases where the configured model
            # contains :latest but the API returns the
            # base model name or vice versa.
            configured_base = (
                self.model.split(":")[0]
            )

            for available_model in available_models:

                available_base = (
                    available_model.split(":")[0]
                )

                if (
                    available_base
                    == configured_base
                ):
                    return True

            return False

        except (
            requests.RequestException,
            ValueError,
        ):

            return False

    # =========================================================
    # CHAT REQUEST
    # =========================================================

    def _chat(
        self,
        messages: list[
            dict[str, str]
        ],
        temperature: float = 0.1,
    ) -> str:

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        try:

            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=180,
            )

        except requests.RequestException as exc:

            raise RuntimeError(
                f"Unable to connect to Ollama: {exc}"
            ) from exc

        if response.status_code != 200:

            try:

                error_data = (
                    response.json()
                )

                error_message = (
                    error_data.get(
                        "error",
                        response.text,
                    )
                )

            except ValueError:

                error_message = (
                    response.text
                )

            raise RuntimeError(
                "Ollama returned an error "
                f"({response.status_code}): "
                f"{error_message}"
            )

        try:

            data = response.json()

        except ValueError as exc:

            raise RuntimeError(
                "Ollama returned invalid JSON."
            ) from exc

        message = data.get(
            "message",
            {},
        )

        content = message.get(
            "content",
            "",
        )

        if not isinstance(
            content,
            str,
        ):

            content = str(
                content
            )

        return content.strip()

    # =========================================================
    # GENERATE
    # =========================================================

    def generate(
        self,
        messages: list[
            dict[str, str]
        ],
        temperature: float = 0.1,
    ) -> str:

        return self._chat(
            messages,
            temperature=temperature,
        )

    # =========================================================
    # JSON PARSER
    # =========================================================

    @staticmethod
    def _parse_json(
        text: str,
    ) -> dict[str, Any] | None:

        if not text:
            return None

        cleaned = text.strip()

        # -----------------------------------------------------
        # Direct JSON
        # -----------------------------------------------------

        try:

            parsed = json.loads(
                cleaned
            )

            if isinstance(
                parsed,
                dict,
            ):
                return parsed

        except (
            json.JSONDecodeError,
            TypeError,
        ):
            pass

        # -----------------------------------------------------
        # JSON inside markdown code block
        # -----------------------------------------------------

        code_block_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            cleaned,
            flags=re.DOTALL,
        )

        if code_block_match:

            try:

                parsed = json.loads(
                    code_block_match.group(1)
                )

                if isinstance(
                    parsed,
                    dict,
                ):
                    return parsed

            except (
                json.JSONDecodeError,
                TypeError,
            ):
                pass

        # -----------------------------------------------------
        # First JSON object
        # -----------------------------------------------------

        object_match = re.search(
            r"\{.*\}",
            cleaned,
            flags=re.DOTALL,
        )

        if object_match:

            try:

                parsed = json.loads(
                    object_match.group(0)
                )

                if isinstance(
                    parsed,
                    dict,
                ):
                    return parsed

            except (
                json.JSONDecodeError,
                TypeError,
            ):
                pass

        return None

    # =========================================================
    # DYNAMIC QUESTION DECOMPOSITION
    # =========================================================

    def decompose_question(
        self,
        question: str,
    ) -> list[str]:

        """
        Decompose a multi-part user question into
        independent retrieval questions.

        This method contains no document-specific
        knowledge.
        """

        if not question or not question.strip():

            return [question]

        prompt = f"""
You are a question decomposition component inside
a document-based RAG system.

USER QUESTION:
{question}

TASK:
Determine whether the user question contains multiple
independent information requests.

If it contains multiple requests:

- Split the question into separate, self-contained
  questions.
- Preserve the exact meaning of the original question.
- Keep document names, task numbers, names, dates,
  entities, and references from the original question.
- Each question must be independently understandable.
- Do not answer the questions.
- Do not add information that is not present in the
  original question.

If the question contains only one information request:

- Return the original question as one item.

IMPORTANT:
1. Do not use outside knowledge.
2. Do not answer the question.
3. Do not invent missing details.
4. Do not remove important references from the question.
5. Return ONLY valid JSON.

Required format:

{{
    "questions": [
        "question 1",
        "question 2"
    ]
}}
"""

        messages = [
            {
                "role": "system",
                "content": (
                    "You decompose user questions for "
                    "a document RAG retrieval system. "
                    "Return only the requested JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        try:

            response = self._chat(
                messages,
                temperature=0.0,
            )

        except Exception as exc:

            print(
                "Question decomposition error:",
                exc,
            )

            return [question]

        parsed = self._parse_json(
            response
        )

        if not isinstance(
            parsed,
            dict,
        ):

            return [question]

        questions = parsed.get(
            "questions",
            [],
        )

        if not isinstance(
            questions,
            list,
        ):

            return [question]

        cleaned_questions: list[
            str
        ] = []

        for item in questions:

            if not isinstance(
                item,
                str,
            ):
                continue

            item = item.strip()

            if not item:
                continue

            duplicate = any(
                item.lower()
                == existing.lower()
                for existing in cleaned_questions
            )

            if duplicate:
                continue

            cleaned_questions.append(
                item
            )

        if not cleaned_questions:

            return [question]

        return cleaned_questions

    # =========================================================
    # FOLLOW-UP QUESTION RESOLUTION
    # =========================================================

    def resolve_follow_up(
        self,
        chat_history: list[
            dict[str, str]
        ],
        question: str,
    ) -> str:

        history_text_parts: list[str] = []

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

            history_text_parts.append(
                f"{role.upper()}: {content}"
            )

        history_text = (
            "\n".join(
                history_text_parts
            )
        )

        prompt = f"""
Resolve the user's follow-up question into a
self-contained retrieval question.

Conversation history:
{history_text}

Current question:
{question}

Rules:
1. Use only the conversation history.
2. Preserve the user's intended meaning.
3. Resolve references such as "it", "that", "this",
   "its", or "the previous task" when the history
   clearly identifies them.
4. Do not answer the question.
5. Do not add outside knowledge.
6. Return only the resolved question.
"""

        try:

            response = self._chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You resolve follow-up "
                            "questions for document retrieval."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.0,
            )

        except Exception as exc:

            print(
                "Follow-up resolution error:",
                exc,
            )

            return question

        resolved = response.strip()

        return (
            resolved
            if resolved
            else question
        )

    # =========================================================
    # EVIDENCE VERIFICATION
    # =========================================================

    def verify_evidence(
        self,
        question: str,
        candidates: list[
            dict[str, Any]
        ],
    ) -> dict[str, Any]:

        if not candidates:

            return {
                "supported": False,
                "chunk_ids": [],
                "reason": (
                    "No retrieved evidence was available."
                ),
            }

        evidence_blocks: list[str] = []

        valid_chunk_ids: set[str] = set()

        for index, candidate in enumerate(
            candidates
        ):

            chunk_id = str(
                candidate.get(
                    "chunk_id",
                    f"chunk_{index}",
                )
            )

            text = (
                candidate.get(
                    "text",
                    "",
                )
                or candidate.get(
                    "content",
                    "",
                )
            )

            valid_chunk_ids.add(
                chunk_id
            )

            evidence_blocks.append(
                f"""
EVIDENCE {index + 1}
CHUNK_ID: {chunk_id}

TEXT:
{text}
"""
            )

        evidence_text = (
            "\n".join(
                evidence_blocks
            )
        )

        verification_prompt = f"""
You are an evidence verification component inside
a document RAG system.

USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{evidence_text}

Your task is to identify EVERY retrieved chunk that
contains evidence needed to answer the user's question.

IMPORTANT RULES:

1. Use ONLY the retrieved evidence above.
2. Do not use outside knowledge.
3. A question may contain multiple requested facts
   or independent parts.
4. For EVERY part of the question, find the chunk
   that supports that part.
5. If different parts are supported by different
   chunks, return ALL of those chunk IDs.
6. A chunk does not need to contain the entire answer.
   If it supports one required part, include it.
7. Return the EXACT CHUNK_ID shown after CHUNK_ID.
8. Never return an evidence number such as
   "EVIDENCE 1".
9. Never invent a chunk ID.
10. Do not omit a chunk merely because another chunk
    also supports another part of the answer.
11. If the evidence is insufficient for one or more
    required parts, supported must be false.
12. Return ONLY valid JSON.

Before producing JSON, internally check:

- What separate facts does the question request?
- Which retrieved chunk supports each fact?
- Have I included every chunk needed for every part?
- Are all returned IDs exact IDs from the evidence?

Required JSON:

{{
    "supported": true,
    "chunk_ids": [
        "exact_chunk_id_1",
        "exact_chunk_id_2"
    ],
    "reason": "short explanation"
}}
"""

        try:

            response = self._chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a strict evidence "
                            "verification component. "
                            "Return only valid JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": verification_prompt,
                    },
                ],
                temperature=0.0,
            )

        except Exception as exc:

            return {
                "supported": False,
                "chunk_ids": [],
                "reason": (
                    f"Verification failed: {exc}"
                ),
            }

        parsed = self._parse_json(
            response
        )

        if not isinstance(
            parsed,
            dict,
        ):

            return {
                "supported": False,
                "chunk_ids": [],
                "reason": (
                    "Verifier returned invalid JSON."
                ),
            }

        supported = bool(
            parsed.get(
                "supported",
                False,
            )
        )

        raw_chunk_ids = parsed.get(
            "chunk_ids",
            [],
        )

        if not isinstance(
            raw_chunk_ids,
            list,
        ):

            raw_chunk_ids = []

        verified_chunk_ids: list[
            str
        ] = []

        for chunk_id in raw_chunk_ids:

            if not isinstance(
                chunk_id,
                str,
            ):
                continue

            chunk_id = chunk_id.strip()

            if not chunk_id:
                continue

            if chunk_id not in valid_chunk_ids:
                continue

            if chunk_id in verified_chunk_ids:
                continue

            verified_chunk_ids.append(
                chunk_id
            )

        return {
            "supported": (
                supported
                and bool(
                    verified_chunk_ids
                )
            ),
            "chunk_ids": (
                verified_chunk_ids
            ),
            "reason": str(
                parsed.get(
                    "reason",
                    "",
                )
            ),
        }

    # =========================================================
    # GROUNDED RAG SYSTEM PROMPT
    # =========================================================

    def build_system_prompt(
        self,
    ) -> str:

        return """
You are a document-grounded RAG assistant.

Your job is to answer questions using ONLY the
provided document context.

STRICT RULES:

1. Use only the supplied document context.
2. Do not use outside knowledge.
3. Do not invent facts.
4. Do not guess missing information.
5. Do not fabricate document names.
6. Do not fabricate page numbers.
7. Do not fabricate citations.
8. If the answer is not supported by the supplied
   context, clearly say that the information is
   not available in the uploaded documents.
9. For multi-part questions, answer every part
   that is supported by the supplied context.
10. If different parts are supported by different
    documents, use the appropriate document evidence
    for each part.
11. Keep answers concise and directly relevant.
12. When useful, mention the document name and page
    number exactly as provided in the context.
"""

    # =========================================================
    # BUILD MESSAGES
    # =========================================================

    def build_messages(
        self,
        question: str,
        context: str,
        chat_history: list[
            dict[str, str]
        ] | None = None,
    ) -> list[
        dict[str, str]
    ]:

        chat_history = (
            chat_history
            or []
        )

        messages: list[
            dict[str, str]
        ] = []

        messages.append(
            {
                "role": "system",
                "content": (
                    self.build_system_prompt()
                    + "\n\n"
                    "Use conversation history only to "
                    "understand follow-up references. "
                    "The actual answer must remain grounded "
                    "in the supplied document context."
                ),
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
        # CURRENT QUESTION
        # -----------------------------------------------------

        messages.append(
            {
                "role": "user",
                "content": (
                    f"Question:\n"
                    f"{question}\n\n"
                    f"Document Context:\n"
                    f"{context}"
                ),
            }
        )

        return messages


__all__ = [
    "OllamaClient"
]