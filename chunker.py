from __future__ import annotations

import hashlib
from typing import Any

from config import CHUNK_OVERLAP, CHUNK_SIZE


def _truncate_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 2)

    windows: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        segment = text[start:end].strip()
        if segment:
            windows.append(segment)
        if end >= len(text):
            break
        start += chunk_size - overlap
    return windows


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    if not pages:
        return []

    chunks: list[dict[str, Any]] = []
    for page in pages:
        page_text = (page.get("text") or "").strip()
        if not page_text:
            continue

        segments = _truncate_with_overlap(page_text, chunk_size=chunk_size, overlap=overlap)
        if not segments:
            continue

        document_name = page.get("document_name", "unknown.pdf")
        page_number = int(page.get("page_number", 1))
        for sequence_number, segment in enumerate(segments, start=1):
            chunk_id = hashlib.sha1(
                f"{document_name}:{page_number}:{sequence_number}:{segment[:80]}".encode("utf-8")
            ).hexdigest()[:12]
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_name": document_name,
                    "page_number": page_number,
                    "text": segment,
                }
            )

    return chunks
