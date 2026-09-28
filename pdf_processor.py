from __future__ import annotations

import re
from io import BytesIO
from typing import Any

from pypdf import PdfReader


def clean_extracted_text(raw_text: str) -> str:
    if not raw_text:
        return ""

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n +", "\n", text)
    text = re.sub(r"(?<=[A-Za-z])\n(?=[A-Za-z])", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(file_bytes: bytes, document_name: str = "uploaded.pdf") -> list[dict[str, Any]]:
    if not file_bytes:
        raise ValueError("The uploaded PDF is empty.")

    try:
        reader = PdfReader(BytesIO(file_bytes))
    except Exception as exc:  # pragma: no cover - defensive runtime validation
        raise ValueError(f"Unable to read the PDF file: {exc}") from exc

    pages = reader.pages
    if not pages:
        raise ValueError("This PDF has no pages or no extractable content.")

    extracted_pages: list[dict[str, Any]] = []
    for page_number, page in enumerate(pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - defensive runtime validation
            raise ValueError(f"Failed to extract text from page {page_number}: {exc}") from exc

        cleaned_text = clean_extracted_text(raw_text)
        if cleaned_text:
            extracted_pages.append(
                {
                    "document_name": document_name,
                    "page_number": page_number,
                    "text": cleaned_text,
                }
            )

    if not extracted_pages:
        raise ValueError("No extractable text was found in the uploaded PDF.")

    return extracted_pages


def extract_documents(uploaded_files: list[Any]) -> list[dict[str, Any]]:
    if not uploaded_files:
        raise ValueError("No documents were uploaded.")

    all_pages: list[dict[str, Any]] = []
    for uploaded_file in uploaded_files:
        if uploaded_file is None:
            continue

        filename = getattr(uploaded_file, "name", "uploaded.pdf")
        if not filename.lower().endswith(".pdf"):
            continue

        try:
            pdf_bytes = uploaded_file.read()
        except Exception as exc:  # pragma: no cover - defensive runtime validation
            raise ValueError(f"Unable to read '{filename}': {exc}") from exc

        if not pdf_bytes:
            raise ValueError(f"The file '{filename}' is empty.")

        pages = extract_pdf_pages(pdf_bytes, document_name=filename)
        all_pages.extend(pages)

    if not all_pages:
        raise ValueError("No valid PDF files were found to process.")

    return all_pages
