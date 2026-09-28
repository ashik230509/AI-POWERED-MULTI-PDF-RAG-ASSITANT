from __future__ import annotations

from bm25_retriever import BM25Retriever
from chunker import chunk_pages
from hybrid_retriever import HybridRetriever
from pdf_processor import extract_pdf_pages
from rag_pipeline import RAGPipeline


def _escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _make_pdf_bytes(pages: list[str]) -> bytes:
    objects: dict[int, str] = {}
    page_refs: list[int] = []

    objects[1] = "<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = "<< /Type /Pages /Kids ["

    for index, page_text in enumerate(pages, start=1):
        page_object = 3 + 3 * (index - 1)
        content_object = page_object + 1
        font_object = page_object + 2
        page_refs.append(page_object)

        escaped_text = _escape_pdf_text(page_text)
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET"
        stream_bytes = stream.encode("utf-8")

        objects[page_object] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_object} 0 R "
            f"/Resources << /Font << /F1 {font_object} 0 R >> >> >>"
        )
        objects[content_object] = f"<< /Length {len(stream_bytes)} >>\nstream\n{stream}\nendstream"
        objects[font_object] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    objects[2] = "<< /Type /Pages /Kids [" + " ".join(f"{ref} 0 R" for ref in page_refs) + "] /Count " + str(len(page_refs)) + " >>"

    content = b"%PDF-1.4\n"
    offsets: dict[int, int] = {}
    for obj_num in sorted(objects):
        offsets[obj_num] = len(content)
        content += f"{obj_num} 0 obj\n{objects[obj_num]}\nendobj\n".encode("latin-1")

    xref_offset = len(content)
    content += f"xref\n0 {max(objects) + 1}\n".encode("latin-1")
    content += b"0000000000 65535 f \n"
    for obj_num in range(1, max(objects) + 1):
        offset = offsets.get(obj_num, 0)
        content += f"{offset:010d} 00000 n \n".encode("latin-1")

    content += f"trailer\n<< /Size {max(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin-1")
    return content


def test_pdf_extraction_returns_page_texts():
    pdf_bytes = _make_pdf_bytes(["Project deadline is Friday.", "Team lead is Dana."])

    pages = extract_pdf_pages(pdf_bytes)

    assert len(pages) == 2
    assert "Project deadline" in pages[0]["text"]
    assert "Team lead" in pages[1]["text"]
    assert pages[0]["page_number"] == 1
    assert pages[1]["page_number"] == 2


def test_chunking_preserves_metadata_and_boundaries():
    pages = [
        {"page_number": 1, "text": "Alpha beta gamma delta epsilon zeta eta theta iota kappa", "document_name": "doc1.pdf"},
        {"page_number": 2, "text": "Lambda mu nu xi omicron pi rho sigma tau upsilon", "document_name": "doc1.pdf"},
    ]

    chunks = chunk_pages(pages, chunk_size=30, overlap=10)

    assert chunks
    assert all("chunk_id" in chunk for chunk in chunks)
    assert all(chunk["document_name"] == "doc1.pdf" for chunk in chunks)
    assert all(chunk["page_number"] >= 1 for chunk in chunks)
    assert all(len(chunk["text"].split()) > 0 for chunk in chunks)


def test_bm25_retrieval_ranks_relevant_terms():
    docs = [
        {"chunk_id": "c1", "document_name": "book.pdf", "page_number": 1, "text": "The project deadline is Friday."},
        {"chunk_id": "c2", "document_name": "book.pdf", "page_number": 2, "text": "The team is responsible for release planning."},
    ]
    retriever = BM25Retriever(docs)

    results = retriever.search("deadline friday project", top_k=2)

    assert results
    assert results[0]["chunk_id"] == "c1"


def test_hybrid_retrieval_combines_sources():
    vector_hits = [
        {"chunk_id": "v2", "document_name": "doc.pdf", "page_number": 2, "text": "team lead", "score": 0.95},
    ]
    bm25_hits = [
        {"chunk_id": "b1", "document_name": "doc.pdf", "page_number": 1, "text": "project deadline", "score": 0.80},
    ]
    retriever = HybridRetriever(vector_hits, bm25_hits)

    results = retriever.search("project deadline")

    assert any(item["chunk_id"] == "v2" for item in results)
    assert any(item["chunk_id"] == "b1" for item in results)
    assert results[0]["score"] >= 0


def test_citation_metadata_conforms_to_document_sources():
    citations = [
        {"chunk_id": "c1", "document_name": "memo.pdf", "page_number": 4, "text": "The review is on Friday."},
        {"chunk_id": "c2", "document_name": "manual.pdf", "page_number": 12, "text": "The policy requires notice."},
    ]

    result = RAGPipeline._citation_metadata(citations)

    assert result[0]["document_name"] == "memo.pdf"
    assert result[0]["page_number"] == 4
    assert result[1]["document_name"] == "manual.pdf"
    assert result[1]["page_number"] == 12


def test_rag_pipeline_rejects_insufficient_evidence():
    pipeline = RAGPipeline()
    pipeline.chunks = [
        {"chunk_id": "c1", "document_name": "doc.pdf", "page_number": 1, "text": "This document is about budgets and planning."}
    ]
    pipeline.vector_store = type("VS", (), {"search": lambda self, query, top_k=5: []})()
    pipeline.bm25_retriever = type("BM", (), {"search": lambda self, query, top_k=5: []})()
    pipeline.reranker = type("R", (), {"rerank": lambda self, items, query: items})()

    answer = pipeline.answer_question("What is the secret project code?", chat_history=[])

    assert "I couldn't find sufficient information" in answer["answer"]
    assert not answer["citations"]


def test_rag_pipeline_accepts_ranked_candidates_even_with_negative_reranker_scores():
    class DummyReranker:
        def rerank(self, items, query):
            return [{**item, "score": -0.25} for item in items]

    class DummyLLM:
        def generate(self, messages):
            return "The project deadline is Friday."

    pipeline = RAGPipeline()
    pipeline.chunks = [
        {"chunk_id": "c1", "document_name": "doc.pdf", "page_number": 1, "text": "The project deadline is Friday."}
    ]
    pipeline.vector_store = type("VS", (), {"search": lambda self, query, top_k=5: [{
        "chunk_id": "c1",
        "document_name": "doc.pdf",
        "page_number": 1,
        "text": "The project deadline is Friday.",
        "score": 0.8,
    }]})()
    pipeline.bm25_retriever = type("BM", (), {"search": lambda self, query, top_k=5: [{
        "chunk_id": "c1",
        "document_name": "doc.pdf",
        "page_number": 1,
        "text": "The project deadline is Friday.",
        "score": 0.7,
    }]})()
    pipeline.reranker = DummyReranker()
    pipeline.llm_client = DummyLLM()

    answer = pipeline.answer_question("What is the project deadline?", chat_history=[])

    assert "Friday" in answer["answer"]
    assert answer["citations"]


def test_citation_metadata_deduplicates_by_document_and_page():
    chunks = [
        {"chunk_id": "c1", "document_name": "Machine Learning.pdf", "page_number": 5, "text": "Alpha"},
        {"chunk_id": "c2", "document_name": "Machine Learning.pdf", "page_number": 5, "text": "Alpha repeated"},
        {"chunk_id": "c3", "document_name": "Machine Learning.pdf", "page_number": 7, "text": "Beta"},
    ]

    citations = RAGPipeline._citation_metadata(chunks)

    assert len(citations) == 2
    assert citations[0]["document_name"] == "Machine Learning.pdf"
    assert citations[0]["page_number"] == 5
    assert citations[1]["page_number"] == 7
