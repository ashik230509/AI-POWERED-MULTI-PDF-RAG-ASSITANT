from __future__ import annotations

from typing import Any

import streamlit as st

from rag_pipeline import RAGPipeline


st.set_page_config(page_title="AI PDF Assistant", page_icon="📄", layout="wide")


def _is_greeting(message: str) -> bool:
    normalized = message.strip().lower()
    return normalized in {"hi", "hello", "hey"} or normalized.startswith(("hi ", "hello ", "hey "))


if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_status" not in st.session_state:
    st.session_state.document_status = "No documents processed"


st.sidebar.title("AI PDF Assistant")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
process_button = st.sidebar.button("Process Documents")
clear_button = st.sidebar.button("Clear Documents")

if clear_button:
    st.session_state.pipeline = RAGPipeline()
    st.session_state.messages = []
    st.session_state.document_status = "No documents processed"
    st.rerun()

if process_button and uploaded_files:
    try:
        chunk_count = st.session_state.pipeline.process_documents(uploaded_files)
        st.session_state.document_status = f"{len(uploaded_files)} document(s) processed"
        st.session_state.messages.append({"role": "assistant", "content": "Your documents are ready. Ask me anything about them."})
        st.success(f"{len(uploaded_files)} document(s) processed successfully. {chunk_count} chunks indexed.")
    except Exception as exc:  # pragma: no cover - UI guard
        st.error(str(exc))

st.sidebar.markdown(f"**Document status:** {st.session_state.document_status}")

st.title("AI PDF Assistant")
st.caption("Upload one or more PDFs and ask questions about their contents.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Ask a question about the documents")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    if not getattr(st.session_state.pipeline, "chunks", []) and not _is_greeting(user_input):
        reply = "Please upload and process one or more PDFs before asking a question."
        with st.chat_message("assistant"):
            st.write(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    elif _is_greeting(user_input):
        reply = "Hi! Upload one or more PDFs, process them, and then ask me anything about the content."
        with st.chat_message("assistant"):
            st.write(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    else:
        history = [
            {"role": message["role"], "content": message["content"]}
            for message in st.session_state.messages[:-1]
        ]
        result = st.session_state.pipeline.answer_question(user_input, chat_history=history)
        answer = result.get("answer") or "I couldn't find sufficient information to answer this question from the uploaded documents."
        with st.chat_message("assistant"):
            st.write(answer)
            citations = result.get("citations", [])
            if citations:
                st.markdown("**Sources:**")
                for citation in citations:
                    doc_name = citation.get("document_name", "Unknown document")
                    page_number = citation.get("page_number")
                    st.markdown(f"- {doc_name} — Page {page_number}")
                    with st.expander(f"View excerpt from {doc_name} (Page {page_number})"):
                        st.write(citation.get("text", ""))
        st.session_state.messages.append({"role": "assistant", "content": answer})
