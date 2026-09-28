# AI-Powered Multi-Document RAG Chatbot V2

This project is a real PDF chatbot built with a retrieval-augmented generation (RAG) pipeline. It accepts multiple PDFs, extracts page-aware text, chunks the content, builds semantic and keyword indexes, reranks the strongest evidence, and then asks Ollama to answer using only the retrieved context.

## Architecture overview

The app follows a practical production-style pipeline:

1. PDF upload and validation
2. Page-by-page extraction with pypdf
3. Page-aware cleaning and chunking
4. Semantic indexing with FAISS
5. Keyword retrieval with BM25
6. Hybrid rank fusion for merged candidate evidence
7. Cross-encoder reranking
8. Evidence validation before model generation
9. Follow-up-aware prompting to Ollama
10. Source citations from retrieved chunks

In interview-friendly terms: the chatbot does not memorize the documents. Instead, it finds the most relevant passages from the uploaded PDFs, checks that the evidence is strong enough, and then asks the LLM to answer using those passages as grounded reference material.

## How RAG works here

The system converts each PDF page into clean text, slices that text into meaningful chunks, and stores both semantic and keyword representations. When a user asks a question, the app retrieves candidate chunks from both retrieval paths, combines them, reranks the best results, and only sends the most relevant evidence to the LLM. The model is instructed to answer only from that evidence and to say so when the context is insufficient.

## Project structure

- app.py — Streamlit user interface
- config.py — environment variables and defaults
- pdf_processor.py — PDF reading and page extraction
- chunker.py — page-aware chunk creation
- embeddings.py — embedding model loader and encoder
- vector_store.py — FAISS vector index
- bm25_retriever.py — BM25 search index
- hybrid_retriever.py — reciprocal rank fusion
- reranker.py — cross-encoder reranking
- llm.py — Ollama API integration and prompt setup
- rag_pipeline.py — orchestrates the full workflow
- tests/ — automated checks for core behavior
- data/ — uploaded or working document storage
- vectorstore/ — local vector index folder

## Installation

1. Create a virtual environment.
2. Activate it.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Ollama setup

1. Install Ollama from the official site.
2. Start the service.
3. Pull the model used by the app:

```bash
ollama pull llama3.1:latest
```

The default configuration in `.env` is:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:latest
```

## Running the app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Deploy on Streamlit Community Cloud

1. Create an app at [share.streamlit.io](https://share.streamlit.io/) and select this GitHub repository, branch `main`, and `app.py`.
2. In the app's **Settings > Secrets**, add a hosted OpenAI-compatible chat endpoint. For example:

```toml
LLM_PROVIDER = "openai-compatible"
LLM_BASE_URL = "https://api.openai.com/v1"
LLM_MODEL = "gpt-4o-mini"
LLM_API_KEY = "your-provider-api-key"
```

Use the base URL, model name, and API key supplied by your chosen provider. Keep the API key in Streamlit Secrets; do not commit it to GitHub. The default `ollama` provider remains available for local development, but `localhost` Ollama is not reachable from a hosted Streamlit app.

Community Cloud installs Python packages from `requirements.txt`. The embedding and reranker models are downloaded from Hugging Face the first time the app starts, so the initial startup can take longer.

## Example usage

1. Upload one or more PDFs in the sidebar.
2. Click Process Documents.
3. Ask questions like:
   - What are the key milestones?
   - Summarize the requirements section.
   - What does the second document say about compliance?

The app keeps the conversation natural and shows source citations when grounded answers are available.

## Limitations

- The quality of the answers depends on the uploaded PDFs and the underlying Ollama model.
- The app does not claim hallucination-free performance; it only prevents unsupported answers when the evidence is weak.
- Large or highly scanned PDFs may need pre-processing or clearer source text for the best results.
- If Ollama is unavailable or the model is not installed, the app will show a clear error.

## Manual evaluation checklist

When evaluating the app manually, check the following:

- PDFs upload correctly and process without crashing.
- Different documents are indexed together and can be queried in a single conversation.
- Follow-up questions use conversation history sensibly.
- Unsupported questions return a clear refusal based on absent evidence.
- Grounded answers include correct source names and page numbers.
- The UI stays simple and does not expose technical retrieval controls.

## Notes

This implementation is intentionally backend-controlled and does not expose low-level retrieval settings in the UI. The user experience is a simple chat interface that feels like ChatGPT for the user’s uploaded documents.
