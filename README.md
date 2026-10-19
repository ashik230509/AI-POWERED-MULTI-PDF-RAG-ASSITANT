# AI PDF Assistant: Multi-Document RAG Chatbot

Chat with your PDFs. Upload one or more documents, ask questions in plain language, and get answers grounded in the documents, with page-level source citations when relevant.

**Live demo:** [https://aipdfa.streamlit.app/](https://aipdfa.streamlit.app/)

The hosted app may take a little time to respond after being idle. If Streamlit shows a wake-up screen, follow its prompt and wait for the app to start.

## Features

- **Multi-PDF chat:** query several documents in one conversation.
- **Hybrid retrieval:** combines FAISS semantic search and BM25 keyword search using Reciprocal Rank Fusion (RRF).
- **Cross-encoder reranking:** reranks retrieved passages for relevance before answer generation.
- **Source citations:** grounded responses can show the document name, page number, and an expandable excerpt.
- **Evidence-aware answers:** the app checks retrieved evidence and asks the LLM to avoid unsupported claims.
- **Follow-up questions:** recent conversation history is included to help interpret follow-ups.
- **Typo-tolerant keyword search:** adds likely corrections for query words that are missing from the uploaded-document vocabulary (for example, `notise` may match `notice`).
- **Flexible LLM backend:** supports local Ollama and OpenAI-compatible API endpoints.

## How it works

```text
PDF upload
   |
   v
Page-by-page text extraction and cleaning (pypdf)
   |
   v
Page-aware, overlapping character chunks
   |
   +--> Sentence embeddings --> FAISS semantic search --+
   |                                                     |
   +--> BM25 keyword search with fuzzy correction ------+--> Reciprocal Rank Fusion
                                                               |
                                                               v
                                                     Cross-encoder reranking
                                                               |
                                                               v
                                                  Evidence check and LLM response
                                                               |
                                                               v
                                                   Answer with source citations
```

Semantic search can find passages with similar meaning, while BM25 helps match exact terms such as names or clause numbers. RRF combines their results, and the cross-encoder reranks the candidates before selected evidence is sent to the LLM.

## Tech stack

| Area | Tools |
| --- | --- |
| UI | Streamlit |
| PDF parsing | pypdf |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector search | FAISS |
| Keyword search | rank-bm25 |
| Reranking | sentence-transformers CrossEncoder |
| LLM | Ollama or an OpenAI-compatible API |
| Tests | pytest |

## Project structure

```text
├── app.py               # Streamlit user interface
├── config.py            # Environment variables and defaults
├── pdf_processor.py     # PDF text extraction and cleaning
├── chunker.py           # Page-aware chunk creation
├── embeddings.py        # Embedding model loader and encoder
├── vector_store.py      # FAISS vector index
├── bm25_retriever.py    # BM25 search with fuzzy query expansion
├── hybrid_retriever.py  # Reciprocal Rank Fusion
├── reranker.py          # Cross-encoder reranking
├── llm.py               # LLM integration and evidence prompts
├── rag_pipeline.py      # RAG workflow orchestration
├── tests/               # Automated tests
└── requirements.txt     # Python dependencies
```

## Run locally

### 1. Clone and install

```bash
git clone https://github.com/ashik230509/AI-POWERED-MULTI-PDF-RAG-ASSITANT.git
cd AI-POWERED-MULTI-PDF-RAG-ASSITANT
python -m venv venv
```

Activate the virtual environment:

```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure the LLM

Set configuration values in your environment or a local `.env` file. Do not commit API keys.

**Option A: Local Ollama**

Install and start [Ollama](https://ollama.com/), then download the default model:

```bash
ollama pull llama3.1:latest
```

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:latest
```

**Option B: OpenAI-compatible API**

Use the endpoint and model supplied by your provider:

```env
LLM_PROVIDER=openai-compatible
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=your-api-key
```

### 3. Start the app

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit.

## Deployment

This project is deployed on [Streamlit Community Cloud](https://share.streamlit.io/) at **[aipdfa.streamlit.app](https://aipdfa.streamlit.app/)**.

To deploy your own copy:

1. Push the repository to GitHub.
2. Create an app in Streamlit Community Cloud using the repository, the `main` branch, and `app.py`.
3. Add your LLM configuration under **Settings → Secrets**, for example:

   ```toml
   LLM_PROVIDER = "openai-compatible"
   LLM_BASE_URL = "https://api.openai.com/v1"
   LLM_MODEL = "gpt-4o-mini"
   LLM_API_KEY = "your-provider-api-key"
   ```

Keep API keys in Streamlit Secrets; do not commit them. A hosted app cannot reach Ollama running on your own computer at `localhost`, so use a hosted LLM endpoint for cloud deployment. The embedding and reranker models are downloaded when the app starts, so initial startup may take longer.

## Example questions

- “What are the key milestones in this document?”
- “Summarize the requirements section.”
- “What does the second document say about compliance?”
- “Can you explain that in simpler terms?”

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `120` | Characters shared between adjacent chunks |
| `TOP_K` | `5` | Base number of retrieval results |
| `MAX_CONTEXT_CHUNKS` | `4` | Maximum evidence chunks sent to the LLM |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker model |

## Limitations

- Answer quality depends on the PDF text and the selected LLM. Evidence checks reduce unsupported answers but cannot guarantee zero hallucinations.
- There is no OCR support for image-only scanned pages; pages without extractable text are skipped.
- Chunking uses fixed-size character windows rather than sentence or section boundaries.
- Document indexes are built in memory for the current app session and are not persisted for later sessions.

## Roadmap

- [ ] OCR support for scanned PDFs
- [ ] Sentence- or section-aware chunking
- [ ] Retrieval evaluation set (vector vs. BM25 vs. hybrid vs. hybrid with reranking)
- [ ] Persistent vector store across sessions
- [ ] Streaming responses

## Tests

Run the test suite with:

```bash
pytest
```

## Author

**Ashik T** — B.Tech Information Technology, Nandha College of Technology
GitHub: [@ashik230509](https://github.com/ashik230509)
