from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

load_dotenv()


def _setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is not None:
        return value

    try:
        return str(st.secrets[name])
    except (KeyError, StreamlitSecretNotFoundError):
        return default

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
LLM_PROVIDER = _setting("LLM_PROVIDER", "ollama").strip().lower()
LLM_BASE_URL = _setting("LLM_BASE_URL").strip().rstrip("/")
LLM_MODEL = _setting("LLM_MODEL").strip()
LLM_API_KEY = _setting("LLM_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "4"))

DATA_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)
