"""
Central configuration for the Wikipedia RAG system.
BLG483E Project 3 — Caner Yılmaz (820230775)
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "wikipedia.db")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
ENTITIES_FILE = os.path.join(DATA_DIR, "entities.json")

# ── Chunking ──────────────────────────────────────────────────────────────────
# 500-character chunks with 50-character overlap.
# Rationale: Wikipedia articles are large (5k–50k chars). 500-char chunks
# are small enough to be semantically focused yet large enough to carry context.
# The 50-char overlap prevents boundary-cut information loss.
CHUNK_SIZE = 500        # characters
CHUNK_OVERLAP = 50      # characters
MIN_CHUNK_SIZE = 100    # discard micro-chunks

# ── Embedding ─────────────────────────────────────────────────────────────────
# all-MiniLM-L6-v2: 384-dim, ~22 MB, fast CPU inference, very strong retrieval.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Ollama (local LLM) ────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"   # override via env: OLLAMA_MODEL=mistral python ...

# ── ChromaDB ──────────────────────────────────────────────────────────────────
# Design choice: Option B — ONE collection with entity_type metadata.
# Rationale: Simpler to maintain, allows mixed queries without merging results
# from two separate collections. Filtering is done at query time via metadata.
COLLECTION_NAME = "wikipedia_rag"

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 5
RETRIEVAL_MAX_DISTANCE = 0.62
