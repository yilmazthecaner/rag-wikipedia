"""
Pipeline module — orchestrates all stages end-to-end.

Stages:
  1. build_index():  read SQLite docs → chunk → embed → upsert into ChromaDB
  2. ask():          classify query → embed → retrieve → generate answer
"""

from __future__ import annotations

from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.chunker import chunk_document
from src.embedder import embed_texts
from src.ingest import get_all_documents
from src.retriever import retrieve
from src.generator import generate_answer, stream_answer
from src.vector_store import add_chunks, get_chunk_count


def build_index(verbose: bool = True) -> dict:
    """
    Process all documents in SQLite, chunk them, embed, and store in ChromaDB.

    Returns:
        {"documents": int, "chunks": int}
    """
    docs = get_all_documents()
    if not docs:
        raise RuntimeError("No documents found. Run ingestion first (python main.py ingest).")

    if verbose:
        print(f"Building index for {len(docs)} documents …")

    # ── Chunk ──────────────────────────────────────────────────────────────────
    all_chunk_dicts: list[dict] = []
    for doc in docs:
        chunks = chunk_document(doc, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        all_chunk_dicts.extend(chunks)

    if verbose:
        print(f"  Generated {len(all_chunk_dicts)} chunks.")

    # ── Embed ──────────────────────────────────────────────────────────────────
    texts = [c["text"] for c in all_chunk_dicts]
    embeddings = embed_texts(texts, show_progress=verbose)

    for chunk, emb in zip(all_chunk_dicts, embeddings):
        chunk["embedding"] = emb

    # ── Store ──────────────────────────────────────────────────────────────────
    add_chunks(all_chunk_dicts)

    if verbose:
        total = get_chunk_count()
        print(f"  Index ready — {total} total chunks in vector store.")

    return {"documents": len(docs), "chunks": len(all_chunk_dicts)}


def ask(
    query: str,
    stream: bool = False,
    top_k: int = 5,
) -> tuple:
    """
    Run the full RAG pipeline for a user query.

    Args:
        query:  User question string.
        stream: If True, returns a generator instead of a string.
        top_k:  Number of chunks to retrieve.

    Returns (non-streaming):
        (answer: str, chunks: list[dict], entity_type: str)

    Returns (streaming):
        (token_generator, chunks: list[dict], entity_type: str)
    """
    chunks, entity_type = retrieve(query, top_k=top_k)

    if stream:
        return stream_answer(query, chunks), chunks, entity_type
    else:
        answer = generate_answer(query, chunks)
        return answer, chunks, entity_type
