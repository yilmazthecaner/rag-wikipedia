"""
Vector store module — wraps ChromaDB.

Design choice: Option B — ONE collection with entity_type metadata.

Rationale:
  - A single collection simplifies the codebase: one index to build, one to query.
  - entity_type filtering at query time (ChromaDB `where` clause) gives the same
    isolation as two separate collections, without duplicating index logic.
  - Mixed queries ("Compare Messi and the Eiffel Tower") naturally retrieve from
    both entity types by simply omitting the `where` filter.
  - If the corpus grew significantly, we could migrate to two collections with
    minimal code changes, since the metadata field already exists.
"""

from __future__ import annotations

import os

import chromadb

from src.config import CHROMA_DIR, COLLECTION_NAME


def _get_client() -> chromadb.PersistentClient:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_collection() -> chromadb.Collection:
    """Return (or create) the persistent ChromaDB collection."""
    client = _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine distance for normalised embeddings
    )


def add_chunks(chunks: list[dict]) -> None:
    """
    Upsert chunk dicts into the vector store.

    Each chunk dict must have:
        id        : str  — unique identifier
        text      : str  — document content
        embedding : list[float]
        metadata  : dict  — {"title", "entity_type", "chunk_index", "url", ...}
    """
    collection = get_collection()

    # Batch upserts to avoid memory spikes on large corpora
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.upsert(
            ids=[c["id"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )


def query_store(
    query_embedding: list[float],
    entity_type: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve the top-k most similar chunks.

    Args:
        query_embedding: Normalised query vector.
        entity_type:     "person" | "place" | None (no filter → both types).
        top_k:           Number of results to return.

    Returns:
        List of result dicts:
            {"text": str, "metadata": dict, "distance": float}
    """
    collection = get_collection()

    # If collection is empty return nothing
    if collection.count() == 0:
        return []

    where = None
    if entity_type in ("person", "place"):
        where = {"entity_type": {"$eq": entity_type}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({"text": doc, "metadata": meta, "distance": dist})
    return output


def get_chunk_count(entity_type: str | None = None) -> int:
    """Return number of indexed chunks, optionally filtered by entity_type."""
    collection = get_collection()
    if entity_type is None:
        return collection.count()
    results = collection.get(where={"entity_type": {"$eq": entity_type}})
    return len(results["ids"])


def reset_collection() -> None:
    """Delete and recreate the collection (full index reset)."""
    client = _get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print("Vector store reset.")
