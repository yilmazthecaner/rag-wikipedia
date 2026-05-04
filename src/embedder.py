"""
Embedding module using sentence-transformers (fully local, no external API).

Model: all-MiniLM-L6-v2
  - 384-dimensional embeddings
  - ~22 MB model size — fits comfortably in RAM
  - Excellent semantic similarity performance for retrieval tasks
  - Normalised embeddings → cosine similarity = dot product (fast)

The model is lazy-loaded once and cached for the process lifetime.
"""

from __future__ import annotations

from typing import Sequence

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("Embedding model ready.")
    return _model


def embed_texts(texts: Sequence[str], batch_size: int = 64,
                show_progress: bool = True) -> list[list[float]]:
    """
    Embed a list of strings. Returns a list of normalised float vectors.

    Args:
        texts:         Strings to embed.
        batch_size:    Mini-batch size for GPU/CPU throughput.
        show_progress: Show tqdm progress bar.

    Returns:
        List of embedding vectors (list[float]), one per input string.
    """
    model = _get_model()
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # L2-normalise → cosine similarity ≡ dot product
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string. Returns a single normalised float vector.
    """
    model = _get_model()
    vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vec[0].tolist()
