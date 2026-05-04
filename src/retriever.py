"""
Retriever module: classifies queries and retrieves relevant chunks.

Query classification is rule-based (keyword matching) — intentionally simple
as required by the spec. It checks for:
  1. Presence of known entity names (most reliable signal)
  2. Contextual keywords (who/where/born/located …)
"""

from __future__ import annotations

import re

from src.config import TOP_K
from src.embedder import embed_query
from src.vector_store import query_store

# ── Keyword sets ───────────────────────────────────────────────────────────────

# Required entities (lower-cased for matching)
_PERSON_NAMES = {
    "einstein", "curie", "vinci", "da vinci", "shakespeare", "lovelace",
    "tesla", "messi", "ronaldo", "swift", "taylor swift", "kahlo", "frida",
    "newton", "darwin", "mandela", "cleopatra", "galileo", "van gogh",
    "mozart", "gandhi", "hawking", "camus",
}

_PLACE_NAMES = {
    "eiffel", "great wall", "taj mahal", "grand canyon", "machu picchu",
    "colosseum", "coliseum", "hagia sophia", "statue of liberty", "pyramids",
    "giza", "mount everest", "everest", "stonehenge", "angkor", "petra",
    "acropolis", "niagara", "amazon", "sahara", "great barrier reef",
    "yellowstone", "venice",
}

# Generic context keywords
_PERSON_WORDS = {
    "who", "person", "people", "born", "died", "biography", "life",
    "inventor", "scientist", "artist", "athlete", "musician", "writer",
    "author", "physicist", "mathematician", "president", "king", "queen",
    "philosopher", "painter", "composer", "footballer",
}

_PLACE_WORDS = {
    "where", "place", "location", "country", "city", "monument", "landmark",
    "tower", "wall", "temple", "canyon", "mountain", "wonder", "pyramid",
    "statue", "island", "river", "ocean", "sea", "lake", "park", "forest",
    "desert", "reef", "falls", "waterfall", "ruins",
}


def classify_query(query: str) -> str:
    """
    Classify the query as 'person', 'place', or 'both'.

    Scoring:
      - Named entity match in _PERSON_NAMES → +3
      - Named entity match in _PLACE_NAMES  → +3
      - Keyword match in _PERSON_WORDS      → +1 each
      - Keyword match in _PLACE_WORDS       → +1 each

    Returns 'both' when scores are tied or both > 0.
    """
    q = query.lower()
    words = set(re.findall(r"\b\w+\b", q))

    person_score = 0
    place_score = 0

    # Named entity matches (multi-word checked against full string)
    for name in _PERSON_NAMES:
        if name in q:
            person_score += 3
    for name in _PLACE_NAMES:
        if name in q:
            place_score += 3

    # Generic keyword matches
    person_score += len(words & _PERSON_WORDS)
    place_score += len(words & _PLACE_WORDS)

    if person_score > 0 and place_score > 0:
        return "both"
    if person_score > place_score:
        return "person"
    if place_score > person_score:
        return "place"
    return "both"   # default: search everything


def retrieve(query: str, top_k: int = TOP_K) -> tuple[list[dict], str]:
    """
    Full retrieval pipeline for a user query.

    Steps:
      1. Classify query.
      2. Embed query.
      3. Query vector store with optional entity_type filter.

    Args:
        query:  User query string.
        top_k:  Maximum number of chunks to return.

    Returns:
        (chunks, entity_type)
        chunks: list of {"text": str, "metadata": dict, "distance": float}
        entity_type: "person" | "place" | "both"
    """
    entity_type = classify_query(query)
    query_emb = embed_query(query)

    # For "both", pass None → no metadata filter
    filter_type = entity_type if entity_type != "both" else None
    chunks = query_store(query_emb, entity_type=filter_type, top_k=top_k)

    return chunks, entity_type
