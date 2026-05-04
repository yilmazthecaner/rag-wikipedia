"""
Text chunking module.

Strategy: Fixed-size character chunks with overlap, broken at sentence boundaries.

Design rationale:
- Wikipedia articles can be 5k–80k characters. Processing the whole article as
  a single embedding would lose fine-grained retrieval precision.
- Fixed-size chunks are predictable and work well with vector stores.
- Overlap (50 chars) prevents information loss at chunk boundaries.
- We prefer breaking at sentence ends (". ") or paragraph ends ("\n\n") rather
  than mid-word to keep chunks semantically coherent.
"""

import re


def _clean_wikipedia_text(text: str) -> str:
    """Remove Wikipedia boilerplate sections that add noise."""
    # Remove references / further reading sections at the end
    for section in ("== References ==", "== Further reading ==",
                    "== External links ==", "== See also =="):
        idx = text.find(section)
        if idx != -1:
            text = text[:idx]

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50,
               min_chunk: int = 100) -> list[str]:
    """
    Split *text* into overlapping character-level chunks.

    Args:
        text:       Raw document text.
        chunk_size: Target maximum chunk size in characters.
        overlap:    Number of characters to repeat between consecutive chunks.
        min_chunk:  Chunks shorter than this are discarded.

    Returns:
        List of non-empty string chunks.
    """
    text = _clean_wikipedia_text(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)

        # Try to break at a natural boundary within the last 20% of the window
        if end < n:
            search_from = start + int(chunk_size * 0.8)
            # Prefer double-newline (paragraph), then ". ", then " "
            for boundary_str in ("\n\n", ". ", ".\n", " "):
                pos = text.rfind(boundary_str, search_from, end)
                if pos != -1:
                    end = pos + len(boundary_str)
                    break

        chunk = text[start:end].strip()
        if len(chunk) >= min_chunk:
            chunks.append(chunk)

        # Move forward, stepping back by overlap
        start = end - overlap
        if start <= 0 or (n - start) < min_chunk:
            # Edge-case guard: avoid infinite loop at start of doc
            break

    return chunks


def chunk_document(doc: dict, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Chunk a document dict (from SQLite) into a list of chunk dicts.

    Returns list of:
        {
            "id":          "{title}_{index}",
            "text":        str,
            "metadata": {
                "title":        str,
                "entity_type":  "person" | "place",
                "chunk_index":  int,
                "total_chunks": int,
                "url":          str,
            }
        }
    """
    texts = chunk_text(doc["content"], chunk_size=chunk_size, overlap=overlap)
    total = len(texts)

    return [
        {
            "id": f"{doc['title'].replace(' ', '_')}_{i}",
            "text": text,
            "metadata": {
                "title": doc["title"],
                "entity_type": doc["entity_type"],
                "chunk_index": i,
                "total_chunks": total,
                "url": doc.get("url", ""),
            },
        }
        for i, text in enumerate(texts)
    ]
