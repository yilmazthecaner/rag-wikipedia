"""
Wikipedia ingestion module.
Fetches Wikipedia pages for configured entities and stores raw text in SQLite.
"""

import json
import os
import sqlite3
import time

import requests

from src.config import DB_PATH, DATA_DIR, ENTITIES_FILE

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "BLG483E-Local-Wikipedia-RAG/1.0 (student project)"


# ── Database setup ─────────────────────────────────────────────────────────────

def init_db() -> sqlite3.Connection:
    """Create the SQLite database and documents table if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            requested_title TEXT,
            title         TEXT UNIQUE NOT NULL,
            entity_type   TEXT NOT NULL CHECK(entity_type IN ('person', 'place')),
            content       TEXT NOT NULL,
            url           TEXT,
            word_count    INTEGER,
            ingested_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(documents)")}
    if "requested_title" not in columns:
        conn.execute("ALTER TABLE documents ADD COLUMN requested_title TEXT")
    conn.commit()
    return conn


def get_all_documents() -> list[dict]:
    """Return all raw documents stored in SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, entity_type, content, url FROM documents")
    rows = cursor.fetchall()
    conn.close()
    return [
        {"id": r[0], "title": r[1], "entity_type": r[2], "content": r[3], "url": r[4]}
        for r in rows
    ]


def get_document_count() -> int:
    try:
        conn = sqlite3.connect(DB_PATH)
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def _document_exists(cursor: sqlite3.Cursor, requested_title: str) -> bool:
    """Return True when this configured entity has already been ingested."""
    cursor.execute(
        """
        SELECT 1
        FROM documents
        WHERE lower(title) = lower(?)
           OR lower(requested_title) = lower(?)
        LIMIT 1
        """,
        (requested_title, requested_title),
    )
    return cursor.fetchone() is not None


# ── Wikipedia fetching ─────────────────────────────────────────────────────────

def _fetch_page(title: str) -> dict | None:
    """
    Fetch a Wikipedia page over HTTPS. Returns None on failure.

    The older ``wikipedia`` package can intermittently fail when Wikipedia
    rejects plain HTTP requests. Calling the MediaWiki API directly keeps the
    ingestion step reliable and still uses only Wikipedia as the data source.
    """
    max_attempts = 3
    backoff_base = 0.5

    def _request(params: dict) -> dict:
        resp = requests.get(
            WIKIPEDIA_API_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=(5, 30),
        )
        resp.raise_for_status()
        return resp.json()

    def _search_title(t: str) -> str | None:
        payload = _request(
            {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": t,
                "srlimit": 1,
            }
        )
        matches = payload.get("query", {}).get("search", [])
        return matches[0]["title"] if matches else None

    def _attempt_fetch(t: str) -> dict | None:
        payload = _request(
            {
                "action": "query",
                "format": "json",
                "prop": "extracts|info|pageprops",
                "explaintext": 1,
                "exsectionformat": "wiki",
                "inprop": "url",
                "redirects": 1,
                "titles": t,
            }
        )

        pages = payload.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})

        if "missing" in page:
            suggested = _search_title(t)
            if suggested and suggested != t:
                return _attempt_fetch(suggested)
            return None

        # Avoid indexing disambiguation pages; search usually returns the
        # canonical article first for titles such as "Petra, Jordan".
        if "disambiguation" in page.get("pageprops", {}):
            suggested = _search_title(t)
            if suggested and suggested != page.get("title"):
                return _attempt_fetch(suggested)
            return None

        content = page.get("extract", "").strip()
        if not content:
            return None

        return {
            "title": page["title"],
            "content": content,
            "url": page.get("fullurl", ""),
        }

    # Retry loop with exponential backoff for transient failures
    for attempt in range(1, max_attempts + 1):
        try:
            return _attempt_fetch(title)
        except Exception as e:
            if attempt == max_attempts:
                print(f"  Error fetching '{title}' after {attempt} attempts: {e}")
                return None
            sleep_time = backoff_base * (2 ** (attempt - 1))
            time.sleep(sleep_time)
            continue


# ── Public API ─────────────────────────────────────────────────────────────────

def ingest_entities(
    entities_file: str = ENTITIES_FILE,
    delay: float = 1.0,
    skip_existing: bool = True,
) -> dict:
    """
    Main ingestion function. Reads entities.json, fetches Wikipedia pages,
    stores them in SQLite.

    Args:
        entities_file: Path to entities.json
        delay:         Seconds to wait between Wikipedia requests (politeness)
        skip_existing: Reuse already-ingested documents instead of refetching

    Returns:
        {"success": [...], "failed": [...]}
    """
    with open(entities_file, "r", encoding="utf-8") as f:
        entities = json.load(f)

    conn = init_db()
    cursor = conn.cursor()
    results = {"success": [], "failed": []}

    entity_pairs = (
        [("person", name) for name in entities.get("people", [])]
        + [("place", name) for name in entities.get("places", [])]
    )

    for entity_type, name in entity_pairs:
        if skip_existing and _document_exists(cursor, name):
            print(f"[{entity_type.upper()}] Cached: {name}")
            results["success"].append(name)
            continue

        print(f"[{entity_type.upper()}] Fetching: {name} ...", end=" ", flush=True)
        data = _fetch_page(name)

        if data is None:
            print("✗ FAILED")
            results["failed"].append(name)
            continue

        word_count = len(data["content"].split())
        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO documents
                    (requested_title, title, entity_type, content, url, word_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    data["title"],
                    entity_type,
                    data["content"],
                    data["url"],
                    word_count,
                ),
            )
            conn.commit()
            results["success"].append(name)
            print(f"✓  ({word_count:,} words)")
        except sqlite3.Error as e:
            print(f"✗ DB error: {e}")
            results["failed"].append(name)

        time.sleep(delay)

    conn.close()
    return results
