"""
Wikipedia ingestion module.
Fetches Wikipedia pages for configured entities and stores raw text in SQLite.
"""

import json
import os
import sqlite3
import time

import wikipedia

from src.config import DB_PATH, DATA_DIR, ENTITIES_FILE


# ── Database setup ─────────────────────────────────────────────────────────────

def init_db() -> sqlite3.Connection:
    """Create the SQLite database and documents table if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            title         TEXT UNIQUE NOT NULL,
            entity_type   TEXT NOT NULL CHECK(entity_type IN ('person', 'place')),
            content       TEXT NOT NULL,
            url           TEXT,
            word_count    INTEGER,
            ingested_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
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


# ── Wikipedia fetching ─────────────────────────────────────────────────────────

def _fetch_page(title: str) -> dict | None:
    """
    Fetch a Wikipedia page. Handles disambiguation errors by picking the first
    option that works. Returns None on failure.
    """
    wikipedia.set_lang("en")

    max_attempts = 3
    backoff_base = 0.5

    def _attempt_fetch(t: str) -> dict | None:
        try:
            page = wikipedia.page(t, auto_suggest=False)
            return {"title": page.title, "content": page.content, "url": page.url}
        except wikipedia.DisambiguationError as e:
            for option in e.options[:3]:
                try:
                    page = wikipedia.page(option, auto_suggest=False)
                    return {"title": page.title, "content": page.content, "url": page.url}
                except Exception:
                    continue
            return None
        except wikipedia.PageError:
            # Try with auto_suggest enabled as fallback
            try:
                results = wikipedia.search(t, results=1)
                if results:
                    page = wikipedia.page(results[0], auto_suggest=False)
                    return {"title": page.title, "content": page.content, "url": page.url}
            except Exception:
                pass
            return None
        except Exception as e:
            # Bubble up network/temporary errors for retry
            raise

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

def ingest_entities(entities_file: str = ENTITIES_FILE, delay: float = 0.5) -> dict:
    """
    Main ingestion function. Reads entities.json, fetches Wikipedia pages,
    stores them in SQLite.

    Args:
        entities_file: Path to entities.json
        delay:         Seconds to wait between Wikipedia requests (politeness)

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
                    (title, entity_type, content, url, word_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (data["title"], entity_type, data["content"], data["url"], word_count),
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
