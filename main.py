#!/usr/bin/env python3
"""
CLI entry point for the Wikipedia RAG system.
BLG483E Project 3 — Caner Yılmaz (820230775)

Usage:
    python main.py ingest     # Fetch Wikipedia pages → SQLite
    python main.py index      # Chunk → embed → ChromaDB
    python main.py chat       # Interactive CLI chat
    python main.py status     # System health check
    python main.py all        # ingest + index + chat
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.generator import check_ollama
from src.ingest import get_document_count, ingest_entities
from src.pipeline import ask, build_index
from src.vector_store import get_chunk_count, reset_collection

BANNER = """
╔══════════════════════════════════════════════════════╗
║        Local Wikipedia RAG Assistant                 ║
║        BLG483E Project 3                             ║
║        Caner Yılmaz  ·  820230775                   ║
╚══════════════════════════════════════════════════════╝
"""


# ── Commands ───────────────────────────────────────────────────────────────────

def cmd_status() -> None:
    print(BANNER)
    print("── System Status ──────────────────────────────────────")
    print(f"  Documents in SQLite : {get_document_count()}")
    print(f"  Chunks in ChromaDB  : {get_chunk_count()}")

    from src.config import OLLAMA_MODEL
    ok, msg = check_ollama(OLLAMA_MODEL)
    status = "✅" if ok else "❌"
    print(f"  Ollama              : {status}  {msg}")
    print()


def cmd_ingest() -> None:
    print("Starting Wikipedia ingestion…")
    results = ingest_entities()
    print(f"\n✓ Success : {len(results['success'])} entities")
    if results["failed"]:
        print(f"✗ Failed  : {', '.join(results['failed'])}")


def cmd_index() -> None:
    if get_document_count() == 0:
        print("✗ No documents found. Run `python main.py ingest` first.")
        sys.exit(1)
    stats = build_index(verbose=True)
    print(f"\nDone. Indexed {stats['chunks']} chunks from {stats['documents']} documents.")


def cmd_chat() -> None:
    print(BANNER)
    print("Commands: 'quit'/'q' to exit | 'sources' to toggle source display")
    print("─" * 58)

    if get_chunk_count() == 0:
        print("⚠  Vector index is empty. Run `python main.py index` first.")
        print("   You can still ask questions — retrieval will return nothing.\n")

    show_sources = False

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "q", "exit"):
            print("Goodbye!")
            break
        if query.lower() == "sources":
            show_sources = not show_sources
            print(f"[Source display: {'ON' if show_sources else 'OFF'}]")
            continue
        if query.lower() == "status":
            cmd_status()
            continue

        answer, chunks, entity_type = ask(query)

        print(f"\n[Classified as: {entity_type} | {len(chunks)} chunks retrieved]")
        print(f"\nAssistant: {answer}\n")

        if show_sources and chunks:
            print("Sources:")
            for i, c in enumerate(chunks):
                title = c["metadata"]["title"]
                snippet = c["text"][:120].replace("\n", " ")
                sim = 1 - c["distance"]
                print(f"  [{i+1}] {title}  (sim={sim:.3f})")
                print(f"       {snippet}…")
            print()


def cmd_reset() -> None:
    confirm = input("This will delete the entire vector index. Type 'yes' to confirm: ")
    if confirm.strip().lower() == "yes":
        reset_collection()
        print("Vector index reset.")
    else:
        print("Aborted.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wikipedia RAG CLI — BLG483E Project 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "command",
        choices=["ingest", "index", "chat", "status", "reset", "all"],
        help="Command to run",
    )
    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest()
    elif args.command == "index":
        cmd_index()
    elif args.command == "chat":
        cmd_chat()
    elif args.command == "status":
        cmd_status()
    elif args.command == "reset":
        cmd_reset()
    elif args.command == "all":
        cmd_ingest()
        cmd_index()
        cmd_chat()


if __name__ == "__main__":
    main()
