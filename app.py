"""
Streamlit chat interface for the Wikipedia RAG system.
BLG483E Project 3 — Caner Yılmaz (820230775)

Run with:  streamlit run app.py
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import OLLAMA_MODEL
from src.generator import check_ollama
from src.ingest import get_document_count, ingest_entities
from src.pipeline import ask, build_index
from src.vector_store import get_chunk_count, reset_collection

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wikipedia RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📚 Local Wikipedia RAG Assistant")
st.caption("BLG483E Project 3 — Caner Yılmaz (820230775) · Fully local · No external API")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    show_sources = st.toggle("Show source chunks", value=True)
    show_classification = st.toggle("Show query classification", value=True)
    model_name = st.text_input("Ollama model", value=OLLAMA_MODEL)

    st.divider()

    # ── Status panel ───────────────────────────────────────────────────────────
    st.subheader("📊 System Status")
    col1, col2 = st.columns(2)
    doc_count = get_document_count()
    chunk_count = get_chunk_count()
    col1.metric("Documents", doc_count)
    col2.metric("Indexed chunks", chunk_count)

    ollama_ok, ollama_msg = check_ollama(model_name)
    if ollama_ok:
        st.success(f"✅ {ollama_msg}")
    else:
        st.error(f"❌ {ollama_msg}")

    st.divider()

    # ── Data management ────────────────────────────────────────────────────────
    st.subheader("🗂️ Data Management")

    if st.button("📥 Ingest Wikipedia Data", use_container_width=True):
        with st.spinner("Fetching Wikipedia pages… (this takes ~2 min)"):
            results = ingest_entities()
        st.success(f"Ingested {len(results['success'])} / {len(results['success']) + len(results['failed'])} entities")
        if results["failed"]:
            st.warning(f"Failed: {', '.join(results['failed'])}")
        st.rerun()

    if st.button("🔗 Build Vector Index", use_container_width=True):
        if doc_count == 0:
            st.error("No documents found. Run ingestion first.")
        else:
            with st.spinner("Embedding and indexing…"):
                stats = build_index(verbose=False)
            st.success(f"Indexed {stats['chunks']} chunks from {stats['documents']} documents.")
            st.rerun()

    with st.expander("⚠️ Danger zone"):
        if st.button("🗑️ Reset Vector Index", use_container_width=True):
            reset_collection()
            st.warning("Vector index cleared.")
            st.rerun()

    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # ── Example queries ────────────────────────────────────────────────────────
    st.subheader("💡 Example Queries")
    examples = [
        "Who was Albert Einstein and what is he known for?",
        "What did Marie Curie discover?",
        "Why is Nikola Tesla famous?",
        "Compare Lionel Messi and Cristiano Ronaldo",
        "What is Frida Kahlo known for?",
        "Where is the Eiffel Tower located?",
        "What was the Colosseum used for?",
        "Which famous place is located in Turkey?",
        "Which person is associated with electricity?",
        "Compare Albert Einstein and Nikola Tesla",
        "Who is the president of Mars?",
    ]
    for ex in examples:
        short = ex[:40] + "…" if len(ex) > 40 else ex
        if st.button(short, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.pending_query = ex
            st.rerun()

# ── Chat history ───────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if show_classification and "entity_type" in msg:
                badge = {"person": "🧑", "place": "📍", "both": "🔍"}.get(
                    msg["entity_type"], "🔍"
                )
                st.caption(f"{badge} Query classified as: **{msg['entity_type']}**")
            if show_sources and msg.get("chunks"):
                with st.expander(f"📚 {len(msg['chunks'])} source chunks"):
                    for i, chunk in enumerate(msg["chunks"]):
                        st.markdown(
                            f"**[{i+1}] {chunk['metadata']['title']}** "
                            f"— chunk {chunk['metadata']['chunk_index']} "
                            f"| similarity: {1 - chunk['distance']:.3f}"
                        )
                        st.caption(chunk["text"][:400] + ("…" if len(chunk["text"]) > 400 else ""))
                        if chunk["metadata"].get("url"):
                            st.markdown(f"[Wikipedia ↗]({chunk['metadata']['url']})")
                        st.divider()


def _handle_query(query: str) -> None:
    """Process a query and append results to session state."""
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if chunk_count == 0:
            answer = (
                "⚠️ The vector index is empty. "
                "Please run **Ingest Wikipedia Data** and then **Build Vector Index** "
                "from the sidebar first."
            )
            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "chunks": [], "entity_type": "both"}
            )
            return

        # Stream the answer token by token
        with st.spinner("Searching knowledge base…"):
            token_gen, chunks, entity_type = ask(query, stream=True, top_k=5)

        answer = st.write_stream(token_gen)

        if show_classification:
            badge = {"person": "🧑", "place": "📍", "both": "🔍"}.get(entity_type, "🔍")
            st.caption(f"{badge} Query classified as: **{entity_type}**")

        if show_sources and chunks:
            with st.expander(f"📚 {len(chunks)} source chunks"):
                for i, chunk in enumerate(chunks):
                    st.markdown(
                        f"**[{i+1}] {chunk['metadata']['title']}** "
                        f"— chunk {chunk['metadata']['chunk_index']} "
                        f"| similarity: {1 - chunk['distance']:.3f}"
                    )
                    st.caption(chunk["text"][:400] + ("…" if len(chunk["text"]) > 400 else ""))
                    if chunk["metadata"].get("url"):
                        st.markdown(f"[Wikipedia ↗]({chunk['metadata']['url']})")
                    st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "chunks": chunks,
            "entity_type": entity_type,
        }
    )


# ── Handle example button clicks ───────────────────────────────────────────────
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None
    _handle_query(query)

# ── Chat input ─────────────────────────────────────────────────────────────────
if user_input := st.chat_input(
    "Ask about a famous person or place…",
    disabled=(not ollama_ok and chunk_count > 0),
):
    _handle_query(user_input)
