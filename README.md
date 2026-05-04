# Local Wikipedia RAG Assistant

**BLG483E — AI Aided Computer Engineering · Project 3**  
**Student:** Caner Yılmaz · **No:** 820230775

A fully local Retrieval-Augmented Generation (RAG) system that answers questions about famous people and places using Wikipedia data. No external API is used at any stage.

---

## Architecture Overview

```
User Query
    │
    ▼
[Query Classifier]  ──── keyword-based → person / place / both
    │
    ▼
[Embedder]          ──── all-MiniLM-L6-v2 (sentence-transformers, local)
    │
    ▼
[ChromaDB]          ──── cosine similarity search, metadata filter
    │
    ▼
[Ollama LLM]        ──── llama3.2:3b (fully local)
    │
    ▼
[Answer]
```

**Design choices:**

- **Option B** (one ChromaDB collection + `entity_type` metadata) — simpler, more flexible for mixed queries.
- **Chunking:** 500-char fixed-size chunks with 50-char overlap, sentence-boundary aware.
- **Embeddings:** `all-MiniLM-L6-v2` — 384-dim, ~22 MB, fast CPU inference, no API needed.
- **LLM:** `llama3.2:3b` via Ollama — runs entirely on laptop.

---

## Prerequisites

| Tool   | Version | Install                          |
| ------ | ------- | -------------------------------- |
| Python | ≥ 3.10  | [python.org](https://python.org) |
| Ollama | latest  | [ollama.com](https://ollama.com) |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yilmazthecaner/blg483e-rag-project.git
cd blg483e-rag-project
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> The first run will download the `all-MiniLM-L6-v2` model (~22 MB) automatically.

### 4. Install and start Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start the Ollama daemon
ollama serve
```

### 5. Pull the LLM

```bash
ollama pull llama3.2:3b
```

> Alternative models: `phi3`, `mistral`. If you use a different model, update `OLLAMA_MODEL` in `src/config.py`.

---

## Running the System

### Step 1 — Ingest Wikipedia data

```bash
python main.py ingest
```

Fetches Wikipedia pages for 20 people and 20 places, stores raw text in `data/wikipedia.db`.  
Duration: ~2–3 minutes (network-dependent).

### Step 2 — Build the vector index

```bash
python main.py index
```

Chunks all documents, generates embeddings, and stores them in ChromaDB.  
Duration: ~1–2 minutes on CPU.

### Step 3a — Start the Streamlit UI (recommended)

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Step 3b — Use the CLI instead

```bash
python main.py chat
```

Type your question at the `You:` prompt. Type `sources` to toggle source display, `quit` to exit.

### All-in-one

```bash
python main.py all
```

Runs ingest → index → chat sequentially.

### System status

```bash
python main.py status
```

---

## Example Queries

### People

```
Who was Albert Einstein and what is he known for?
What did Marie Curie discover?
Why is Nikola Tesla famous?
Compare Lionel Messi and Cristiano Ronaldo
What is Frida Kahlo known for?
```

### Places

```
Where is the Eiffel Tower located?
Why is the Great Wall of China important?
What is Machu Picchu?
What was the Colosseum used for?
Where is Mount Everest?
```

### Mixed

```
Which famous place is located in Turkey?
Which person is associated with electricity?
Compare Albert Einstein and Nikola Tesla
Compare the Eiffel Tower and the Statue of Liberty
```

### Expected failure cases (system should say "I don't know")

```
Who is the president of Mars?
Tell me about John Doe
```

---

## Repository Structure

```
.
├── app.py                  # Streamlit chat UI
├── main.py                 # CLI entry point
├── requirements.txt
├── README.md
├── Product_prd.md          # Product Requirements Document
├── recommendation.md       # Production deployment guide
├── data/
│   ├── entities.json       # List of 20 people + 20 places
│   ├── wikipedia.db        # SQLite (created after ingest)
│   └── chroma_db/          # ChromaDB (created after index)
└── src/
    ├── config.py           # Central configuration
    ├── ingest.py           # Wikipedia fetching + SQLite
    ├── chunker.py          # Text chunking strategy
    ├── embedder.py         # sentence-transformers wrapper
    ├── vector_store.py     # ChromaDB wrapper
    ├── retriever.py        # Query classifier + retrieval
    ├── generator.py        # Ollama LLM wrapper (streaming)
    └── pipeline.py         # End-to-end orchestration
```

---

## Configuration

Edit `src/config.py` to adjust:

| Parameter         | Default            | Description                |
| ----------------- | ------------------ | -------------------------- |
| `CHUNK_SIZE`      | 500                | Characters per chunk       |
| `CHUNK_OVERLAP`   | 50                 | Overlap between chunks     |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `OLLAMA_MODEL`    | `llama3.2:3b`      | Local LLM                  |
| `TOP_K`           | 5                  | Chunks retrieved per query |
| `RETRIEVAL_MAX_DISTANCE` | 0.62         | Reject weak matches        |

---

## Demo Video

Demo link: https://youtu.be/F9CTXMH5Kp8

### Runtime note

If Ollama is slow to respond, the UI now shows a short context-based fallback answer instead of staying blank. This keeps example queries usable even when the local model is still loading.
