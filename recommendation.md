# Production Deployment Recommendation

**Project:** Local Wikipedia RAG Assistant  
**Author:** Caner Yılmaz (820230775)

---

## 1. Executive Summary

This document describes how the current local proof-of-concept should be adapted for a production-grade, scalable deployment. The core RAG architecture is sound; the main changes concern infrastructure, scalability, observability, and model quality.

---

## 2. Current Architecture Limitations

| Limitation | Impact |
|-----------|--------|
| SQLite for raw storage | No concurrent writes; not suitable for multi-user ingestion |
| ChromaDB on local disk | Not horizontally scalable; no replication |
| Ollama on localhost | One model instance; no load balancing; latency on CPU |
| Blocking Wikipedia fetcher | Single-threaded; slow for large corpora |
| No authentication | Cannot be exposed publicly |
| No caching | Every identical query calls the LLM |

---

## 3. Recommended Production Stack

### 3.1 Data Ingestion

| Component | Recommendation | Reason |
|-----------|---------------|--------|
| Raw storage | **PostgreSQL** (with `pgvector` extension) | Concurrent writes, ACID guarantees, full-text search built-in |
| Ingestion orchestration | **Apache Airflow** or **Prefect** | Scheduled re-ingestion, failure retries, observability |
| Fetching strategy | Async (`asyncio` + `aiohttp`) with rate limiting | 5-10× throughput improvement |

### 3.2 Embeddings

| Aspect | Recommendation |
|--------|---------------|
| Model | Upgrade to `BAAI/bge-large-en-v1.5` (1024-dim) or `text-embedding-3-small` (OpenAI, if external API allowed) |
| Serving | Deploy as a dedicated microservice (FastAPI + GPU instance) |
| Batching | Queue embedding jobs; process in GPU batches of 256+ |

### 3.3 Vector Store

| Option | When to use |
|--------|------------|
| **pgvector** (PostgreSQL extension) | < 10 M vectors; prefer unified DB stack |
| **Qdrant** | Dedicated vector DB; best OSS performance/filtering at scale |
| **Pinecone** (managed) | Fully managed; zero infrastructure; higher cost |

**Recommendation:** Qdrant for self-hosted or Pinecone for managed. Both support metadata filtering natively (equivalent to our `entity_type` filter).

### 3.4 LLM Serving

| Option | Pros | Cons |
|--------|------|------|
| **vLLM** (self-hosted GPU) | High throughput (continuous batching), OpenAI-compatible API | Requires GPU server |
| **Ollama** (current) | Zero-config, CPU-friendly | Low throughput, no batching |
| **LiteLLM proxy** | Unified API over any backend | Extra hop |
| **OpenAI / Anthropic API** | Best quality; zero infra | Cost; data leaves premises |

**Recommendation:** Deploy `llama3.1:8b` or `mistral:7b-instruct` via **vLLM** on a single A10G or RTX 4090. For production factual QA, consider fine-tuning on Wikipedia-style data.

### 3.5 API Layer

Replace Streamlit with a proper API:

```
FastAPI backend
  ├── POST /query         — RAG query endpoint
  ├── POST /ingest        — trigger ingestion job
  ├── GET  /status        — health + index stats
  └── WebSocket /stream   — streaming token delivery
```

Serve the frontend separately (Next.js or React) to decouple UI from compute.

### 3.6 Caching

- **Query-level cache:** Redis (exact-match query → answer, TTL = 1 hour).
- **Embedding cache:** Cache embeddings in PostgreSQL alongside raw text.

### 3.7 Authentication & Security

- JWT-based authentication on all API endpoints.
- Rate limiting per user (max 60 queries/minute).
- Input sanitisation to prevent prompt injection.

---

## 4. Scalability Targets

| Metric | Current (local) | Production target |
|--------|----------------|-----------------|
| Concurrent users | 1 | 100+ |
| Documents | 40 | 100,000+ |
| Chunks | ~3,000 | 10,000,000+ |
| Query latency | 5–30 s | < 2 s (p95) |
| Ingestion throughput | ~10 pages/min | ~1,000 pages/min |

---

## 5. Observability

- **Logging:** Structured JSON logs (entity queried, retrieval latency, generation latency, model used).
- **Metrics:** Prometheus + Grafana — track token throughput, cache hit rate, embedding queue depth.
- **Tracing:** OpenTelemetry spans across ingest → embed → retrieve → generate.
- **Evaluation:** Periodic offline RAG evaluation using `RAGAS` framework (faithfulness, answer relevancy, context precision).

---

## 6. Deployment Topology

```
                         ┌─────────────┐
                         │  Frontend   │  (Next.js / React)
                         └──────┬──────┘
                                │ HTTPS
                         ┌──────▼──────┐
                         │  FastAPI    │  (+ Redis cache)
                         └──┬──────┬──┘
                   Embed     │      │  Generate
          ┌────────────────┐ │      │ ┌──────────────────┐
          │ Embedding svc  │◄┘      └►│   vLLM server    │
          │ (BGE-large)    │          │ (llama3.1 / GPU) │
          └────────────────┘          └──────────────────┘
                    │                          │
                    ▼                          ▼
          ┌─────────────────┐      ┌──────────────────────┐
          │   Qdrant        │      │   PostgreSQL          │
          │ (vector store)  │      │ (raw docs + metadata) │
          └─────────────────┘      └──────────────────────┘
```

---

## 7. Cost Estimate (self-hosted, 100 users)

| Resource | Spec | Monthly cost (est.) |
|---------|------|---------------------|
| LLM GPU server | 1× A10G (24 GB VRAM) | ~$300 |
| Embedding server | 1× T4 GPU or CPU | ~$80 |
| Qdrant / PostgreSQL | 8-core, 32 GB RAM, 500 GB SSD | ~$150 |
| API server | 4-core, 8 GB RAM | ~$40 |
| **Total** | | **~$570 / month** |

For managed alternatives (Pinecone + OpenAI API), cost shifts toward per-query pricing but eliminates infrastructure management.

---

## 8. Migration Path

1. **Phase 1 (current):** Local proof-of-concept — SQLite + ChromaDB + Ollama. ✅
2. **Phase 2:** Containerise with Docker Compose (FastAPI + ChromaDB + Ollama containers).
3. **Phase 3:** Replace ChromaDB with Qdrant; PostgreSQL for raw storage; Redis cache.
4. **Phase 4:** vLLM on GPU; async ingestion pipeline; authentication.
5. **Phase 5:** Monitoring, evaluation loop, CI/CD.
