"""
Answer generation module.
Calls the local Ollama LLM with retrieved context (RAG pattern).

No external API is used. The model runs entirely on localhost via Ollama.
"""

from __future__ import annotations

import json
import re
from typing import Generator

import requests

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL

# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a factual assistant that answers questions about famous people and places.
Rules you must follow:
1. Answer ONLY based on the context provided below.
2. If the answer is not in the context, reply exactly: "I don't know based on my available data."
3. Do NOT invent facts, dates, names, or statistics.
4. Be concise but complete. Prefer short paragraphs over bullet lists.
5. When comparing two entities, structure your answer clearly."""


def _build_prompt(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return query

    context_blocks = []
    for c in chunks:
        title = c["metadata"].get("title", "Unknown")
        context_blocks.append(f"[Source: {title}]\n{c['text']}")

    context = "\n\n---\n\n".join(context_blocks)
    return f"""Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"""


def _extract_summary(text: str, max_sentences: int = 2, max_chars: int = 260) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    summary = " ".join(sentences[:max_sentences]).strip()
    if len(summary) > max_chars:
        summary = summary[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    return summary


def _fallback_answer(chunks: list[dict]) -> str:
    if not chunks:
        return "I don't know based on my available data."

    pieces = []
    for chunk in chunks[:2]:
        title = chunk["metadata"].get("title", "Unknown")
        summary = _extract_summary(chunk["text"])
        if summary:
            pieces.append(f"{title}: {summary}")

    if not pieces:
        return "I don't know based on my available data."

    return "Based on the available context: " + " ".join(pieces)


# ── Non-streaming generation ───────────────────────────────────────────────────

def generate_answer(
    query: str,
    chunks: list[dict],
    model: str = OLLAMA_MODEL,
) -> str:
    """
    Generate an answer for *query* given retrieved *chunks*.
    Returns the complete answer string (blocking).
    """
    if not chunks:
        return "I don't know based on my available data."

    prompt = _build_prompt(query, chunks)
    fallback = _fallback_answer(chunks)

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "temperature": 0.1,   # low temperature → more factual
                    "top_p": 0.9,
                    "num_predict": 128,
                },
            },
            timeout=(5, 20),
        )
        resp.raise_for_status()
        return resp.json().get("response", "Error: Empty response from model.")

    except requests.exceptions.ConnectionError:
        return (
            "⚠️  Cannot connect to Ollama. "
            "Please run `ollama serve` and ensure your model is pulled "
            f"(`ollama pull {model}`)."
        )
    except requests.exceptions.Timeout:
        return (
            "⚠️  Ollama took too long to respond. "
            f"{fallback}"
        )
    except Exception as e:
        return f"⚠️  Generation error: {e}"


# ── Streaming generation ───────────────────────────────────────────────────────

def stream_answer(
    query: str,
    chunks: list[dict],
    model: str = OLLAMA_MODEL,
) -> Generator[str, None, None]:
    """
    Streaming version of generate_answer.
    Yields token strings as they arrive from Ollama.
    """
    if not chunks:
        yield "I don't know based on my available data."
        return

    prompt = _build_prompt(query, chunks)
    fallback = _fallback_answer(chunks)

    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": True,
                "keep_alive": "10m",
                "options": {"temperature": 0.1, "num_predict": 128},
            },
            stream=True,
            timeout=(5, 20),
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        break

    except requests.exceptions.ConnectionError:
        yield (
            "\n⚠️  Cannot connect to Ollama. "
            f"Run `ollama serve` and `ollama pull {model}`."
        )
    except requests.exceptions.Timeout:
        yield f"⚠️  Ollama took too long to respond. {fallback}"
    except Exception as e:
        yield f"\n⚠️  Generation error: {e}"


# ── Ollama health check ────────────────────────────────────────────────────────

def check_ollama(model: str = OLLAMA_MODEL) -> tuple[bool, str]:
    """
    Returns (ok, message).
    Checks if Ollama is running and the model is available.
    """
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        # Accept prefix match (e.g. "llama3.2:3b" matches "llama3.2:3b-instruct")
        model_ok = any(m.startswith(model.split(":")[0]) for m in models)
        if model_ok:
            return True, f"Ollama running. Model '{model}' available."
        return (
            False,
            f"Ollama running but model '{model}' not found. "
            f"Available: {models}. Run: ollama pull {model}",
        )
    except requests.exceptions.ConnectionError:
        return False, "Ollama not running. Start with: ollama serve"
    except Exception as e:
        return False, f"Ollama check failed: {e}"
