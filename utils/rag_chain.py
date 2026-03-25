# utils/rag_chain.py
"""
Component 3: Grounded Generation
- Formats retrieved chunks into a numbered context block
- Injects into RAG prompt template
- Calls Mistral via Ollama
- Returns response + source citations
"""

import json
import requests
from pathlib import Path
from typing import List, Dict, Tuple

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    OLLAMA_BASE_URL, LLM_MODEL, LLM_TEMPERATURE,
    RAG_SYSTEM_PROMPT, BASELINE_SYSTEM_PROMPT,
    RETRIEVAL_K
)
from utils.vectorstore import VectorStore


def _call_ollama(system: str, user: str) -> str:
    """Send a chat request to Ollama and return the assistant text."""
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {"temperature": LLM_TEMPERATURE},
    }
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return "⚠️ Cannot connect to Ollama. Make sure it is running: `ollama serve`"
    except Exception as e:
        return f"⚠️ LLM error: {e}"


def format_context(chunks: List[Dict]) -> Tuple[str, str]:
    """
    Build the numbered context block and sources list
    that get injected into the RAG prompt.
    Returns (context_str, sources_str).
    """
    context_parts = []
    source_parts  = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[{i}] {chunk['text']}")
        page_note = f", p.{chunk['page']}" if chunk.get("page") else ""
        source_parts.append(f"[{i}] {chunk['source']}{page_note} — {chunk['url']}")

    return "\n\n".join(context_parts), "\n".join(source_parts)


class RAGChain:
    """Full RAG pipeline: retrieve → format → generate."""

    def __init__(self, vectorstore: VectorStore):
        self.vs = vectorstore

    def answer(self, question: str, k: int = RETRIEVAL_K) -> Dict:
        """
        Run the RAG pipeline for a question.
        Returns dict with keys: answer, chunks, sources_md
        """
        # 1. Retrieve
        chunks = self.vs.retrieve(question, k=k)

        if not chunks:
            return {
                "answer": "I don't have information about this in my knowledge base.",
                "chunks": [],
                "sources_md": "",
            }

        # 2. Format context
        context_str, sources_str = format_context(chunks)

        # 3. Fill prompt
        user_prompt = RAG_SYSTEM_PROMPT.format(
            context=context_str,
            sources=sources_str,
            question=question,
        )

        # 4. Call LLM (system prompt is empty; everything is in user turn for Mistral)
        answer = _call_ollama(
            system="You are a helpful healthcare assistant. Follow the instructions in the user message exactly.",
            user=user_prompt,
        )

        # 5. Build readable sources markdown
        sources_md = "\n".join(
            f"**[{i+1}]** {c['source']} *(relevance: {c['score']:.2f})*  \n{c['url']}"
            for i, c in enumerate(chunks)
        )

        return {
            "answer":     answer,
            "chunks":     chunks,
            "sources_md": sources_md,
        }

    def answer_baseline(self, question: str) -> str:
        """A1-style answer: pure LLM, no retrieval."""
        return _call_ollama(
            system=BASELINE_SYSTEM_PROMPT,
            user=question,
        )
