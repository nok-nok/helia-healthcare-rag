# utils/vectorstore.py
"""
Component 2: Embedding and Retrieval
- Embeds chunks with sentence-transformers (local, no API key)
- Persists to ChromaDB
- Implements hybrid retrieval: dense cosine similarity + BM25
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    CHROMA_PERSIST_DIR, COLLECTION_NAME,
    EMBEDDING_MODEL, RETRIEVAL_K, HYBRID_ALPHA
)


class VectorStore:
    """Wraps ChromaDB + local embeddings + BM25 for hybrid retrieval."""

    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        # BM25 index rebuilt from stored docs when needed
        self._bm25: BM25Okapi = None
        self._bm25_docs: List[Dict] = []

    # ── Indexing ──────────────────────────────────────────────

    def index_chunks(self, chunks: List[Dict], batch_size: int = 64):
        """Embed and store all chunks. Skips if already indexed."""
        existing = self.collection.count()
        if existing > 0:
            print(f"  Collection already has {existing} chunks – skipping re-index.")
            print("  (Delete ./data/chroma_db to force re-index)")
            self._rebuild_bm25()
            return

        print(f"Embedding {len(chunks)} chunks (this may take a minute)...")
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            texts = [c["text"] for c in batch]
            embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()
            ids = [f"chunk_{i + j}" for j in range(len(batch))]
            metadatas = [
                {
                    "source": c["source"],
                    "url":    c["url"][:500],   # Chroma limit
                    "type":   c["type"],
                    "chunk_index": c["chunk_index"],
                    "page":   str(c.get("page") or ""),
                }
                for c in batch
            ]
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
            )
            print(f"  Indexed {min(i + batch_size, len(chunks))}/{len(chunks)}")

        self._rebuild_bm25()
        print(f"✓ Indexed {self.collection.count()} chunks into ChromaDB")

    def _rebuild_bm25(self):
        """Load all stored docs and build a BM25 index."""
        result = self.collection.get(include=["documents", "metadatas"])
        self._bm25_docs = [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(result["documents"], result["metadatas"])
        ]
        tokenized = [d["text"].lower().split() for d in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenized)

    # ── Retrieval ─────────────────────────────────────────────

    def retrieve(self, query: str, k: int = RETRIEVAL_K) -> List[Dict]:
        """
        Hybrid retrieval: combine dense cosine scores with BM25 scores.
        Returns top-k chunks with metadata.
        """
        if self.collection.count() == 0:
            return []

        # -- Dense retrieval
        q_emb = self.embedder.encode([query]).tolist()
        dense_results = self.collection.query(
            query_embeddings=q_emb,
            n_results=min(k * 2, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        dense_docs  = dense_results["documents"][0]
        dense_metas = dense_results["metadatas"][0]
        dense_dists = dense_results["distances"][0]

        # Normalise distance to similarity score (cosine distance → similarity)
        dense_scores = {
            dense_docs[i]: 1 - dense_dists[i]
            for i in range(len(dense_docs))
        }

        if not self._bm25:
            self._rebuild_bm25()

        # -- BM25 retrieval
        bm25_scores_raw = self._bm25.get_scores(query.lower().split())
        # Normalise to [0, 1]
        max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1
        bm25_map = {
            self._bm25_docs[i]["text"]: bm25_scores_raw[i] / max_bm25
            for i in range(len(self._bm25_docs))
        }

        # -- Fuse scores for docs in dense results
        fused: Dict[str, Tuple] = {}
        for text, meta in zip(dense_docs, dense_metas):
            d_score = dense_scores.get(text, 0)
            b_score = bm25_map.get(text, 0)
            hybrid  = HYBRID_ALPHA * d_score + (1 - HYBRID_ALPHA) * b_score
            fused[text] = (hybrid, meta)

        # Sort and return top-k
        ranked = sorted(fused.items(), key=lambda x: x[1][0], reverse=True)[:k]
        return [
            {
                "text":     text,
                "source":   meta["source"],
                "url":      meta["url"],
                "score":    round(score, 4),
                "page":     meta.get("page", ""),
            }
            for text, (score, meta) in ranked
        ]

    def is_empty(self) -> bool:
        return self.collection.count() == 0

    def doc_count(self) -> int:
        return self.collection.count()
