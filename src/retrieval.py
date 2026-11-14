"""Day 6 pgvector dense search, BM25 lexical search, and reciprocal-rank fusion."""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .chunking import by_paragraph
from .embeddings import MyEmbedder
from .models import DocumentPage
from .store import PgVectorStore

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab")


def load_chunks() -> list[dict[str, Any]]:
    pages = [DocumentPage(**json.loads(line)) for line in
             (ROOT / "data/day6/corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    return [{"id": c.chunk_id, "source_id": c.document_id, "source": c.source,
             "page": c.page, "text": c.text, "metadata": c.metadata}
            for page in pages for c in by_paragraph(page, 650)]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


class BM25:
    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks
        self.rows = [Counter(tokenize(c["text"])) for c in chunks]
        self.lengths = [sum(row.values()) for row in self.rows]
        self.average_length = sum(self.lengths) / len(self.lengths)
        df = Counter(term for row in self.rows for term in row)
        self.idf = {term: math.log(1 + (len(chunks) - n + 0.5) / (n + 0.5))
                    for term, n in df.items()}

    def search(self, query: str, top_k: int, filters: dict | None = None) -> list[dict]:
        terms = set(tokenize(query))
        scored = []
        for chunk, row, length in zip(self.chunks, self.rows, self.lengths):
            if filters and any(chunk["metadata"].get(k) != v for k, v in filters.items()):
                continue
            score = sum(self.idf.get(term, 0) * (row[term] * 2.2) /
                        (row[term] + 1.2 * (0.25 + 0.75 * length / self.average_length))
                        for term in terms if row[term])
            if score > 0:
                scored.append({**chunk, "score": score, "strategy": "bm25"})
        return sorted(scored, key=lambda c: (-c["score"], c["id"]))[:top_k]


class RetrievalSystem:
    def __init__(self, *, use_dense: bool = True):
        self.chunks = load_chunks()
        self.lexical = BM25(self.chunks)
        self.embedder = MyEmbedder(MODEL_NAME) if use_dense else None
        self.store = PgVectorStore(DATABASE_URL, self.embedder.dimensions) if self.embedder else None

    def index(self) -> int:
        if not self.embedder or not self.store:
            raise ValueError("Dense embedder required")
        pages = [DocumentPage(**json.loads(line)) for line in
                 (ROOT / "data/day6/corpus.jsonl").read_text(encoding="utf-8").splitlines()]
        chunks = [c for page in pages for c in by_paragraph(page, 650)]
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self.store.create_schema()
        self.store.upsert("C", chunks, vectors)
        return len(chunks)

    def search(self, query: str, method: str, top_k: int = 3,
               filters: dict | None = None) -> list[dict]:
        if method not in ("dense", "bm25", "hybrid") or top_k < 1:
            raise ValueError("Invalid retrieval method or top_k")
        if filters and set(filters) - {"chapter"}:
            raise ValueError("Only chapter filter is supported")
        if method == "bm25":
            return self.lexical.search(query, top_k, filters)
        if not self.embedder or not self.store:
            raise ValueError("Dense search requires the Day 6 model and vector store")
        query_vector = self.embedder.embed_query(query)
        raw = self.store.search("C", query_vector, max(top_k, 30) if method == "hybrid" else top_k, filters)
        dense = [{"id": c["chunk_id"], "source_id": c["document_id"],
                  "source": c["source"], "page": c["page"], "text": c["text"],
                  "metadata": c["metadata"], "score": c["score"], "strategy": "dense"}
                 for c in raw]
        if method == "dense":
            return dense
        lexical = self.lexical.search(query, max(top_k, 30), filters)
        fused: dict[str, dict] = {}
        for ranking in (dense, lexical):
            for rank, chunk in enumerate(ranking, 1):
                row = fused.setdefault(chunk["id"], {**chunk, "score": 0.0,
                                                     "strategy": "hybrid", "ranks": {}})
                row["score"] += 1 / (60 + rank)
                row["ranks"][ranking[0]["strategy"]] = rank
        return sorted(fused.values(), key=lambda c: (-c["score"], c["id"]))[:top_k]


def pack_context(results: list[dict], budget_chars: int) -> tuple[str, list[str]]:
    blocks, ids = [], []
    for c in results:
        block = f"[{c['id']} | {c['source']} | unit {c['page']}]\n{c['text']}"
        candidate = "\n\n".join([*blocks, block])
        if len(candidate) <= budget_chars and c["id"] not in ids:
            blocks.append(block)
            ids.append(c["id"])
    return "\n\n".join(blocks), ids


def evidence_hit(chunk: dict, gold: list[dict]) -> bool:
    normalize = lambda text: " ".join(text.split())
    return any(chunk["source_id"] == g["source_id"] and chunk["page"] == g["page"]
               and normalize(g["evidence"]) in normalize(chunk["text"]) for g in gold)
