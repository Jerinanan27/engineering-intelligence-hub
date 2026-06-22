"""Retrieval pipeline: dense + sparse, fused, then reranked.

Why hybrid? Dense (semantic) retrieval is great for paraphrase but weak on exact
identifiers — function names, error codes, service names like `auth-svc`. BM25
(lexical) nails those but misses synonyms. Reciprocal Rank Fusion combines both
rank lists without needing to calibrate incomparable score scales:

    RRF(d) = sum_over_retrievers( 1 / (k + rank_r(d)) )

A cross-encoder reranker then re-scores the fused shortlist by jointly attending
to (query, chunk), which is far more precise than the bi-encoder used for the
first-stage recall.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

from .schema import Chunk, RetrievedChunk, SourceType
from .config import RetrievalConfig

if TYPE_CHECKING:  # heavy deps only needed at construction time, not for logic
    from .embeddings import Embedder
    from .store import VectorStore


def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


def _payload_to_chunk(p: dict) -> Chunk:
    meta = {k: v for k, v in p.items()
            if k not in {"chunk_id", "doc_id", "source_type", "text"}}
    return Chunk(
        chunk_id=p["chunk_id"],
        doc_id=p["doc_id"],
        source_type=SourceType(p["source_type"]),
        text=p["text"],
        metadata=meta,
    )


class HybridRetriever:
    def __init__(self, cfg: RetrievalConfig, embedder: "Embedder", store: "VectorStore"):
        self.cfg = cfg
        self.embedder = embedder
        self.store = store
        self._bm25_corpus: list[dict] | None = None
        self._bm25: BM25Okapi | None = None

    # --- sparse index is built lazily from the vector store payloads -------- #
    def _ensure_bm25(self):
        if self._bm25 is None:
            self._bm25_corpus = self.store.all_payloads()
            tokenized = [_tokenize(p["text"]) for p in self._bm25_corpus]
            self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def _dense(self, query: str, source_types):
        qv = self.embedder.embed_query(query)
        hits = self.store.search(qv, self.cfg.dense_top_k, source_types=source_types)
        return [p["chunk_id"] for p, _ in hits], {p["chunk_id"]: p for p, _ in hits}

    def _sparse(self, query: str, source_types):
        self._ensure_bm25()
        if not self._bm25:
            return [], {}
        wanted = {s.value for s in source_types} if source_types else None
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ids, payloads = [], {}
        for i in ranked:
            payload = self._bm25_corpus[i]
            if wanted and payload["source_type"] not in wanted:
                continue  # honor the same filter the dense path applies
            ids.append(payload["chunk_id"])
            payloads[payload["chunk_id"]] = payload
            if len(ids) >= self.cfg.sparse_top_k:
                break
        return ids, payloads

    @staticmethod
    def _rrf(rank_lists: list[list[str]], k: int) -> dict[str, float]:
        fused: dict[str, float] = defaultdict(float)
        for ranks in rank_lists:
            for rank, cid in enumerate(ranks):
                fused[cid] += 1.0 / (k + rank)
        return fused

    def retrieve(self, query: str, source_types: list[SourceType] | None = None
                 ) -> list[RetrievedChunk]:
        dense_ids, dense_p = self._dense(query, source_types)
        sparse_ids, sparse_p = self._sparse(query, source_types)
        payloads = {**sparse_p, **dense_p}

        fused = self._rrf([dense_ids, sparse_ids], self.cfg.rrf_k)
        ranked_ids = sorted(fused, key=fused.get, reverse=True)

        candidates = [
            RetrievedChunk(
                chunk=_payload_to_chunk(payloads[cid]),
                score=fused[cid],
                provenance={"stage": "rrf",
                            "in_dense": cid in dense_ids,
                            "in_sparse": cid in sparse_ids},
            )
            for cid in ranked_ids if cid in payloads
        ]

        if self.cfg.use_reranker and candidates:
            candidates = self._rerank(query, candidates)
        return candidates[: self.cfg.rerank_top_n]

    def _rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        model = _load_reranker(self.cfg.reranker_model)
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = model.predict(pairs)
        for c, s in zip(candidates, scores):
            c.score = float(s)
            c.provenance["stage"] = "rerank"
        return sorted(candidates, key=lambda c: c.score, reverse=True)


@lru_cache(maxsize=1)
def _load_reranker(name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name)
