"""Qdrant vector store wrapper.

Defaults to in-process mode (`location=":memory:"`) so the project runs with no
external services. Point `store.url` at a running Qdrant (docker-compose.yml)
for persistence. Chunk metadata is stored as the point payload, which is what
lets us do filtered retrieval (by source_type, severity, service, ...).
"""
from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .schema import Chunk, SourceType
from .config import StoreConfig


def _stable_uuid(chunk_id: str) -> str:
    # Qdrant point ids must be int or UUID; derive a deterministic UUID.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class VectorStore:
    def __init__(self, cfg: StoreConfig, dim: int):
        self.cfg = cfg
        self.dim = dim
        if cfg.url:
            self.client = QdrantClient(url=cfg.url)
        else:
            self.client = QdrantClient(location=cfg.location)
        self._ensure_collection()

    def _ensure_collection(self):
        existing = {c.name for c in self.client.get_collections().collections}
        if self.cfg.collection not in existing:
            self.client.create_collection(
                collection_name=self.cfg.collection,
                vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
            )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]):
        points = [
            qm.PointStruct(
                id=_stable_uuid(c.chunk_id),
                vector=v,
                payload={
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "source_type": c.source_type.value,
                    "text": c.text,
                    **c.metadata,
                },
            )
            for c, v in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.cfg.collection, points=points)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        source_types: list[SourceType] | None = None,
        payload_filter: dict[str, Any] | None = None,
    ):
        conditions = []
        if source_types:
            conditions.append(
                qm.FieldCondition(
                    key="source_type",
                    match=qm.MatchAny(any=[s.value for s in source_types]),
                )
            )
        for key, val in (payload_filter or {}).items():
            conditions.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=val)))
        flt = qm.Filter(must=conditions) if conditions else None

        hits = self.client.query_points(
            collection_name=self.cfg.collection,
            query=query_vector,
            limit=top_k,
            query_filter=flt,
            with_payload=True,
        ).points
        return [(h.payload, float(h.score)) for h in hits]

    def all_payloads(self) -> list[dict[str, Any]]:
        """Scroll the full corpus — used to build the in-process BM25 index."""
        out, offset = [], None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.cfg.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            out.extend(p.payload for p in points)
            if offset is None:
                break
        return out
