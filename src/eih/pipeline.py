"""Top-level orchestrator. This is the object the API/UI/CLI all talk to."""
from __future__ import annotations

from pathlib import Path

from .config import Config
from .schema import Answer, SourceType
from .embeddings import Embedder
from .store import VectorStore
from .retrieval import HybridRetriever
from .generation import LLM, Generator
from .ingestion import loaders


class EngineeringIntelligenceHub:
    def __init__(self, config: Config | None = None):
        self.cfg = config or Config.load()
        self.embedder = Embedder(self.cfg.embedding)
        self.store = VectorStore(self.cfg.store, dim=self.embedder.dim)
        self.retriever = HybridRetriever(self.cfg.retrieval, self.embedder, self.store)
        self.generator = Generator(LLM(self.cfg.llm))

    # --- ingestion --------------------------------------------------------- #
    def ingest(self, path: str | Path) -> dict:
        documents, chunks = loaders.ingest_directory(path, self.cfg.chunking)
        if not chunks:
            return {"documents": 0, "chunks": 0}
        vectors = self.embedder.embed_passages([c.text for c in chunks])
        self.store.upsert(chunks, vectors)
        self.retriever._bm25 = None  # invalidate sparse index after new data
        by_type: dict[str, int] = {}
        for c in chunks:
            by_type[c.source_type.value] = by_type.get(c.source_type.value, 0) + 1
        return {"documents": len(documents), "chunks": len(chunks), "by_type": by_type}

    # --- query ------------------------------------------------------------- #
    def ask(self, question: str, source_types: list[SourceType] | None = None) -> Answer:
        chunks = self.retriever.retrieve(question, source_types=source_types)
        return self.generator.answer(question, chunks)
