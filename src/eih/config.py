"""Lightweight configuration: a YAML file mapped onto dataclasses.

Kept dependency-free on purpose (no pydantic-settings) so the repo stays small
and the config surface is obvious to a reviewer reading the code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import yaml


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"
    # bge models recommend a query prefix for retrieval; passages use none.
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    batch_size: int = 32


@dataclass
class StoreConfig:
    # ":memory:" runs Qdrant in-process (zero setup). For persistence,
    # set url to a running Qdrant (see docker-compose.yml).
    location: str = ":memory:"
    url: str | None = None
    collection: str = "eih_chunks"


@dataclass
class RetrievalConfig:
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60          # Reciprocal Rank Fusion constant
    rerank_top_n: int = 5    # final chunks passed to the LLM
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"


@dataclass
class LLMConfig:
    # Resolution order at runtime: GROQ_API_KEY -> Ollama -> echo fallback.
    provider: str = "auto"            # auto | ollama | groq | echo
    ollama_model: str = "qwen2.5:7b"
    ollama_url: str = "http://localhost:11434"
    groq_model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.1
    max_tokens: int = 800


@dataclass
class ChunkingConfig:
    doc_max_chars: int = 1200
    doc_overlap_chars: int = 150
    code_max_chars: int = 1600  # functions/classes can be large


@dataclass
class Config:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        if path is None:
            path = os.getenv("EIH_CONFIG", "config/config.yaml")
        p = Path(path)
        if not p.exists():
            return cls()  # sensible defaults
        raw = yaml.safe_load(p.read_text()) or {}
        return cls(
            embedding=EmbeddingConfig(**raw.get("embedding", {})),
            store=StoreConfig(**raw.get("store", {})),
            retrieval=RetrievalConfig(**raw.get("retrieval", {})),
            llm=LLMConfig(**raw.get("llm", {})),
            chunking=ChunkingConfig(**raw.get("chunking", {})),
        )
