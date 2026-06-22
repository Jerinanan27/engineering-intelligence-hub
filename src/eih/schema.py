"""Core data structures shared across the pipeline.

A `Document` is a raw source item (a markdown file, a code file, an incident
report). A `Chunk` is a retrieval-sized unit derived from a Document, carrying
enough metadata to (a) filter at query time and (b) build a precise citation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import hashlib


class SourceType(str, Enum):
    """The four heterogeneous source types this system ingests."""
    DOC = "doc"            # technical documentation / READMEs
    CODE = "code"          # source code repositories
    INCIDENT = "incident"  # postmortems / incident reports
    DIAGRAM = "diagram"    # architecture diagrams (multimodal, via VLM caption)


@dataclass
class Document:
    """A raw ingested source item before chunking."""
    doc_id: str
    source_type: SourceType
    content: str
    # path / repo / title — anything useful for provenance & filtering
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrieval-sized unit with provenance for citation."""
    chunk_id: str
    doc_id: str
    source_type: SourceType
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make_id(doc_id: str, ordinal: int, text: str) -> str:
        """Deterministic id so re-ingesting identical content is idempotent."""
        h = hashlib.sha1(f"{doc_id}:{ordinal}:{text}".encode()).hexdigest()[:16]
        return f"{doc_id}::{ordinal}::{h}"


@dataclass
class RetrievedChunk:
    """A chunk plus the score it earned during retrieval/reranking."""
    chunk: Chunk
    score: float
    # where the score came from, useful for debugging the hybrid pipeline
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class Answer:
    """Final grounded answer returned to the user."""
    question: str
    text: str
    citations: list[RetrievedChunk] = field(default_factory=list)
