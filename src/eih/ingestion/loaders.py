"""Loaders turn files on disk into `Document`s, attaching provenance metadata.

Incident reports are parsed for light structured metadata (severity, affected
services) because those fields make excellent payload filters at query time
("show me high-severity incidents touching the auth service").
"""
from __future__ import annotations

import re
from pathlib import Path

from ..schema import Document, SourceType
from ..config import ChunkingConfig
from . import chunkers


_DOC_EXTS = {".md", ".markdown", ".rst", ".txt"}
_CODE_EXTS = {".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c"}


def load_doc(path: Path) -> Document:
    return Document(
        doc_id=path.stem,
        source_type=SourceType.DOC,
        content=path.read_text(encoding="utf-8", errors="ignore"),
        metadata={"path": str(path), "title": path.stem, "source": "docs"},
    )


def load_code(path: Path) -> Document:
    return Document(
        doc_id=path.stem,
        source_type=SourceType.CODE,
        content=path.read_text(encoding="utf-8", errors="ignore"),
        metadata={"path": str(path), "language": path.suffix.lstrip("."), "source": "repo"},
    )


_SEVERITY_RE = re.compile(r"severity\s*[:=]\s*(\w+)", re.IGNORECASE)
_SERVICES_RE = re.compile(r"(?:affected\s+services?|services?)\s*[:=]\s*(.+)", re.IGNORECASE)


def load_incident(path: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="ignore")
    sev = _SEVERITY_RE.search(text)
    svc = _SERVICES_RE.search(text)
    services = []
    if svc:
        services = [s.strip() for s in re.split(r"[,/]", svc.group(1)) if s.strip()]
    return Document(
        doc_id=path.stem,
        source_type=SourceType.INCIDENT,
        content=text,
        metadata={
            "path": str(path),
            "source": "incidents",
            "severity": sev.group(1).lower() if sev else "unknown",
            "services": services,
        },
    )


def load_directory(root: str | Path) -> list[Document]:
    """Walk a directory tree and classify files by location/extension."""
    root = Path(root)
    docs: list[Document] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        parent = p.parent.name.lower()
        if "incident" in parent:
            docs.append(load_incident(p))
        elif p.suffix in _CODE_EXTS:
            docs.append(load_code(p))
        elif p.suffix in _DOC_EXTS:
            docs.append(load_doc(p))
    return docs


def ingest_directory(root: str | Path, cfg: ChunkingConfig):
    """Load + chunk a directory, returning all chunks ready for embedding."""
    documents = load_directory(root)
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunkers.chunk(doc, cfg))
    return documents, all_chunks
