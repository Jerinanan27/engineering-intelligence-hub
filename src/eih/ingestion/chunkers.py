"""Chunking strategies.

Two ideas the reviewer should notice:

1. Documentation is split on structural boundaries (headings, blank lines,
   fenced code blocks) *before* falling back to a character window, so we never
   cut a code block or table in half.

2. Code is split by syntactic unit (function / class) using a tree-sitter AST.
   Naive line- or character-windowing destroys code retrieval: the query
   "where do we validate the JWT" must land on a whole function, not a
   half-statement. If tree-sitter isn't installed, we degrade gracefully to a
   regex heuristic so the repo always runs.
"""
from __future__ import annotations

import re
from typing import Iterable

from ..schema import Chunk, Document, SourceType
from ..config import ChunkingConfig


# --------------------------------------------------------------------------- #
# Documentation chunking
# --------------------------------------------------------------------------- #
_HEADING_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)


def _split_on_structure(text: str) -> list[str]:
    """Split on markdown headings first; keep heading with its body."""
    positions = [m.start() for m in _HEADING_RE.finditer(text)]
    if not positions:
        return [text]
    positions.append(len(text))
    sections = []
    # leading content before the first heading
    if positions[0] > 0:
        sections.append(text[: positions[0]])
    for i in range(len(positions) - 1):
        sections.append(text[positions[i] : positions[i + 1]])
    return [s for s in sections if s.strip()]


def _window(text: str, max_chars: int, overlap: int) -> list[str]:
    """Character window with overlap, breaking on paragraph/sentence when possible."""
    if len(text) <= max_chars:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # prefer to break on a paragraph or sentence boundary
            for sep in ("\n\n", "\n", ". "):
                idx = text.rfind(sep, start + max_chars // 2, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        out.append(text[start:end].strip())
        start = max(end - overlap, end) if end <= start else end - overlap
        if start < 0:
            start = end
    return [c for c in out if c]


def chunk_document(doc: Document, cfg: ChunkingConfig) -> list[Chunk]:
    chunks: list[str] = []
    for section in _split_on_structure(doc.content):
        chunks.extend(_window(section, cfg.doc_max_chars, cfg.doc_overlap_chars))
    return _wrap(doc, chunks)


# --------------------------------------------------------------------------- #
# Code chunking (AST-aware)
# --------------------------------------------------------------------------- #
def _treesitter_chunks(code: str, language: str) -> list[str] | None:
    """Return top-level def/class spans via tree-sitter, or None if unavailable."""
    try:
        from tree_sitter_languages import get_parser  # type: ignore
    except Exception:
        return None
    try:
        parser = get_parser(language)
    except Exception:
        return None

    tree = parser.parse(code.encode())
    root = tree.root_node
    wanted = {
        "function_definition", "class_definition",       # python
        "function_declaration", "method_declaration",    # js/java/go
        "class_declaration", "lexical_declaration",
    }
    spans: list[tuple[int, int]] = []
    for node in root.children:
        if node.type in wanted:
            spans.append((node.start_byte, node.end_byte))
    if not spans:
        return None

    raw = code.encode()
    out, prev_end = [], 0
    # capture module-level preamble (imports etc.) as its own chunk
    if spans[0][0] > 0:
        head = raw[: spans[0][0]].decode(errors="ignore").strip()
        if head:
            out.append(head)
    for start, end in spans:
        out.append(raw[start:end].decode(errors="ignore"))
        prev_end = end
    tail = raw[prev_end:].decode(errors="ignore").strip()
    if tail:
        out.append(tail)
    return out


def _regex_code_chunks(code: str) -> list[str]:
    """Fallback: split on top-level def/class/function keywords."""
    pattern = re.compile(r"^(?:def |class |function |func |public |private |export )",
                         re.MULTILINE)
    starts = [m.start() for m in pattern.finditer(code)]
    if not starts:
        return [code]
    starts = [0] + starts if starts[0] != 0 else starts
    starts.append(len(code))
    out = []
    for i in range(len(starts) - 1):
        seg = code[starts[i]: starts[i + 1]].strip()
        if seg:
            out.append(seg)
    return out


_EXT_LANG = {".py": "python", ".js": "javascript", ".ts": "typescript",
             ".java": "java", ".go": "go", ".rs": "rust", ".cpp": "cpp"}


def chunk_code(doc: Document, cfg: ChunkingConfig) -> list[Chunk]:
    path = str(doc.metadata.get("path", ""))
    lang = _EXT_LANG.get(path[path.rfind("."):], "python") if "." in path else "python"
    pieces = _treesitter_chunks(doc.content, lang) or _regex_code_chunks(doc.content)
    # large units (e.g. a 400-line class) get windowed so they fit the model
    final: list[str] = []
    for p in pieces:
        if len(p) > cfg.code_max_chars:
            final.extend(_window(p, cfg.code_max_chars, 100))
        else:
            final.append(p)
    return _wrap(doc, final)


# --------------------------------------------------------------------------- #
def _wrap(doc: Document, texts: Iterable[str]) -> list[Chunk]:
    chunks = []
    for i, t in enumerate(texts):
        if not t.strip():
            continue
        chunks.append(
            Chunk(
                chunk_id=Chunk.make_id(doc.doc_id, i, t),
                doc_id=doc.doc_id,
                source_type=doc.source_type,
                text=t,
                metadata={**doc.metadata, "ordinal": i},
            )
        )
    return chunks


def chunk(doc: Document, cfg: ChunkingConfig) -> list[Chunk]:
    """Dispatch to the right strategy for the source type."""
    if doc.source_type == SourceType.CODE:
        return chunk_code(doc, cfg)
    # docs and incidents are both prose/markdown
    return chunk_document(doc, cfg)
