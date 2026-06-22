from eih.config import ChunkingConfig
from eih.schema import Document, SourceType
from eih.ingestion import chunkers


def test_code_chunking_separates_functions():
    code = "import os\n\ndef a():\n    return 1\n\ndef b():\n    return 2\n"
    doc = Document("m", SourceType.CODE, code, {"path": "m.py"})
    chunks = chunkers.chunk(doc, ChunkingConfig())
    joined = "\n".join(c.text for c in chunks)
    assert "def a" in joined and "def b" in joined
    assert len(chunks) >= 2  # not one giant blob


def test_doc_chunking_keeps_headings():
    md = "# Title\n\nbody one\n\n## Sub\n\nbody two\n"
    doc = Document("d", SourceType.DOC, md, {"path": "d.md"})
    chunks = chunkers.chunk(doc, ChunkingConfig())
    assert any("# Title" in c.text for c in chunks)
