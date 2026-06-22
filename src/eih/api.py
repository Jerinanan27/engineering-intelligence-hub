"""FastAPI service. Run: uvicorn eih.api:app --reload

Endpoints
---------
POST /ingest  {"path": "data/sample"}        -> ingestion stats
POST /ask     {"question": "...", "source_types": ["doc","code"]}  -> answer + citations
GET  /healthz                                -> liveness probe (used by k8s)
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .pipeline import EngineeringIntelligenceHub
from .schema import SourceType

app = FastAPI(title="Engineering Intelligence Hub", version="0.1.0")
hub = EngineeringIntelligenceHub()


class IngestRequest(BaseModel):
    path: str = "data/sample"


class AskRequest(BaseModel):
    question: str
    source_types: list[str] | None = None


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest):
    return hub.ingest(req.path)


@app.post("/ask")
def ask(req: AskRequest):
    types = [SourceType(t) for t in req.source_types] if req.source_types else None
    ans = hub.ask(req.question, source_types=types)
    return {
        "question": ans.question,
        "answer": ans.text,
        "citations": [
            {
                "index": i,
                "doc_id": c.chunk.doc_id,
                "source_type": c.chunk.source_type.value,
                "path": c.chunk.metadata.get("path"),
                "score": round(c.score, 4),
                "preview": c.chunk.text[:240],
            }
            for i, c in enumerate(ans.citations, 1)
        ],
    }
