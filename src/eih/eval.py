"""A small, dependency-light evaluation harness.

For a portfolio project, *showing measured quality* separates you from the
tutorial crowd. This computes the metrics that matter for RAG, against a tiny
gold set you control (eval/goldset.json), without requiring the full RAGAS
stack (which you can graduate to later — see README roadmap).

Metrics
-------
- recall@k / hit@k : did the relevant doc make it into the top-k retrieval?
- mrr              : mean reciprocal rank of the first relevant doc.
- citation_rate    : fraction of answers that cite at least one source
                     (a cheap, LLM-free proxy for groundedness).

Each gold item: {"question": str, "relevant_doc_ids": [str, ...]}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .pipeline import EngineeringIntelligenceHub


_CITE_RE = re.compile(r"\[\d+\]")


def evaluate(hub: EngineeringIntelligenceHub, goldset_path: str | Path, k: int = 5) -> dict:
    gold = json.loads(Path(goldset_path).read_text())
    hits, rr, cited, n = 0, 0.0, 0, len(gold)

    for item in gold:
        q = item["question"]
        relevant = set(item["relevant_doc_ids"])
        retrieved = hub.retriever.retrieve(q)
        ranked_docs = [rc.chunk.doc_id for rc in retrieved][:k]

        if relevant & set(ranked_docs):
            hits += 1
        for rank, doc_id in enumerate(ranked_docs, 1):
            if doc_id in relevant:
                rr += 1.0 / rank
                break

        ans = hub.generator.answer(q, retrieved)
        if _CITE_RE.search(ans.text):
            cited += 1

    return {
        "n": n,
        f"hit@{k}": round(hits / n, 3) if n else 0.0,
        "mrr": round(rr / n, 3) if n else 0.0,
        "citation_rate": round(cited / n, 3) if n else 0.0,
    }
