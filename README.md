# Engineering Intelligence Hub

A developer-focused **Retrieval-Augmented Generation (RAG)** system that ingests
heterogeneous engineering knowledge — technical docs, source code, incident
reports (and architecture diagrams, on the roadmap) — and answers questions with
**grounded, cited** responses. The goal is to accelerate onboarding and cut
troubleshooting time by letting engineers ask natural-language questions across
all of a team's scattered context.

Runs **fully local and free** (Ollama + open-source models), with a one-line swap
to a free hosted API for a public demo. No paid services required.

---

## Why this is not a toy RAG

Most "RAG" demos chunk text on a fixed character window, embed with one model,
do cosine top-k, and stuff it into a prompt. That falls apart on engineering
content. This system addresses the specific failure modes:

| Problem with naive RAG | What this does |
|---|---|
| Character-windowing shreds code mid-function | **AST-aware code chunking** (tree-sitter) splits by function/class, with a regex fallback so it always runs |
| Semantic search misses exact identifiers (`auth-svc`, error codes, `RS256`) | **Hybrid retrieval**: dense + BM25 lexical, fused with **Reciprocal Rank Fusion** |
| Bi-encoder recall is imprecise | **Cross-encoder reranking** re-scores the shortlist by jointly reading (query, chunk) |
| Answers hallucinate / can't be audited | Prompt forces **answer-from-context-only** with **inline `[n]` citations**; UI shows the exact source chunks and scores |
| "It works on my 5 examples" | A small **evaluation harness** (hit@k, MRR, citation rate) against a gold set |

**Architecture diagrams** (images) are supported via **vision captioning**: a hosted
vision model (Groq Llama 4 Scout) reads each diagram and writes a structured text
description, which is embedded and retrieved by the same pipeline as text. No
heavy vision model runs in the app — captions are generated once, offline, and
cached on disk (`<image>.caption.txt`), so deployment stays free-tier-friendly.

A deliberate design decision worth calling out: **a single unified embedder is
used for all source types.** Mixing embedders with different dimensions/vector
spaces in one collection silently breaks similarity search. A per-modality
embedder design (separate collections, cross-collection fusion) is a documented
roadmap item rather than a hidden bug.

---

## Architecture

```
                 ┌─────────────┐   docs / code / incidents / diagrams
   sources  ───▶ │  Ingestion  │   • structure-aware doc chunking
                 │             │   • AST code chunking (tree-sitter)
                 └──────┬──────┘   • metadata extraction (severity, services)
                        ▼
                 ┌─────────────┐   bge-small-en-v1.5 (normalized)
                 │  Embedding  │
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐   Qdrant (in-memory or server)
                 │ Vector store│   payload = text + metadata (filterable)
                 └──────┬──────┘
                        ▼
   query  ──▶   ┌─────────────────────────────────────────┐
                │  Hybrid retrieval                        │
                │   dense (Qdrant)  +  sparse (BM25)       │
                │            └── RRF fusion ──┐            │
                │   cross-encoder reranker ◀──┘            │
                └──────────────────┬──────────────────────┘
                                   ▼
                 ┌─────────────┐   Ollama (local)  |  Groq (free demo)
                 │ Generation  │   grounded prompt + [n] citations
                 └─────────────┘
```

Interfaces: **CLI**, **FastAPI** service, **Streamlit** UI.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                      # exposes `import eih`

# Ask a question (in-memory store; ingests the sample corpus first)
python scripts/query.py "How is token validation done, and what broke in INC-2025-014?"
```

First run downloads the embedding + reranker models (~150 MB total). With no LLM
installed, the system still runs and returns the retrieved sources (the `echo`
provider) so you can verify retrieval before wiring up generation.

### Enabling generation (pick one, both free)

```bash
# Option A — fully local
#   install Ollama (https://ollama.com), then:
ollama pull qwen2.5:7b

# Option B — free hosted API (good for a public demo link)
export GROQ_API_KEY=...              # console.groq.com, free tier
```

Provider resolution is automatic: `GROQ_API_KEY` → Ollama → echo.

### Run the API / UI

```bash
uvicorn eih.api:app --reload         # http://localhost:8000/docs
streamlit run src/eih/ui.py          # http://localhost:8501
```

### Persistent vector store (optional)

```bash
docker compose up -d                 # starts Qdrant on :6333
# then set store.url: http://localhost:6333 in config/config.yaml
```

---

## Evaluation

```python
from eih import EngineeringIntelligenceHub
from eih.eval import evaluate

hub = EngineeringIntelligenceHub()
hub.ingest("data/sample")
print(evaluate(hub, "eval/goldset.json", k=5))
# -> {'n': 4, 'hit@5': ..., 'mrr': ..., 'citation_rate': ...}
```

Edit `eval/goldset.json` to grow the gold set. The harness is intentionally
dependency-light; it graduates cleanly to [RAGAS](https://github.com/explodinggradients/ragas)
(faithfulness, context precision/recall) when you want LLM-judged metrics.

---

## Project layout

```
src/eih/
  schema.py        # Document / Chunk / RetrievedChunk / Answer
  config.py        # YAML-backed dataclass config
  ingestion/
    chunkers.py    # structure-aware doc + AST code chunking
    loaders.py     # per-type loaders + metadata extraction
    vision.py      # diagram captioning (Groq Llama 4 Scout) + caption cache
  embeddings.py    # sentence-transformers wrapper
  store.py         # Qdrant wrapper (in-memory or server)
  retrieval.py     # dense + BM25 + RRF + cross-encoder rerank
  generation.py    # LLM abstraction (Ollama/Groq/echo) + grounded prompt
  eval.py          # hit@k / MRR / citation-rate harness
  pipeline.py      # top-level orchestrator
  api.py           # FastAPI
  ui.py            # Streamlit demo
```

## Adding your own architecture diagrams

1. Drop image files (`.png` / `.jpg`) into `data/sample/diagrams/`.
2. Generate captions once, offline, with your Groq key:
   ```bash
   export GROQ_API_KEY=...
   python scripts/caption_diagrams.py data/sample/diagrams
   ```
   This writes a `<image>.caption.txt` next to each image.
3. Commit the images **and** their `.caption.txt` files. The deployed app reads
   the cached captions and never calls the vision API at startup.

## Roadmap

- **Per-modality embedders** — multi-collection design with cross-collection
  fusion, so code can use a code-specialized embedder.
- **SVG / Mermaid diagrams as text** — read vector-diagram source directly
  (labels are already text) instead of captioning a rasterized image.
- **Repo ingestion at scale** — git-aware loader, language detection, incremental
  re-index on commit.
- **RAGAS metrics** + a small labeled benchmark.
- **Query routing** — classify a question and route to the right source subset.

## License

MIT.
