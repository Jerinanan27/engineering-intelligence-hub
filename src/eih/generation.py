"""Generation: an LLM abstraction + a grounded answer builder.

Provider resolution (so the same code is free-local AND demo-deployable):
    GROQ_API_KEY set    -> Groq (free hosted API, for live demos)
    else Ollama running -> local model (fully offline, $0)
    else                -> echo provider (returns assembled context; lets you
                           demo retrieval without any LLM installed)

The prompt forces the model to answer ONLY from retrieved context and to cite
sources by bracketed index, so answers are grounded and auditable.
"""
from __future__ import annotations

import os
import requests

from .schema import Answer, RetrievedChunk
from .config import LLMConfig


SYSTEM_PROMPT = (
    "You are the Engineering Intelligence Hub assistant. Answer the engineer's "
    "question using ONLY the numbered context sources below. If the answer is not "
    "in the context, say so plainly. Cite every claim with its source number in "
    "square brackets, e.g. [2]. Be precise and concise; prefer concrete names, "
    "commands, and file paths over generalities."
)


def _build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, rc in enumerate(chunks, 1):
        src = rc.chunk.metadata.get("path") or rc.chunk.doc_id
        blocks.append(f"[{i}] (source: {src} | type: {rc.chunk.source_type.value})\n{rc.chunk.text}")
    return "\n\n".join(blocks)


def _build_prompt(question: str, context: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n### CONTEXT\n{context}\n\n### QUESTION\n{question}\n\n### ANSWER\n"


# --------------------------------------------------------------------------- #
class LLM:
    """Resolves to a concrete provider lazily."""
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.provider = self._resolve(cfg)

    def _resolve(self, cfg: LLMConfig) -> str:
        if cfg.provider != "auto":
            return cfg.provider
        if os.getenv("GROQ_API_KEY"):
            return "groq"
        try:
            requests.get(f"{cfg.ollama_url}/api/tags", timeout=1.5)
            return "ollama"
        except Exception:
            return "echo"

    def complete(self, prompt: str) -> str:
        return getattr(self, f"_{self.provider}")(prompt)

    def _ollama(self, prompt: str) -> str:
        r = requests.post(
            f"{self.cfg.ollama_url}/api/generate",
            json={"model": self.cfg.ollama_model, "prompt": prompt, "stream": False,
                  "options": {"temperature": self.cfg.temperature,
                              "num_predict": self.cfg.max_tokens}},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["response"].strip()

    def _groq(self, prompt: str) -> str:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
            json={"model": self.cfg.groq_model,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": self.cfg.temperature,
                  "max_tokens": self.cfg.max_tokens},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def _echo(self, prompt: str) -> str:
        return ("[no LLM configured — showing retrieved context only]\n"
                "Install Ollama or set GROQ_API_KEY to enable generation.\n"
                "The retrieved sources used to answer are listed below.")


class Generator:
    def __init__(self, llm: LLM):
        self.llm = llm

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> Answer:
        if not chunks:
            return Answer(question=question,
                          text="No relevant context found in the indexed sources.",
                          citations=[])
        context = _build_context(chunks)
        text = self.llm.complete(_build_prompt(question, context))
        return Answer(question=question, text=text, citations=chunks)
