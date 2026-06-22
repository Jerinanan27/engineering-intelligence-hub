"""Embedding model wrapper.

Design note (deliberate, documented in README): a single unified text embedder
is used for *all* source types. Mixing embedders with different output
dimensions / vector spaces in one collection silently breaks similarity search.
A multi-collection design with per-modality embedders is a roadmap item.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from .config import EmbeddingConfig


class Embedder:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self._model = _load_model(cfg.model_name)

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(list(texts))

    def embed_query(self, text: str) -> list[float]:
        # bge retrieval models expect an instruction prefix on the query side
        return self._encode([self.cfg.query_prefix + text])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=True,   # cosine == dot product after this
            show_progress_bar=False,
        )
        return vecs.tolist()


@lru_cache(maxsize=2)
def _load_model(name: str):
    # imported lazily so `import eih` doesn't pull torch unless needed
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)
