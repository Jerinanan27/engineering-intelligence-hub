"""Diagram captioning — the multimodal ingestion step.

Images can't be embedded into the same "meaning space" as text, so we don't try.
Instead we ask a vision model to *read* the diagram and write a structured text
description (components, connections, protocols). That description is then
embedded and retrieved by the exact same pipeline as docs and code — no new
embedding model, no extra memory on the host.

Two design choices that matter for a free-tier deployment:
  1. We use a HOSTED vision model (Groq Llama 4 Scout) via the key you already
     have. No heavy local vision model is loaded into the app's memory.
  2. Captions are CACHED next to the image as `<image>.caption.txt`. The app
     reads the cache at startup and never calls the vision API on boot. You
     (re)generate captions once, offline, with scripts/caption_diagrams.py.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

import requests

GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

CAPTION_PROMPT = (
    "You are documenting an engineering architecture diagram for a searchable "
    "knowledge base. Describe it as precise plain text so an engineer could find "
    "it by searching. Cover: (1) every component/box and its exact label; "
    "(2) every arrow/connection and what flows along it; (3) any protocols, "
    "ports, services, or technologies named. Do not invent details that are not "
    "visible. Write 4-8 sentences, no preamble."
)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
          ".webp": "image/webp", ".gif": "image/gif"}


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in _IMAGE_EXTS


def _encode(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    media = _MEDIA.get(path.suffix.lower(), "image/png")
    return f"data:{media};base64,{b64}"


def caption_image(path: str | Path, model: str = GROQ_VISION_MODEL) -> str:
    """Call the hosted vision model to describe one diagram. Needs GROQ_API_KEY."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set — cannot caption images.")
    path = Path(path)
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": CAPTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": _encode(path)}},
                ],
            }],
            "temperature": 0.1,
            "max_tokens": 400,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def cached_caption(path: str | Path, model: str = GROQ_VISION_MODEL) -> str | None:
    """Return the caption for an image, preferring the on-disk cache.

    Cache file is `<image path>.caption.txt`. If absent, try the vision API and
    write the cache. If the API is unavailable (no key / offline), return None so
    ingestion can skip the diagram gracefully instead of crashing.
    """
    path = Path(path)
    cache = path.with_suffix(path.suffix + ".caption.txt")
    if cache.exists():
        text = cache.read_text(encoding="utf-8").strip()
        if text:
            return text
    try:
        text = caption_image(path, model=model)
    except Exception:
        return None
    try:
        cache.write_text(text, encoding="utf-8")
    except Exception:
        pass
    return text
