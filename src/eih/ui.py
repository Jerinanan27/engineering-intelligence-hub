"""Streamlit demo UI. Run: streamlit run src/eih/ui.py

A deliberately minimal, editorial layout: ask a question, see the grounded
answer, and inspect the exact sources + retrieval scores behind it. When a
matched source is an architecture diagram, the original image is shown inline.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

# Bridge Streamlit's secret store into the environment the LLM/vision code reads.
if "GROQ_API_KEY" in st.secrets and not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from eih import EngineeringIntelligenceHub, SourceType

st.set_page_config(page_title="Engineering Intelligence Hub", layout="wide")


@st.cache_resource
def get_hub():
    hub = EngineeringIntelligenceHub()
    hub.ingest("data/sample")
    return hub


st.title("Engineering Intelligence Hub")
st.caption("Hybrid RAG over docs, code, incidents & architecture diagrams — grounded, cited answers.")

hub = get_hub()

with st.sidebar:
    st.subheader("Filter sources")
    chosen = st.multiselect(
        "Restrict retrieval to:",
        options=[s.value for s in SourceType],
        default=[],
        help="Leave empty to search everything.",
    )
    st.caption(f"LLM provider: **{hub.generator.llm.provider}**")

question = st.text_input("Ask an engineering question",
                         placeholder="How does token validation work, and what broke in INC-2025-014?")

if question:
    types = [SourceType(t) for t in chosen] if chosen else None
    with st.spinner("Retrieving and reasoning…"):
        ans = hub.ask(question, source_types=types)

    st.markdown("### Answer")
    st.write(ans.text)

    st.markdown("### Sources")
    for i, c in enumerate(ans.citations, 1):
        src = c.chunk.metadata.get("path", c.chunk.doc_id)
        with st.expander(f"[{i}] {src}  ·  {c.chunk.source_type.value}  ·  score {c.score:.3f}"):
            # For diagrams, show the actual image alongside its captioned text.
            if c.chunk.source_type == SourceType.DIAGRAM:
                img_path = c.chunk.metadata.get("path")
                if img_path and Path(img_path).exists():
                    st.image(img_path, caption="Matched architecture diagram")
                st.caption("Vision-model description used for retrieval:")
            st.code(c.chunk.text)
