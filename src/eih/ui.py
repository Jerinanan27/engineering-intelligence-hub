"""Streamlit demo UI. Run: streamlit run src/eih/ui.py

A deliberately minimal, editorial layout: ask a question, see the grounded
answer, and inspect the exact sources + retrieval scores behind it. The source
inspector is the part that signals engineering depth to a reviewer.
"""
from __future__ import annotations

import streamlit as st

from eih import EngineeringIntelligenceHub, SourceType

st.set_page_config(page_title="Engineering Intelligence Hub", layout="wide")


@st.cache_resource
def get_hub():
    hub = EngineeringIntelligenceHub()
    hub.ingest("data/sample")
    return hub


st.title("Engineering Intelligence Hub")
st.caption("Hybrid RAG over docs, code & incidents — grounded, cited answers.")

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
            st.code(c.chunk.text)
