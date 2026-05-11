"""Chat panel: message history, input box, streaming responses, citations.

Streamlit re-runs the entire script on every interaction, so conversation
history is kept in st.session_state — a dict that persists across reruns
within the same browser session.

Each assistant message stores both the response text and the source chunks
used to generate it, so we can render citations alongside the answer.
"""

import streamlit as st

from src.retrieval.retriever import retrieve, format_context
from src.retrieval.vectorstore import list_indexed_papers
from src.generation.llm import query_stream


def render_chat() -> None:
    _init_state()
    _render_history()
    _render_input()


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _render_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_citations(msg["sources"])


def _render_input() -> None:
    if not list_indexed_papers():
        st.info("Upload research papers using the sidebar to get started.")
        return

    if prompt := st.chat_input("Ask a question about the research papers..."):
        # Add user message to history and render it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve relevant chunks
        with st.spinner("Searching papers..."):
            chunks = retrieve(prompt)

        if not chunks:
            with st.chat_message("assistant"):
                response = "No relevant passages found in the indexed papers for that question."
                st.markdown(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": [],
            })
            return

        context = format_context(chunks)

        # Stream the LLM response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            for token in query_stream(prompt, context):
                full_response += token
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)
            _render_citations(chunks)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": chunks,
        })


def _render_citations(chunks: list[dict]) -> None:
    """Render source paper references in a collapsed expander."""
    if not chunks:
        return

    # Deduplicate by source filename
    seen = set()
    unique_sources = []
    for chunk in chunks:
        src = chunk["metadata"].get("source", "Unknown")
        if src not in seen:
            seen.add(src)
            unique_sources.append(chunk)

    with st.expander(f"Sources ({len(unique_sources)} paper{'s' if len(unique_sources) != 1 else ''})"):
        for i, chunk in enumerate(unique_sources, start=1):
            meta = chunk["metadata"]
            title = meta.get("title", meta.get("source", "Unknown"))
            source = meta.get("source", "")
            relevance = 1 - chunk.get("distance", 0)

            st.markdown(f"**[{i}] {title}**")
            if source != title:
                st.caption(source)
            st.caption(f"Relevance: {relevance:.0%}")
            st.markdown(f"> {chunk['text'][:300]}{'...' if len(chunk['text']) > 300 else ''}")
            if i < len(unique_sources):
                st.divider()
