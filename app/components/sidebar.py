"""Sidebar: document management panel.

Shows which papers are indexed, lets users upload new PDFs, and
provides a delete button per paper. All indexing runs synchronously
inside the Streamlit process — fine for a local tool.
"""

import tempfile
from pathlib import Path

import streamlit as st

from src import config
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_document
from src.retrieval.vectorstore import add_chunks, list_indexed_papers, delete_paper
from src.generation.llm import check_ollama_connection


def render_sidebar() -> None:
    with st.sidebar:
        st.title("Research Papers")

        _render_connection_status()
        st.divider()
        _render_upload_section()
        st.divider()
        _render_paper_list()


def _render_connection_status() -> None:
    ok, message = check_ollama_connection()
    if ok:
        st.success(message, icon="✅")
    else:
        st.error(message, icon="🚫")


def _render_upload_section() -> None:
    st.subheader("Add Paper")
    uploaded = st.file_uploader(
        "Upload a PDF",
        type="pdf",
        help="Drag and drop or click to select. Paper is indexed immediately.",
        label_visibility="collapsed",
    )

    if uploaded is not None:
        # Check not already indexed
        indexed = list_indexed_papers()
        if uploaded.name in indexed:
            st.warning(f"{uploaded.name} is already indexed.")
            return

        with st.spinner(f"Indexing {uploaded.name}..."):
            try:
                # Write to a temp file so pdfplumber can open it by path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = Path(tmp.name)

                doc = load_pdf(tmp_path)
                doc.metadata["source"] = uploaded.name
                doc.metadata["title"] = Path(uploaded.name).stem.replace("_", " ").replace("-", " ").title()

                chunks = chunk_document(doc)
                add_chunks(chunks)
                tmp_path.unlink(missing_ok=True)

                st.success(f"Indexed {len(chunks)} chunks from {uploaded.name}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to index: {e}")


def _render_paper_list() -> None:
    st.subheader("Indexed Papers")
    papers = list_indexed_papers()

    if not papers:
        st.caption("No papers indexed yet. Upload a PDF above.")
        return

    st.caption(f"{len(papers)} paper{'s' if len(papers) != 1 else ''} in knowledge base")

    for paper in papers:
        col1, col2 = st.columns([4, 1])
        with col1:
            # Truncate long filenames for display
            display = paper if len(paper) <= 35 else paper[:32] + "..."
            st.text(display)
        with col2:
            if st.button("✕", key=f"del_{paper}", help=f"Remove {paper}"):
                n = delete_paper(paper)
                st.toast(f"Removed {paper} ({n} chunks)")
                st.rerun()
