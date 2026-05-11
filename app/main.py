"""Streamlit entry point.

Run with:
    streamlit run app/main.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of where Streamlit is invoked from
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from app.components.sidebar import render_sidebar
from app.components.chat import render_chat


def main() -> None:
    st.set_page_config(
        page_title="Research Assistant",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_styles()
    render_sidebar()

    st.title("Research Assistant")
    st.caption("Ask questions about your indexed ML/AI research papers.")
    render_chat()


def _inject_styles() -> None:
    """Minimal CSS overrides to clean up Streamlit's default appearance."""
    st.markdown(
        """
        <style>
        /* Tighten the header */
        .block-container { padding-top: 2rem; }

        /* Style the chat input */
        .stChatInput textarea { font-size: 0.95rem; }

        /* Subtle citation expander */
        .streamlit-expanderHeader {
            font-size: 0.85rem;
            color: #888;
        }

        /* Hide Streamlit's default footer */
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
