"""High-level retrieval: turn a question into ranked context chunks.

This is the "R" in RAG. Given a natural language query, it:
  1. Embeds the query into a vector
  2. Finds the closest matching chunks in ChromaDB
  3. Returns those chunks as context for the LLM to reason over

Keeping this as its own module makes it easy to later add re-ranking,
hybrid search (vector + keyword), or query expansion.
"""

from src.retrieval.embedder import embed_query
from src.retrieval.vectorstore import search
from src import config


def retrieve(query: str, top_k: int = None) -> list[dict]:
    """Return the most relevant chunks for a query, sorted by relevance."""
    top_k = top_k or config.TOP_K
    query_vec = embed_query(query)
    results = search(query_vec, top_k=top_k)
    return results


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a prompt-ready context block.

    Each chunk is labeled with its source so the LLM can attribute answers.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk["metadata"].get("source", "Unknown")
        title = chunk["metadata"].get("title", source)
        parts.append(f"[Source {i}: {title}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)
