"""Wrap sentence-transformers to produce text embeddings.

An embedding is a fixed-size list of floats (a vector) that represents the
semantic meaning of a piece of text. Similar texts produce similar vectors.
The embedding model is completely separate from the LLM — it's a smaller,
encoder-only model whose only job is this conversion.

BAAI/bge-base-en-v1.5 is a 109M parameter model that consistently ranks
near the top of the MTEB benchmark for retrieval tasks on English text.
"""

from functools import lru_cache
from sentence_transformers import SentenceTransformer
from src import config


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the embedding model once and cache it for the process lifetime."""
    print(f"Loading embedding model: {config.EMBED_MODEL}")
    return SentenceTransformer(config.EMBED_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert a list of strings to a list of embedding vectors.

    BGE models perform better when queries are prefixed with a task instruction.
    For encoding documents (at index time) the prefix is omitted.
    """
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string with the BGE instruction prefix."""
    model = _get_model()
    # BGE-specific: prepend this prefix to queries (not to indexed documents)
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    embedding = model.encode(prefixed, normalize_embeddings=True)
    return embedding.tolist()
