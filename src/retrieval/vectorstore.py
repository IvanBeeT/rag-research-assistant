"""ChromaDB operations: index chunks and search by vector similarity.

ChromaDB stores three things per record:
  1. The embedding vector (for similarity search)
  2. The raw text (returned with results so we can pass it to the LLM)
  3. Metadata (paper title, page, chunk index — used for citations)

It persists everything to disk at CHROMA_PATH, so indexing only needs to
happen once per paper.
"""

import hashlib
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src import config
from src.retrieval.embedder import embed_texts


def _get_collection() -> chromadb.Collection:
    """Open (or create) the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(
        path=str(config.CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[dict]) -> None:
    """Embed and store a list of chunks in ChromaDB.

    Uses a content hash as the document ID so re-indexing the same paper
    is safe — duplicate IDs are silently skipped by ChromaDB's upsert.
    """
    if not chunks:
        return

    collection = _get_collection()

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [_chunk_id(c) for c in chunks]
    embeddings = embed_texts(texts)

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"  Indexed {len(chunks)} chunks into ChromaDB")


def search(query_embedding: list[float], top_k: int = None) -> list[dict]:
    """Return the top-k most similar chunks for a query embedding.

    Each result contains:
      - 'text': the chunk content
      - 'metadata': source paper, chunk index, etc.
      - 'distance': cosine distance (0 = identical, 2 = opposite)
    """
    top_k = top_k or config.TOP_K
    collection = _get_collection()

    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "text": text,
            "metadata": metadata,
            "distance": distance,
        })

    return output


def list_indexed_papers() -> list[str]:
    """Return the unique paper titles currently in the index."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    all_meta = collection.get(include=["metadatas"])["metadatas"]
    seen = set()
    titles = []
    for m in all_meta:
        title = m.get("source", "Unknown")
        if title not in seen:
            seen.add(title)
            titles.append(title)
    return sorted(titles)


def delete_paper(source_filename: str) -> int:
    """Remove all chunks belonging to a specific paper from the index."""
    collection = _get_collection()
    results = collection.get(where={"source": source_filename}, include=["metadatas"])
    ids_to_delete = results["ids"]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)


def _chunk_id(chunk: dict) -> str:
    """Stable ID based on source file + chunk index."""
    key = f"{chunk['metadata']['source']}::{chunk['metadata']['chunk_index']}"
    return hashlib.md5(key.encode()).hexdigest()
