"""Split documents into overlapping chunks for embedding.

Why chunk at all? Embedding models have a token limit (typically 512 tokens).
A 15-page paper has far more text than fits in one embedding. We split it into
smaller pieces so each piece can be meaningfully encoded as a single vector.

Why overlap? If a key sentence sits at a chunk boundary, splitting there would
lose its context. Overlapping windows ensure every sentence appears in full
in at least one chunk.

Chunk size of 800 chars ≈ 150-200 tokens. Small enough for the embedding model,
large enough to contain a complete thought or paragraph.
"""

from src.ingestion.pdf_loader import Document
from src import config


def chunk_document(doc: Document, chunk_size: int = None, overlap: int = None) -> list[dict]:
    """Split a Document into overlapping text chunks.

    Returns a list of dicts, each containing:
      - 'text': the chunk content
      - 'metadata': inherited from the source document, plus chunk_index
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    text = doc.text
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size

        # Prefer to break at a sentence boundary within the last 15% of the window
        if end < len(text):
            boundary = _find_sentence_boundary(text, end, lookback=int(chunk_size * 0.15))
            if boundary:
                end = boundary

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **doc.metadata,
                    "chunk_index": chunk_index,
                }
            })
            chunk_index += 1

        # Advance by (chunk_size - overlap) so the next chunk shares some context
        start += chunk_size - overlap

    return chunks


def chunk_documents(docs: list[Document]) -> list[dict]:
    """Chunk a list of documents and return all chunks flat."""
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  Chunked '{doc.metadata['title']}' → {len(chunks)} chunks")
    return all_chunks


def _find_sentence_boundary(text: str, pos: int, lookback: int) -> int | None:
    """Find the position of the last sentence-ending punctuation before pos."""
    window = text[max(0, pos - lookback):pos]
    for i in range(len(window) - 1, -1, -1):
        if window[i] in ".!?" and (i + 1 >= len(window) or window[i + 1] == " "):
            return pos - lookback + i + 1
    return None
