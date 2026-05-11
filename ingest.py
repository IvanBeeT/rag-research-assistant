"""Index PDFs from the data/papers directory into ChromaDB.

Run this whenever you add new papers:
    python ingest.py

Only new/changed papers need to be re-indexed — existing chunks are skipped
via upsert (content-hash IDs).
"""

from src import config
from src.ingestion.pdf_loader import load_papers_dir
from src.ingestion.chunker import chunk_documents
from src.retrieval.vectorstore import add_chunks, list_indexed_papers


def main():
    print(f"Loading PDFs from: {config.PAPERS_DIR}")
    docs = load_papers_dir(config.PAPERS_DIR)

    if not docs:
        print("No documents to index. Add PDFs to data/papers/ and run again.")
        return

    print(f"\nChunking {len(docs)} documents...")
    chunks = chunk_documents(docs)

    print(f"\nIndexing {len(chunks)} total chunks...")
    add_chunks(chunks)

    print(f"\nDone. Papers in index:")
    for paper in list_indexed_papers():
        print(f"  - {paper}")


if __name__ == "__main__":
    main()
