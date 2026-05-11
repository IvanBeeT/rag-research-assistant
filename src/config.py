"""Central configuration. All tuneable parameters live here.

Copy .env.example to .env and override values there. Defaults are sensible
for an RTX 4080 with llama3.1:8b.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
ROOT_DIR = Path(__file__).parent.parent
PAPERS_DIR = ROOT_DIR / os.getenv("PAPERS_DIR", "data/papers")
CHROMA_PATH = ROOT_DIR / os.getenv("CHROMA_PATH", ".chromadb")

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Embeddings — runs locally via sentence-transformers
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cuda")

# Chunking strategy
# 800 chars ≈ 150-200 tokens. Overlap ensures context isn't lost at boundaries.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

# Retrieval: number of chunks returned per query
TOP_K = int(os.getenv("TOP_K", 5))

# ChromaDB collection name
COLLECTION_NAME = "research_papers"
