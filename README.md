# Research Assistant

A local RAG (Retrieval-Augmented Generation) chatbot for querying ML and AI research papers. Upload PDFs, ask natural language questions, and get answers grounded in the actual content of the papers — with citations showing exactly which paper and passage each answer came from.

Everything runs on your machine. No API keys, no external services, no data leaving your hardware.

![Research Assistant UI](docs/Screenshot.png)

---

## What it does and why it works

When you upload a research paper, the system doesn't just store the text — it converts every passage into a mathematical representation of its meaning, called an embedding vector, and stores those vectors in a local database. When you ask a question, your question gets converted into the same kind of vector, and the system finds the passages whose meaning is closest to your question using cosine similarity. Those passages are then handed to a local language model, which reads them and writes a focused answer.

This approach is called RAG because the model *retrieves* relevant information before *generating* its answer, rather than relying on what it memorized during training. The result is an assistant that can answer precise questions about documents it has never seen before, with verifiable sources. If the papers don't contain the answer, it says so.

---

## Architecture

```
PDF upload
    │
    ▼
pdf_loader.py       Extracts text with pdfplumber, handles multi-column layouts
    │
    ▼
chunker.py          Splits text into 800-character overlapping windows
    │                 (overlap prevents losing context at boundaries)
    ▼
embedder.py         Converts each chunk to a 768-dim vector via BAAI/bge-base-en-v1.5
    │
    ▼
vectorstore.py      Stores vectors + text + metadata in ChromaDB (persisted to disk)
    │
    ▼  (at query time)
retriever.py        Embeds the question, retrieves top-5 chunks by cosine similarity
    │
    ▼
llm.py              Passes retrieved context + question to llama3.1:8b via Ollama
    │
    ▼
Streamlit UI        Streams the response token-by-token, renders source citations
```

A few decisions worth explaining:

**No LangChain.** The pipeline is built directly on ChromaDB, sentence-transformers, and the Ollama Python client. This keeps each component inspectable, makes the data flow obvious, and means there are no framework abstractions to debug when something goes wrong. The entire retrieval pipeline is around 200 lines of Python.

**Separate embedding model and LLM.** The embedding model (BAAI/bge-base-en-v1.5, 109M parameters) converts text to vectors — a fundamentally different task from generating text. Keeping them separate means either can be swapped independently without touching the other.

**ChromaDB over FAISS.** FAISS is a fast similarity index but stores only vectors, with no metadata support and no persistence out of the box. ChromaDB stores metadata (paper title, chunk index) alongside vectors, persists to disk automatically, and supports filtered queries — all of which are needed for citations and document management.

---

## Stack

| Component | Technology |
|---|---|
| Language model | llama3.1:8b via [Ollama](https://ollama.com) |
| Embedding model | BAAI/bge-base-en-v1.5 via sentence-transformers |
| Vector database | ChromaDB |
| PDF parsing | pdfplumber |
| UI | Streamlit |
| Container | Docker + Docker Compose |

---

## Setup

### Requirements

- Python 3.11 or higher
- [Ollama](https://ollama.com/download) installed and running
- 6 GB of free disk space (model weights)
- 8 GB RAM minimum, 16 GB recommended

### Native install (Windows, macOS, Linux)

```bash
# Clone the repository
git clone https://github.com/IvanBeeT/rag-research-assistant.git
cd rag-research-assistant

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Pull the language model (one-time download, ~4.9 GB)
ollama pull llama3.1:8b

# Copy the example config (optional — defaults work out of the box)
cp .env.example .env
```

Then start the app:

```bash
streamlit run app/main.py
```

Open `http://localhost:8501` in your browser. Use the sidebar to upload PDF papers, then ask questions in the chat window. The embedding model (~440 MB) downloads automatically on first run.

### Docker

The Docker setup runs the Streamlit app and Ollama as separate containers. This is the recommended way to run the project on a server or share it with others.

```bash
# Build and start both services
docker compose up --build

# In a separate terminal, pull the language model into the Ollama container
docker compose exec ollama ollama pull llama3.1:8b
```

Open `http://localhost:8501`.

GPU acceleration inside Docker requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on Linux. Uncomment the `deploy` block in `docker-compose.yml` to enable it. On Windows and macOS, Ollama runs natively and uses the GPU automatically — only the app container is needed.

---

## Configuration

Copy `.env.example` to `.env` and edit as needed. All settings have sensible defaults.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.1:8b` | Any model you've pulled with `ollama pull` |
| `EMBED_MODEL` | `BAAI/bge-base-en-v1.5` | Any model from sentence-transformers |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Passages retrieved per query |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |

To use a larger, more capable model (if your hardware supports it):

```bash
ollama pull qwen2.5:14b
# then set OLLAMA_MODEL=qwen2.5:14b in .env
```

### Indexing papers from the command line

If you prefer to index a batch of papers without using the upload UI, drop PDFs into `data/papers/` and run:

```bash
python ingest.py
```

---

## Extending the project

The codebase is structured so that new capabilities can be added without rewriting existing components.

**Swap the language model** by changing `OLLAMA_MODEL` in `.env` to any model available through Ollama. The prompt template in `src/generation/llm.py` may need tuning for models with different instruction formats.

**Add multiple paper collections** by exposing a `collection_name` parameter in `vectorstore.py` and adding a collection selector to the sidebar. ChromaDB supports multiple named collections in the same database.

**Persist conversation history** by accumulating the message list in `llm.py` and passing prior turns back on each request. The current implementation is stateless by design to keep responses focused on the retrieved context.

**Improve retrieval quality** with hybrid search — combining vector similarity with BM25 keyword matching in `retriever.py` and merging the ranked results. This helps with queries that contain rare terms or proper nouns that embeddings handle poorly.

**Add a re-ranking pass** using a cross-encoder model after the initial retrieval step. Cross-encoders score query-document pairs jointly and are significantly more accurate than bi-encoders, at the cost of being too slow for full-corpus search. Running one over the top-20 retrieved chunks to select the best 5 is a common pattern.

---

## Acknowledgements

Built with assistance from [Claude Code](https://claude.ai/code) (Anthropic).
