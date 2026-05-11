"""Ollama LLM interface with streaming support.

Ollama runs LLMs locally and exposes a simple HTTP API. The `ollama` Python
library wraps this. We use streaming so the UI can show tokens as they arrive
rather than waiting for the full response — this makes the app feel fast even
when the model is generating a long answer.

The system prompt is critical in RAG: it tells the LLM to stick to the
provided context and cite sources rather than hallucinating from its training.
"""

from collections.abc import Generator

import ollama

from src import config

_SYSTEM_PROMPT = """You are a research assistant specializing in machine learning and AI.
Answer questions using ONLY the provided context from research papers.
For each claim you make, cite the source paper using its label (e.g., [Source 1]).
If the context does not contain enough information to answer the question, say so clearly.
Do not fabricate information or draw on knowledge outside the provided context.
Keep answers concise but complete."""


def query_stream(user_message: str, context: str) -> Generator[str, None, None]:
    """Stream an answer from the LLM given a user question and retrieved context.

    Yields string tokens as they arrive so the UI can render incrementally.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context from research papers:\n\n{context}\n\nQuestion: {user_message}",
        },
    ]

    stream = ollama.chat(
        model=config.OLLAMA_MODEL,
        messages=messages,
        stream=True,
        options={
            "temperature": 0.1,   # low temperature = more factual, less creative
            "num_ctx": 4096,      # context window passed to the model
        },
    )

    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            yield token


def query(user_message: str, context: str) -> str:
    """Non-streaming version. Returns the full response as a string."""
    return "".join(query_stream(user_message, context))


def check_ollama_connection() -> tuple[bool, str]:
    """Verify Ollama is running and the configured model is available."""
    try:
        models = ollama.list()
        model_names = [m.model for m in models.models]
        if config.OLLAMA_MODEL not in model_names:
            available = ", ".join(model_names) if model_names else "none"
            return False, (
                f"Model '{config.OLLAMA_MODEL}' not found. "
                f"Run: ollama pull {config.OLLAMA_MODEL}\n"
                f"Available models: {available}"
            )
        return True, f"Connected. Using {config.OLLAMA_MODEL}"
    except Exception as e:
        return False, f"Cannot reach Ollama at {config.OLLAMA_HOST}. Is it running? ({e})"
