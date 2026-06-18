"""Lazy fastembed singleton — 384-dim multilingual embeddings via ONNX."""

import asyncio
import os

from fastembed import TextEmbedding

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE_DIR = os.getenv("FASTEMBED_CACHE_PATH")

_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(MODEL_NAME, cache_dir=CACHE_DIR)
    return _model


def _embed_sync(text: str) -> list[float]:
    embeddings = list(_get_model().embed([text]))
    return embeddings[0].tolist()


async def embed(text: str) -> list[float]:
    """Return a 384-dim embedding for text. Runs ONNX inference in a thread executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_sync, text)
