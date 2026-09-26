

from __future__ import annotations

import logging
import time

from openai import OpenAI

import config
from llm import providers

logger = logging.getLogger("embedding")

_client: OpenAI | None = None


def _endpoint() -> providers.Endpoint:
    return providers.provider_for(config.EMBEDDING_MODEL).endpoint


def available() -> bool:
    """Whether an embedding backend is configured."""
    return bool(_endpoint().api_key)


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        endpoint = _endpoint()
        _client = OpenAI(api_key=endpoint.api_key, base_url=endpoint.base_url)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """Embed texts in batches, preserving input order.

    Return None for empty input, missing credentials, or failed batch requests.
    Callers decide how to handle unavailable vectors.
    """
    if not texts or not available():
        return None
    client = _get_client()
    vectors: list[list[float]] = []
    try:
        for start in range(0, len(texts), config.EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + config.EMBEDDING_BATCH_SIZE]
            response = None
            for attempt in range(1, max(1, int(config.LLM_MAX_RETRIES)) + 1):
                try:
                    response = client.embeddings.create(
                        model=config.EMBEDDING_MODEL,
                        input=batch,
                        timeout=config.EMBEDDING_TIMEOUT_SECONDS,
                    )
                    break
                except Exception as exc:
                    if attempt >= max(1, int(config.LLM_MAX_RETRIES)):
                        raise
                    delay = float(config.LLM_RETRY_BACKOFF_SECONDS) * attempt
                    logger.warning(
                        "embedding batch failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt,
                        max(1, int(config.LLM_MAX_RETRIES)),
                        delay,
                        exc,
                    )
                    time.sleep(delay)
            if response is None:
                raise RuntimeError("embedding backend returned no response")
            # Response order may differ from input order; align vectors by index.
            ordered = sorted(response.data, key=lambda item: item.index)
            if len(ordered) != len(batch):
                raise ValueError(
                    f"expected {len(batch)} embeddings, got {len(ordered)}"
                )
            vectors.extend(item.embedding for item in ordered)
    except Exception as e:  # Report batch failures as unavailable vectors.
        logger.warning("embedding request failed: %s", e)
        return None
    return vectors


def embed_text(text: str) -> list[float] | None:
    """Embed a single text; returns None if the backend is unavailable."""
    vectors = embed_texts([text])
    return vectors[0] if vectors else None
