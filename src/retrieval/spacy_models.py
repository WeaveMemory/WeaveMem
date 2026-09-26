

from __future__ import annotations

import logging
import threading
from typing import Any

import config

logger = logging.getLogger(__name__)

_nlp_lemma: Any | None = None
_load_failed_lemma = False
_lock = threading.Lock()


def _ensure_model_available() -> None:
    """Ensure the configured spaCy English model can be loaded."""
    try:
        import spacy
    except ImportError as exc:
        raise ImportError(
            "spaCy is not installed. Install it with: pip install -e '.[nlp]'"
        ) from exc

    model_name = config.BM25_LEMMATIZATION_MODEL
    if spacy.util.is_package(model_name):
        return

    if not config.BM25_LEMMATIZATION_AUTO_DOWNLOAD:
        raise RuntimeError(
            f"spaCy model {model_name} is not installed. "
            f"Install it with: python -m spacy download {model_name}"
        )

    logger.info("Downloading spaCy model %s...", model_name)
    try:
        from spacy.cli import download

        download(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download spaCy model {model_name}: {exc}. "
            f"Install it manually with: python -m spacy download {model_name}"
        ) from exc
    logger.info("spaCy model %s downloaded successfully", model_name)


def get_nlp_lemma() -> Any | None:
    """Return one shared spaCy pipeline optimized for BM25 lemmatization."""
    global _load_failed_lemma, _nlp_lemma

    if _load_failed_lemma:
        return None
    if _nlp_lemma is not None:
        return _nlp_lemma

    with _lock:
        if _nlp_lemma is not None:
            return _nlp_lemma
        if _load_failed_lemma:
            return None
        try:
            _ensure_model_available()
            import spacy

            _nlp_lemma = spacy.load(
                config.BM25_LEMMATIZATION_MODEL,
                disable=["ner", "parser"],
            )
            logger.info(
                "spaCy lemma model %s loaded",
                config.BM25_LEMMATIZATION_MODEL,
            )
        except Exception as exc:
            logger.warning(
                "BM25 lemmatization unavailable; using legacy tokenization: %s",
                exc,
            )
            _load_failed_lemma = True
            return None
    return _nlp_lemma


def reset_nlp_lemma_for_tests() -> None:
    """Reset loader state; intended for deterministic unit tests only."""
    global _load_failed_lemma, _nlp_lemma

    with _lock:
        _nlp_lemma = None
        _load_failed_lemma = False
