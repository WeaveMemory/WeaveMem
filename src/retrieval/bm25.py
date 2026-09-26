

from __future__ import annotations

from functools import lru_cache
import math
import re

import config
from retrieval.lemmatization import lemmatize_for_bm25

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize_legacy(text: str) -> list[str]:
    """Original lowercase word/number tokenizer."""
    return _TOKEN_RE.findall((text or "").lower())


@lru_cache(maxsize=16384)
def _lemmatize_cached(text: str) -> str:
    return lemmatize_for_bm25(text)


def tokenize(text: str, *, lemmatize: bool | None = None) -> list[str]:

    source = text or ""
    enabled = config.BM25_LEMMATIZATION_ENABLED if lemmatize is None else lemmatize
    if not enabled:
        return tokenize_legacy(source)
    return _TOKEN_RE.findall(_lemmatize_cached(source).lower())


def clear_tokenization_cache() -> None:
    """Clear normalized-text caching; primarily useful in tests."""
    _lemmatize_cached.cache_clear()


class BM25:
    """Score a fixed corpus of token-lists against ad-hoc queries."""

    def __init__(self, corpus: list[list[str]]):
        self.corpus = corpus
        self.n = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avg_len = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.df: dict[str, int] = {}
        for doc in corpus:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in self.df.items()
        }

    def scores(self, query: list[str]) -> list[float]:
        """BM25 score of the query against every document, in corpus order."""
        k1, b = config.BM25_K1, config.BM25_B
        out = [0.0] * self.n
        if not self.avg_len:
            return out
        for i, doc in enumerate(self.corpus):
            if not doc:
                continue
            freqs: dict[str, int] = {}
            for term in doc:
                freqs[term] = freqs.get(term, 0) + 1
            norm = k1 * (1 - b + b * self.doc_len[i] / self.avg_len)
            score = 0.0
            for term in query:
                tf = freqs.get(term)
                if not tf:
                    continue
                score += self.idf.get(term, 0.0) * (tf * (k1 + 1)) / (tf + norm)
            out[i] = score
        return out
