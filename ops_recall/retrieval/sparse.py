"""BM25 sparse vectors -- the lexical lane of the hybrid search.

BM25 is factored into a document side (term frequency saturation + length
normalization) and a query side (inverse document frequency), so their dot
product reproduces the BM25 score. That lets Qdrant do the retrieval with its
native sparse index while the scoring stays standard and inspectable.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Iterable, Sequence

from ops_recall.retrieval.text import expand_token, tokenize

K1 = 1.5  # term-frequency saturation
B = 0.75  # length normalization strength


def token_index(token: str) -> int:
    """Stable 31-bit index for a token (Qdrant sparse indices are unsigned)."""
    return int.from_bytes(
        hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(), "big"
    ) & 0x7FFFFFFF


def _terms(text: str) -> list[str]:
    terms: list[str] = []
    for token in tokenize(text):
        terms.extend(expand_token(token))
    return terms


class BM25Encoder:
    """Fit on the corpus at index time; reused unchanged at query time."""

    def __init__(self) -> None:
        self.doc_freq: dict[str, int] = {}
        self.doc_count: int = 0
        self.avg_len: float = 1.0

    def fit(self, corpus: Sequence[str]) -> "BM25Encoder":
        doc_freq: Counter[str] = Counter()
        total_len = 0
        for text in corpus:
            terms = _terms(text)
            total_len += len(terms)
            doc_freq.update(set(terms))
        self.doc_freq = dict(doc_freq)
        self.doc_count = len(corpus)
        self.avg_len = (total_len / len(corpus)) if corpus else 1.0
        return self

    def idf(self, term: str) -> float:
        """Robertson/Sparck-Jones idf, floored at zero so terms appearing in
        most documents cannot push a score negative."""
        n = self.doc_freq.get(term, 0)
        return max(0.0, math.log(1.0 + (self.doc_count - n + 0.5) / (n + 0.5)))

    def encode_document(self, text: str) -> dict[int, float]:
        terms = _terms(text)
        if not terms:
            return {}
        length = len(terms)
        counts = Counter(terms)
        norm = K1 * (1 - B + B * length / (self.avg_len or 1.0))
        weights: dict[int, float] = {}
        for term, tf in counts.items():
            weight = (tf * (K1 + 1)) / (tf + norm)
            index = token_index(term)
            # Hash collisions are rare; keep the stronger of the two signals.
            weights[index] = max(weights.get(index, 0.0), weight)
        return weights

    def encode_query(
        self,
        text: str,
        extra_terms: Iterable[str] = (),
        corpus_terms_only: bool = False,
    ) -> dict[int, float]:
        terms = _terms(text) + [t.lower() for t in extra_terms]
        weights: dict[int, float] = {}
        for term in dict.fromkeys(terms):
            if corpus_terms_only and term not in self.doc_freq:
                continue
            idf = self.idf(term)
            if idf <= 0.0:
                continue
            index = token_index(term)
            weights[index] = max(weights.get(index, 0.0), idf)
        return weights

    def to_dict(self) -> dict:
        return {
            "doc_freq": self.doc_freq,
            "doc_count": self.doc_count,
            "avg_len": self.avg_len,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BM25Encoder":
        encoder = cls()
        encoder.doc_freq = dict(data.get("doc_freq", {}))
        encoder.doc_count = int(data.get("doc_count", 0))
        encoder.avg_len = float(data.get("avg_len", 1.0))
        return encoder
