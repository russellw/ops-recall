"""Dense embeddings.

Two implementations behind one protocol:

* `HashingEmbedder` -- deterministic, dependency-free, no network. It is a
  hashed bag of words plus character trigrams, which captures morphological and
  co-occurrence similarity but *not* meaning: "pool starved" and "no free
  connections" stay far apart. It exists so the whole system runs and is
  testable offline, and it is a genuinely reasonable floor for a corpus this
  jargon-heavy.
* `VoyageEmbedder` -- a real semantic model, and what you should run in
  production. Voyage is the embedding provider Anthropic recommends alongside
  Claude; the sparse lane and the exact-identifier lane stay identical.

Swapping embedders changes vector geometry, so the collection is versioned by
`embedder.signature` and re-indexed when that changes.
"""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, Protocol, Sequence, runtime_checkable

from ops_recall.config import Settings
from ops_recall.retrieval.text import expand_token, tokenize


@runtime_checkable
class Embedder(Protocol):
    dim: int
    signature: str

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


class HashingEmbedder:
    """Feature-hashing embedder over words and character trigrams."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.signature = f"hashing-{dim}"

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % self.dim
        # Signed hashing keeps unrelated collisions from stacking up.
        sign = 1.0 if digest[4] & 1 else -1.0
        return index, sign

    def _features(self, text: str) -> Iterable[str]:
        tokens = tokenize(text)
        for token in tokens:
            for part in expand_token(token):
                yield f"w:{part}"
        for token in tokens:
            padded = f"^{token}$"
            for i in range(len(padded) - 2):
                yield f"c:{padded[i : i + 3]}"

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            counts: dict[str, int] = {}
            for feature in self._features(text):
                counts[feature] = counts.get(feature, 0) + 1
            vec = [0.0] * self.dim
            for feature, count in counts.items():
                index, sign = self._bucket(feature)
                # Sublinear tf: the tenth mention of "timeout" says little more
                # than the second.
                vec[index] += sign * (1.0 + math.log(count))
            vectors.append(_l2_normalize(vec))
        return vectors


class VoyageEmbedder:
    """Voyage AI embeddings over HTTPS."""

    def __init__(self, api_key: str, model: str = "voyage-3", dim: int = 1024) -> None:
        if not api_key:
            raise ValueError("VoyageEmbedder requires an API key")
        self._api_key = api_key
        self.model = model
        self.dim = dim
        self.signature = f"voyage-{model}"

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        import httpx  # imported lazily: the offline path must not need it

        vectors: list[list[float]] = []
        # Voyage caps batch size; 128 is comfortably inside every current tier.
        for start in range(0, len(texts), 128):
            batch = list(texts[start : start + 128])
            response = httpx.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"input": batch, "model": self.model, "input_type": "document"},
                timeout=60.0,
            )
            response.raise_for_status()
            payload = response.json()
            for item in sorted(payload["data"], key=lambda d: d["index"]):
                vectors.append(item["embedding"])
        if vectors:
            self.dim = len(vectors[0])
        return vectors


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "voyage":
        if not settings.voyage_api_key:
            raise RuntimeError(
                "OPS_RECALL_EMBEDDER=voyage requires OPS_RECALL_VOYAGE_API_KEY"
            )
        return VoyageEmbedder(settings.voyage_api_key, settings.voyage_model)
    return HashingEmbedder(settings.embedding_dim)
