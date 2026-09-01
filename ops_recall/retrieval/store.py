"""Qdrant-backed incident store.

One collection holds two kinds of point, distinguished by `payload.kind`:

* `incident` -- one point per historical incident, the unit that gets ranked
  and reconstructed.
* `fragment` -- raw Slack messages, PagerDuty log entries and wiki sections,
  carrying `incident_id` so a reconstruction can quote what people actually
  said at the time.

Each point carries a dense vector (`semantic`) and a sparse BM25 vector
(`lexical`). The two lanes are queried separately rather than fused server-side:
the ranker needs the per-lane scores to explain the headline similarity number,
and "87% because the error code matched exactly" is a different claim from
"87% because the prose felt similar".
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from qdrant_client import QdrantClient, models

from ops_recall.config import Settings
from ops_recall.models import Fragment, Incident, Severity
from ops_recall.retrieval.embeddings import Embedder, build_embedder
from ops_recall.retrieval.sparse import BM25Encoder

logger = logging.getLogger(__name__)

DENSE = "semantic"
SPARSE = "lexical"

_NAMESPACE = uuid.UUID("6f1c2a3e-8b1d-4c7a-9a1e-0f5b2c9d7e11")

_SEVERITY_ORDER = {Severity.SEV1: 1, Severity.SEV2: 2, Severity.SEV3: 3, Severity.SEV4: 4}


def point_id(kind: str, key: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{kind}:{key}"))


@dataclass
class SearchFilters:
    """Metadata constraints applied inside Qdrant, before scoring."""

    services: Sequence[str] = ()
    max_age_days: float | None = None
    min_severity: Severity | None = None
    exclude_ids: Sequence[str] = ()
    require_keywords: Sequence[str] = ()

    def to_qdrant(self, kind: str, now: datetime | None = None) -> models.Filter:
        must: list[models.Condition] = [
            models.FieldCondition(key="kind", match=models.MatchValue(value=kind))
        ]
        must_not: list[models.Condition] = []
        if self.services:
            must.append(
                models.FieldCondition(
                    key="services", match=models.MatchAny(any=[s.lower() for s in self.services])
                )
            )
        if self.max_age_days is not None:
            now = now or datetime.now(timezone.utc)
            cutoff = now.timestamp() - self.max_age_days * 86400
            must.append(
                models.FieldCondition(key="started_at", range=models.Range(gte=cutoff))
            )
        if self.min_severity is not None:
            must.append(
                models.FieldCondition(
                    key="severity_rank",
                    range=models.Range(lte=_SEVERITY_ORDER[self.min_severity]),
                )
            )
        if self.require_keywords:
            must.append(
                models.FieldCondition(
                    key="keywords",
                    match=models.MatchAny(any=[k.lower() for k in self.require_keywords]),
                )
            )
        if self.exclude_ids:
            must_not.append(
                models.FieldCondition(
                    key="incident_id", match=models.MatchAny(any=list(self.exclude_ids))
                )
            )
        return models.Filter(must=must, must_not=must_not or None)


@dataclass
class Candidate:
    """A retrieved incident with its raw, un-fused lane scores."""

    incident: Incident
    semantic: float = 0.0
    lexical: float = 0.0
    lanes: set[str] = field(default_factory=set)


class IncidentStore:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder or build_embedder(settings)
        self.client = client or QdrantClient(**settings.qdrant_location())
        # idf is a property of the collection being searched, so incidents and
        # fragments get their own statistics. Sharing one table let common
        # PagerDuty boilerplate in the fragments distort the incident lane.
        self.bm25 = BM25Encoder()
        self.bm25_fragments = BM25Encoder()
        self.collection = settings.collection
        self.index_counts: dict[str, int] = {}

    # -- lifecycle ------------------------------------------------------

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if exists:
            return
        self.client.create_collection(
            self.collection,
            vectors_config={
                DENSE: models.VectorParams(
                    size=self.embedder.dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams()},
        )
        # Payload indexes matter on a real cluster and are a no-op (with a
        # warning) in embedded mode.
        is_embedded = not self.settings.qdrant_url
        for field_name, schema in (
            ("kind", models.PayloadSchemaType.KEYWORD),
            ("incident_id", models.PayloadSchemaType.KEYWORD),
            ("services", models.PayloadSchemaType.KEYWORD),
            ("keywords", models.PayloadSchemaType.KEYWORD),
            ("source_kind", models.PayloadSchemaType.KEYWORD),
            ("started_at", models.PayloadSchemaType.FLOAT),
            ("severity_rank", models.PayloadSchemaType.INTEGER),
        ):
            if is_embedded:
                continue
            try:
                self.client.create_payload_index(self.collection, field_name, schema)
            except Exception:  # filtering still works without an explicit index
                logger.debug("could not create payload index on %s", field_name)

    def index(
        self,
        incidents: Sequence[Incident],
        fragments: Sequence[Fragment] = (),
        recreate: bool = True,
    ) -> dict[str, int]:
        """Fit the lexical model on the full corpus, then write every point.

        BM25 idf is corpus-global, so indexing is a whole-corpus operation. For
        incremental ingest, append to the corpus and re-index; at post-mortem
        volumes (hundreds to low thousands of documents) that costs seconds.
        """
        self.ensure_collection(recreate=recreate)

        # Two texts per incident, one per lane -- see `Incident.matching_text`.
        incident_texts = [i.searchable_text() for i in incidents]
        incident_match_texts = [i.matching_text() for i in incidents]
        fragment_texts = [f.text for f in fragments]
        self.bm25.fit(incident_texts)
        self.bm25_fragments.fit(fragment_texts)

        points: list[models.PointStruct] = []
        if incidents:
            vectors = self.embedder.encode(incident_match_texts)
            for incident, text, dense in zip(incidents, incident_texts, vectors):
                points.append(
                    models.PointStruct(
                        id=point_id("incident", incident.id),
                        vector={
                            DENSE: dense,
                            SPARSE: _sparse(self.bm25.encode_document(text)),
                        },
                        payload={
                            "kind": "incident",
                            "incident_id": incident.id,
                            "services": [s.lower() for s in incident.services],
                            "keywords": [k.lower() for k in incident.keywords],
                            "started_at": incident.started_at.timestamp(),
                            "severity_rank": _SEVERITY_ORDER[incident.severity],
                            "doc": incident.model_dump(mode="json"),
                        },
                    )
                )
        if fragments:
            vectors = self.embedder.encode(fragment_texts)
            for fragment, text, dense in zip(fragments, fragment_texts, vectors):
                points.append(
                    models.PointStruct(
                        id=point_id("fragment", fragment.id),
                        vector={
                            DENSE: dense,
                            SPARSE: _sparse(self.bm25_fragments.encode_document(text)),
                        },
                        payload={
                            "kind": "fragment",
                            "incident_id": fragment.incident_id or "",
                            "source_kind": fragment.source.kind.value,
                            "keywords": [],
                            "services": [],
                            "started_at": (
                                fragment.timestamp.timestamp() if fragment.timestamp else 0.0
                            ),
                            "severity_rank": 4,
                            "doc": fragment.model_dump(mode="json"),
                        },
                    )
                )
        if points:
            self.client.upsert(self.collection, points=points, wait=True)
        self.index_counts = {"incidents": len(incidents), "fragments": len(fragments)}
        return dict(self.index_counts)

    # -- retrieval ------------------------------------------------------

    def search_incidents(
        self,
        query_text: str,
        keywords: Sequence[str] = (),
        limit: int | None = None,
        filters: SearchFilters | None = None,
    ) -> list[Candidate]:
        """Run both lanes and merge by incident id, keeping raw lane scores."""
        limit = limit or self.settings.candidate_limit
        qdrant_filter = (filters or SearchFilters()).to_qdrant("incident")

        merged: dict[str, Candidate] = {}

        dense_vector = self.embedder.encode([query_text])[0]
        for scored in self._query(dense_vector, DENSE, qdrant_filter, limit):
            self._candidate(merged, scored)

        sparse_weights = self.bm25.encode_query(query_text, extra_terms=keywords)
        if sparse_weights:
            for scored in self._query(_sparse(sparse_weights), SPARSE, qdrant_filter, limit):
                self._candidate(merged, scored)

        # Each lane only reports scores for the points *it* returned, so a
        # candidate found lexically has no semantic score and vice versa.
        # Re-score the merged set directly against both stored vectors: at
        # candidate-set size this is a single retrieve and some arithmetic, and
        # it makes every breakdown complete rather than lane-dependent.
        self._backfill_lane_scores(merged, dense_vector, sparse_weights)
        return list(merged.values())

    def search_fragments(
        self,
        query_text: str,
        incident_ids: Sequence[str] = (),
        source_kinds: Sequence[str] = (),
        limit: int = 8,
    ) -> list[tuple[Fragment, float]]:
        """Supporting quotes, optionally restricted to specific incidents or
        source systems (`source_kinds=["wiki"]` is how runbooks are looked up)."""
        must: list[models.Condition] = [
            models.FieldCondition(key="kind", match=models.MatchValue(value="fragment"))
        ]
        if incident_ids:
            must.append(
                models.FieldCondition(
                    key="incident_id", match=models.MatchAny(any=list(incident_ids))
                )
            )
        if source_kinds:
            must.append(
                models.FieldCondition(
                    key="source_kind", match=models.MatchAny(any=list(source_kinds))
                )
            )
        dense_vector = self.embedder.encode([query_text])[0]
        results = self._query(dense_vector, DENSE, models.Filter(must=must), limit)
        return [
            (Fragment.model_validate(point.payload["doc"]), float(point.score))
            for point in results
        ]

    def get_incident(self, incident_id: str) -> Incident | None:
        records = self.client.retrieve(
            self.collection, ids=[point_id("incident", incident_id)], with_payload=True
        )
        if not records:
            return None
        return Incident.model_validate(records[0].payload["doc"])

    def all_incidents(self) -> list[Incident]:
        incidents: list[Incident] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                self.collection,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="kind", match=models.MatchValue(value="incident"))]
                ),
                limit=256,
                offset=offset,
                with_payload=True,
            )
            incidents.extend(Incident.model_validate(r.payload["doc"]) for r in records)
            if offset is None:
                break
        return sorted(incidents, key=lambda i: i.started_at, reverse=True)

    def stats(self) -> dict[str, Any]:
        if not self.client.collection_exists(self.collection):
            return {"collection": self.collection, "exists": False}
        info = self.client.get_collection(self.collection)
        return {
            "collection": self.collection,
            "exists": True,
            "points": info.points_count,
            "embedder": self.embedder.signature,
            "incident_vocabulary": len(self.bm25.doc_freq),
            "fragment_vocabulary": len(self.bm25_fragments.doc_freq),
        }

    # -- internals ------------------------------------------------------

    def _backfill_lane_scores(
        self,
        merged: dict[str, Candidate],
        dense_query: Sequence[float],
        sparse_query: dict[int, float],
    ) -> None:
        if not merged:
            return
        ids = [point_id("incident", cid) for cid in merged]
        records = self.client.retrieve(
            self.collection, ids=ids, with_payload=True, with_vectors=True
        )
        query_norm = math.sqrt(sum(v * v for v in dense_query)) or 1.0
        for record in records:
            payload = record.payload or {}
            candidate = merged.get(payload.get("incident_id", ""))
            if candidate is None:
                continue
            vectors = record.vector or {}
            dense = vectors.get(DENSE)
            if dense is not None:
                doc_norm = math.sqrt(sum(v * v for v in dense)) or 1.0
                dot = sum(a * b for a, b in zip(dense_query, dense))
                candidate.semantic = max(0.0, dot / (query_norm * doc_norm))
                candidate.lanes.add(DENSE)
            sparse = vectors.get(SPARSE)
            if sparse is not None and sparse_query:
                doc_weights = dict(zip(sparse.indices, sparse.values))
                candidate.lexical = max(
                    0.0,
                    sum(w * doc_weights.get(i, 0.0) for i, w in sparse_query.items()),
                )
                candidate.lanes.add(SPARSE)

    def _query(
        self,
        query: Any,
        using: str,
        qdrant_filter: models.Filter,
        limit: int,
    ) -> list[models.ScoredPoint]:
        return self.client.query_points(
            self.collection,
            query=query,
            using=using,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
        ).points

    @staticmethod
    def _candidate(
        merged: dict[str, Candidate], scored: models.ScoredPoint
    ) -> Candidate | None:
        payload = scored.payload or {}
        incident_id = payload.get("incident_id")
        if not incident_id:
            return None
        candidate = merged.get(incident_id)
        if candidate is None:
            candidate = Candidate(incident=Incident.model_validate(payload["doc"]))
            merged[incident_id] = candidate
        return candidate


def _sparse(weights: dict[int, float]) -> models.SparseVector:
    if not weights:
        # Qdrant rejects a query against an empty sparse vector; a single
        # impossible index is a harmless no-match.
        return models.SparseVector(indices=[0], values=[0.0])
    indices = list(weights.keys())
    return models.SparseVector(indices=indices, values=[weights[i] for i in indices])

