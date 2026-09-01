"""Hybrid ranking: three lanes, one explainable number.

    similarity = (w_sem * semantic + w_lex * lexical + w_kw * keyword) * recency

The lane weights renormalize when a lane does not apply -- an alert with no
error codes or host ids in it has nothing for the keyword lane to match, and
should not be capped at 80% for it. Recency is a multiplier rather than a
fourth term so it can never manufacture similarity that is not there: a recent
incident that does not match still scores zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ops_recall.config import Settings
from ops_recall.models import (
    Alert,
    RetrievedIncident,
    ScoreBreakdown,
    utcnow,
)
from ops_recall.retrieval.sparse import BM25Encoder
from ops_recall.retrieval.store import Candidate, IncidentStore, SearchFilters
from ops_recall.retrieval.temporal import age_in_days, decay_weight
from ops_recall.retrieval.text import extract_identifiers, keyword_overlap


def _lexical_ceiling(bm25: BM25Encoder, query_text: str, keywords: list[str]) -> float:
    """Score a perfect lexical match would earn for this query.

    Normalizing against the query's own idf mass keeps the lexical lane on an
    absolute 0-1 scale, instead of the usual trick of dividing by the best
    score in the result set (which makes the top hit look perfect even when the
    whole result set is poor).

    Only terms the corpus actually contains count toward the ceiling. A query
    term that appears in no document has the highest idf of all and is
    unmatchable by construction; leaving it in would let one unfamiliar token in
    an alert deflate the lexical score of every result.
    """
    weights = bm25.encode_query(query_text, extra_terms=keywords, corpus_terms_only=True)
    return sum(weights.values()) or 1.0


@dataclass
class RankedResults:
    query_text: str
    query_keywords: list[str]
    results: list[RetrievedIncident]
    considered: int

    @property
    def best(self) -> RetrievedIncident | None:
        return self.results[0] if self.results else None


class HybridRanker:
    def __init__(self, store: IncidentStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings

    def rank(
        self,
        query_text: str,
        keywords: list[str] | None = None,
        top_k: int | None = None,
        filters: SearchFilters | None = None,
        now: datetime | None = None,
    ) -> RankedResults:
        now = now or utcnow()
        keywords = keywords if keywords is not None else extract_identifiers(query_text)
        candidates = self.store.search_incidents(
            query_text, keywords=keywords, filters=filters
        )
        ceiling = _lexical_ceiling(self.store.bm25, query_text, keywords)

        scored = [self._score(c, keywords, ceiling, now) for c in candidates]
        scored.sort(key=lambda r: r.similarity, reverse=True)
        keep = [r for r in scored if r.similarity >= self.settings.min_similarity]
        return RankedResults(
            query_text=query_text,
            query_keywords=keywords,
            results=keep[: top_k or self.settings.top_k],
            considered=len(candidates),
        )

    def rank_alert(self, alert: Alert, **kwargs) -> RankedResults:
        text = alert.query_text()
        keywords = extract_identifiers(text)
        # The alert's own service and host are exact-match tokens by
        # definition, whatever they look like: `checkout-api` carries no digit
        # and so is not identifier-shaped, but matching it is not a coincidence.
        for explicit in (alert.service, alert.host):
            if explicit:
                keywords.append(explicit.lower())
        return self.rank(text, keywords=list(dict.fromkeys(keywords)), **kwargs)

    # -- scoring --------------------------------------------------------

    def _score(
        self,
        candidate: Candidate,
        query_keywords: list[str],
        lexical_ceiling: float,
        now: datetime,
    ) -> RetrievedIncident:
        settings = self.settings
        incident = candidate.incident

        semantic = min(1.0, max(0.0, candidate.semantic))
        lexical = min(1.0, max(0.0, candidate.lexical / lexical_ceiling))
        keyword, matched = keyword_overlap(
            query_keywords, [*incident.keywords, *incident.services, incident.id]
        )

        weights = {
            "semantic": settings.weight_semantic,
            "lexical": settings.weight_lexical,
            "keyword": settings.weight_keyword if query_keywords else 0.0,
        }
        total = sum(weights.values()) or 1.0
        base = (
            weights["semantic"] * semantic
            + weights["lexical"] * lexical
            + weights["keyword"] * keyword
        ) / total

        age = age_in_days(incident.started_at, now)
        recency = decay_weight(age, settings.half_life_days, settings.recency_floor)
        final = base * recency

        return RetrievedIncident(
            incident=incident,
            similarity=round(final, 4),
            breakdown=ScoreBreakdown(
                semantic=round(semantic, 4),
                lexical=round(lexical, 4),
                keyword=round(keyword, 4),
                matched_keywords=matched,
                base=round(base, 4),
                recency_weight=round(recency, 4),
                age_days=round(age, 1),
                final=round(final, 4),
            ),
        )
