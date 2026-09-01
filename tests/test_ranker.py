"""Ranking behavior: the three lanes, the metadata filters, and time decay."""

from __future__ import annotations

import pytest

from ops_recall.config import Settings
from ops_recall.models import Incident, Severity
from ops_recall.retrieval.ranker import HybridRanker
from ops_recall.retrieval.store import IncidentStore, SearchFilters
from tests.conftest import days_ago


def test_flagship_alert_recalls_the_right_family(service, pool_alert):
    ranked = service.agent.ranker.rank_alert(pool_alert)
    ids = [item.incident.id for item in ranked.results]
    assert ids[:2] == ["INC-2197", "INC-1842"]
    assert ranked.results[0].similarity_pct > 65


def test_novel_alert_matches_nothing(service, novel_alert):
    ranked = service.agent.ranker.rank_alert(novel_alert)
    assert ranked.results == []
    assert ranked.considered > 0  # candidates were retrieved and then rejected


def test_exact_identifiers_are_matched_exactly(service, pool_alert):
    ranked = service.agent.ranker.rank_alert(pool_alert)
    breakdown = ranked.results[0].breakdown
    assert "err_5012" in breakdown.matched_keywords
    assert "db-prod-03" in breakdown.matched_keywords
    assert breakdown.keyword == 1.0


def test_every_lane_is_reported_for_every_result(service, pool_alert):
    ranked = service.agent.ranker.rank_alert(pool_alert)
    for item in ranked.results:
        b = item.breakdown
        # A lane score of exactly zero across the board would mean the result
        # was carried by one lane and never scored on the others.
        assert b.semantic > 0
        assert b.lexical > 0
        assert 0 < b.recency_weight <= 1
        assert b.final == pytest.approx(b.base * b.recency_weight, abs=0.01)


def test_keyword_weight_is_redistributed_when_the_query_has_no_identifiers(service):
    ranked = service.agent.ranker.rank("consumer lag keeps climbing", keywords=[])
    assert ranked.results
    for item in ranked.results:
        assert item.breakdown.keyword == 0.0
        # Without redistribution the base could not exceed 1 - weight_keyword.
        assert item.breakdown.base <= 1.0


def _twin_store(settings: Settings) -> IncidentStore:
    """Two incidents that are textually identical apart from their date."""
    body = dict(
        title="widget-api latency spike from a saturated thread pool",
        summary="Requests queued behind a saturated thread pool.",
        root_cause="Thread pool saturated by a slow downstream call.",
        severity=Severity.SEV2,
        services=["widget-api"],
        symptoms=["p99 latency above 5s", "thread pool queue depth climbing"],
        keywords=["widget-api", "ERR_9001"],
    )
    incidents = [
        Incident(id="INC-OLD", started_at=days_ago(1200), **body),
        Incident(id="INC-NEW", started_at=days_ago(20), **body),
    ]
    store = IncidentStore(settings)
    store.index(incidents, [], recreate=True)
    return store


def test_recency_breaks_ties_between_identical_incidents(settings):
    store = _twin_store(settings)
    ranked = HybridRanker(store, settings).rank(
        "widget-api p99 latency spike, thread pool queue climbing", top_k=2
    )
    ids = [item.incident.id for item in ranked.results]
    assert ids == ["INC-NEW", "INC-OLD"]

    new, old = ranked.results
    # Identical text, so the difference is entirely the decay multiplier.
    assert new.breakdown.base == pytest.approx(old.breakdown.base, abs=0.01)
    assert new.breakdown.recency_weight > old.breakdown.recency_weight


def test_decay_never_buries_a_strong_old_match(settings):
    store = _twin_store(settings)
    aggressive = settings.model_copy(update={"half_life_days": 30, "min_similarity": 0.0})
    ranked = HybridRanker(store, aggressive).rank(
        "widget-api p99 latency spike, thread pool queue climbing", top_k=2
    )
    old = next(i for i in ranked.results if i.incident.id == "INC-OLD")
    assert old.breakdown.recency_weight >= aggressive.recency_floor


def test_service_filter_is_applied_in_the_store(service):
    ranked = service.agent.ranker.rank(
        "connection pool exhausted", filters=SearchFilters(services=["kafka"]), top_k=10
    )
    for item in ranked.results:
        assert "kafka" in [s.lower() for s in item.incident.services]


def test_age_filter_excludes_old_incidents(service):
    ranked = service.agent.ranker.rank(
        "connection pool exhausted on orders_db",
        filters=SearchFilters(max_age_days=200),
        top_k=10,
    )
    ids = [item.incident.id for item in ranked.results]
    assert "INC-1842" not in ids  # 2024, well outside the window
    assert ids  # but the recent sibling is still found


def test_severity_filter_keeps_only_severe_incidents(service):
    ranked = service.agent.ranker.rank(
        "database problem",
        filters=SearchFilters(min_severity=Severity.SEV1),
        top_k=10,
    )
    for item in ranked.results:
        assert item.incident.severity is Severity.SEV1
