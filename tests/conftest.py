from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from ops_recall.config import REPO_ROOT, Settings
from ops_recall.ingest.pipeline import build_index, load_corpus
from ops_recall.models import Alert, Severity
from ops_recall.retrieval.store import IncidentStore
from ops_recall.service import build_service
from ops_recall.telemetry.mock import FixtureTelemetryProvider


@pytest.fixture(scope="session")
def settings() -> Settings:
    """In-memory index, deterministic embedder, no model calls."""
    return Settings(
        reasoner="template",
        embedder="hashing",
        data_dir=REPO_ROOT / "data" / "seed",
        telemetry_fixture=REPO_ROOT / "data" / "telemetry" / "current_state.json",
        qdrant_url=None,
        qdrant_path=None,
    )


@pytest.fixture(scope="session")
def corpus(settings: Settings):
    return load_corpus(settings)


@pytest.fixture(scope="session")
def store(settings: Settings) -> IncidentStore:
    store, _ = build_index(settings)
    return store


@pytest.fixture(scope="session")
def service(settings: Settings, store: IncidentStore):
    return build_service(settings, store=store)


@pytest.fixture
def telemetry_state() -> dict:
    path = REPO_ROOT / "data" / "telemetry" / "current_state.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def quiet_provider(telemetry_state: dict) -> FixtureTelemetryProvider:
    """The same system with nothing wrong: no blocking session, healthy pool."""
    state = json.loads(json.dumps(telemetry_state))
    state["databases"]["orders_db"]["blocking_sessions"] = []
    state["services"]["checkout-api"]["pool"] = {
        "active": 7,
        "idle": 43,
        "max": 50,
        "pending": 0,
    }
    state["services"]["checkout-api"]["error_rate"] = 0.002
    state["services"]["checkout-api"]["deploys"] = [
        {"version": "2026.09.01-1", "minutes_ago": 12, "by": "sasha"}
    ]
    return FixtureTelemetryProvider(state=state)


@pytest.fixture
def pool_alert() -> Alert:
    return Alert(
        id="ALRT-TEST-1",
        title="checkout-api 5xx ratio > 25% (ERR_5012 connection timeout)",
        description=(
            "HikariPool-1 - Connection is not available, request timed out after "
            "30000ms. java.sql.SQLTransientConnectionException on POST /checkout."
        ),
        service="checkout-api",
        host="db-prod-03",
        severity=Severity.SEV1,
        labels={"alertname": "HighErrorRatio", "error_code": "ERR_5012"},
    )


@pytest.fixture
def novel_alert() -> Alert:
    return Alert(
        id="ALRT-TEST-2",
        title="Sudden spike in GraphQL schema validation failures",
        description="Unrecognized directive @defer on the storefront gateway",
        service="storefront-gw",
        severity=Severity.SEV3,
    )


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def days_ago(days: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)
