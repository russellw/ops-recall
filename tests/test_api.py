import pytest
from fastapi.testclient import TestClient

from ops_recall.api.app import create_app


@pytest.fixture(scope="module")
def client(service):
    with TestClient(create_app(service)) as client:
        yield client


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_stats_reports_the_corpus_and_the_reasoner(client):
    stats = client.get("/v1/stats").json()
    assert stats["corpus"]["incidents"] == 10
    assert stats["index"]["exists"] is True
    assert stats["telemetry"] == "fixture"
    assert stats["reasoner"] == "template"


def test_post_alert_returns_a_reconstruction(client):
    response = client.post(
        "/v1/alerts",
        json={
            "title": "checkout-api 5xx ratio > 25% (ERR_5012 connection timeout)",
            "description": "HikariPool-1 - Connection is not available, request timed out",
            "service": "checkout-api",
            "host": "db-prod-03",
            "severity": "sev1",
        },
    )
    assert response.status_code == 200
    body = response.json()

    assert [r["incident_id"] for r in body["retrieved"]][:2] == ["INC-2197", "INC-1842"]
    assert body["retrieved"][0]["similarity_pct"] > 65
    # The score is explainable, not just a number.
    assert set(body["retrieved"][0]["breakdown"]) >= {
        "semantic", "lexical", "keyword", "recency_weight", "age_days"
    }
    reconstruction = body["reconstruction"]
    assert reconstruction["evidence"]
    assert reconstruction["recommended_actions"]
    assert [entry["node"] for entry in body["trace"]][0] == "retrieve"


def test_pagerduty_webhook_is_accepted_directly(client):
    response = client.post(
        "/v1/alerts/pagerduty",
        json={
            "event": {
                "data": {
                    "id": "PXK7N2QW",
                    "title": "checkout-api 5xx ratio > 25% (ERR_5012 connection timeout)",
                    "created_at": "2026-09-01T14:12:00Z",
                    "urgency": "high",
                    "service": {"summary": "checkout-api"},
                    "body": {"details": {"host": "db-prod-03", "error_code": "ERR_5012",
                                         "description": "HikariPool-1 connection timeout"}},
                }
            }
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["reconstruction"]["alert_id"] == "PXK7N2QW"
    assert "INC-2197" in body["reconstruction"]["headline"]


def test_malformed_pagerduty_payload_is_rejected(client):
    assert client.post("/v1/alerts/pagerduty", json=[]).status_code == 422


def test_search_exposes_the_score_breakdown(client):
    body = client.post(
        "/v1/search", json={"query": "orders-consumer lag climbing", "top_k": 3}
    ).json()
    assert body["results"][0]["incident_id"] == "INC-2088"
    assert body["considered"] > 0
    assert body["results"][0]["breakdown"]["recency_weight"] <= 1.0


def test_search_honors_metadata_filters(client):
    body = client.post(
        "/v1/search",
        json={"query": "database problem", "services": ["kafka"], "top_k": 5},
    ).json()
    for result in body["results"]:
        assert "kafka" in [s.lower() for s in result["services"]]


def test_incident_lookup_is_case_insensitive(client):
    assert client.get("/v1/incidents/inc-2197").json()["id"] == "INC-2197"
    assert client.get("/v1/incidents/INC-9999").status_code == 404


def test_incident_list_includes_provenance(client):
    incidents = client.get("/v1/incidents").json()
    assert len(incidents) == 10
    flagship = next(i for i in incidents if i["id"] == "INC-2197")
    assert set(flagship["sources"]) == {"postmortem", "slack", "pagerduty"}


def test_probes_are_discoverable(client):
    probes = client.get("/v1/probes").json()
    names = {p["name"] for p in probes}
    assert "check_db_locks" in names
    schema = next(p for p in probes if p["name"] == "check_db_locks")["schema"]
    assert schema["required"] == ["database"]


def test_novel_alert_is_reported_as_novel(client):
    body = client.post(
        "/v1/alerts",
        json={"title": "Sudden spike in GraphQL schema validation failures",
              "description": "Unrecognized directive @defer", "service": "storefront-gw"},
    ).json()
    assert body["retrieved"] == []
    assert body["reconstruction"]["reasoner"] == "cold-start"
