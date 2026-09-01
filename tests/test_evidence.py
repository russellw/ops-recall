from ops_recall.agent.evidence import EvidenceChecker, evidence_summary, retarget
from ops_recall.models import (
    Alert,
    CheckStatus,
    Incident,
    RetrievedIncident,
    ScoreBreakdown,
    Signal,
)
from ops_recall.telemetry.mock import FixtureTelemetryProvider
from ops_recall.telemetry.provider import NullTelemetryProvider
from tests.conftest import days_ago


def _retrieved(incident: Incident, similarity: float = 0.8) -> RetrievedIncident:
    return RetrievedIncident(
        incident=incident, similarity=similarity, breakdown=ScoreBreakdown()
    )


def _incident(incident_id: str, signals: list[Signal], services=("checkout-api",)) -> Incident:
    return Incident(
        id=incident_id,
        title="pool exhausted",
        started_at=days_ago(30),
        services=list(services),
        signals=signals,
    )


def test_confirmed_and_refuted_reflect_live_state(service, pool_alert, quiet_provider):
    ranked = service.agent.ranker.rank_alert(pool_alert)

    loud = EvidenceChecker(service.provider).check(ranked.results, pool_alert)
    assert all(c.status is CheckStatus.CONFIRMED for c in loud)

    quiet = EvidenceChecker(quiet_provider).check(ranked.results, pool_alert)
    statuses = {c.signal_id: c.status for c in quiet}
    assert statuses["blocking-lock-orders-db"] is CheckStatus.REFUTED
    assert statuses["pool-saturated-checkout"] is CheckStatus.REFUTED
    # A deploy 12 minutes ago refutes "no deploy preceded this".
    assert statuses["no-recent-deploy"] is CheckStatus.REFUTED


def test_unavailable_telemetry_is_unknown_not_refuted():
    incident = _incident(
        "INC-X",
        [Signal(id="s1", statement="locks", probe="check_db_locks",
                probe_args={"database": "orders_db"})],
    )
    checks = EvidenceChecker(NullTelemetryProvider()).check([_retrieved(incident)])
    assert checks[0].status is CheckStatus.UNKNOWN
    assert checks[0].observation is not None
    assert checks[0].observation.ok is False


def test_a_shared_signal_is_probed_once_and_credited_to_both(telemetry_state):
    signal = Signal(
        id="shared", statement="locks", probe="check_db_locks",
        probe_args={"database": "orders_db", "min_seconds": 60},
    )
    incidents = [_incident("INC-A", [signal]), _incident("INC-B", [signal])]
    checks = EvidenceChecker(FixtureTelemetryProvider(state=telemetry_state)).check(
        [_retrieved(i) for i in incidents]
    )
    assert len(checks) == 1
    assert checks[0].incident_ids == ["INC-A", "INC-B"]


def test_probe_is_retargeted_at_the_service_the_alert_names():
    signal = Signal(
        id="pool", statement="pool saturated", probe="check_connection_pool",
        probe_args={"service": "checkout-api"},
    )
    alert = Alert(id="A", title="payments-api pool exhausted", service="payments-api")
    args, note = retarget(signal, ["checkout-api"], alert)
    assert args["service"] == "payments-api"
    assert "checkout-api -> payments-api" in note


def test_probe_is_left_alone_when_the_argument_was_not_the_subject():
    signal = Signal(
        id="pool", statement="pool saturated", probe="check_connection_pool",
        probe_args={"service": "some-other-service"},
    )
    alert = Alert(id="A", title="payments-api pool exhausted", service="payments-api")
    args, note = retarget(signal, ["checkout-api"], alert)
    assert args["service"] == "some-other-service"
    assert note == ""


def test_signals_without_a_usable_probe_are_skipped():
    incident = _incident(
        "INC-Y",
        [
            Signal(id="narrative", statement="the team was tired"),
            Signal(id="bogus", statement="?", probe="check_the_vibes"),
        ],
    )
    assert EvidenceChecker(NullTelemetryProvider()).check([_retrieved(incident)]) == []


def test_confirmed_evidence_is_ordered_first(service, pool_alert, quiet_provider):
    ranked = service.agent.ranker.rank_alert(pool_alert)
    checks = EvidenceChecker(quiet_provider).check(ranked.results, pool_alert)
    statuses = [c.status for c in checks]
    assert statuses == sorted(
        statuses,
        key=lambda s: {CheckStatus.CONFIRMED: 0, CheckStatus.REFUTED: 1,
                       CheckStatus.UNKNOWN: 2}[s],
    )


def test_evidence_summary_counts_every_status():
    assert evidence_summary([]) == {"confirmed": 0, "refuted": 0, "unknown": 0}
