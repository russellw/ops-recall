"""End-to-end pipeline behavior with the deterministic reconstructor."""

from ops_recall.agent.graph import IncidentAgent
from ops_recall.agent.reconstruct import TemplateReconstructor
from ops_recall.models import CheckStatus


def test_flagship_alert_reconstructs_the_incident(service, pool_alert):
    state = service.agent.run(pool_alert)
    reconstruction = state["reconstruction"]

    assert [b.incident_id for b in reconstruction.similar_incidents][:2] == [
        "INC-2197",
        "INC-1842",
    ]
    assert reconstruction.similar_incidents[0].similarity_pct > 65
    assert "INC-2197" in reconstruction.headline

    # The root cause and the actions come from the archive, not from nowhere.
    assert "VACUUM FULL" in reconstruction.root_cause_hypothesis
    assert any(
        "Terminate the blocking backend" in action.action
        for action in reconstruction.recommended_actions
    )

    assert all(c.status is CheckStatus.CONFIRMED for c in reconstruction.evidence)
    assert reconstruction.confidence > 0.6
    assert reconstruction.reasoner == "template"


def test_destructive_recommendations_are_flagged_and_gated(service, pool_alert):
    reconstruction = service.agent.reconstruct(pool_alert)
    terminate = next(
        a for a in reconstruction.recommended_actions if "Terminate" in a.action
    )
    assert terminate.risk == "high"
    assert terminate.requires_human_approval is True
    assert any("re-derive" in caveat for caveat in reconstruction.caveats)


def test_refuted_evidence_lowers_confidence_and_says_so(
    settings, store, quiet_provider, pool_alert
):
    """Same alert, same archive, healthy system: the resemblance is superficial
    and the answer has to say that."""
    agent = IncidentAgent(
        store=store,
        provider=quiet_provider,
        settings=settings,
        reconstructor=TemplateReconstructor(settings),
    )
    reconstruction = agent.reconstruct(pool_alert)

    assert any(c.status is CheckStatus.REFUTED for c in reconstruction.evidence)
    assert "does NOT show" in reconstruction.narrative
    assert reconstruction.confidence < 0.4
    assert any("superficial" in caveat for caveat in reconstruction.caveats)


def test_novel_alert_takes_the_cold_start_path(service, novel_alert):
    state = service.agent.run(novel_alert)
    reconstruction = state["reconstruction"]

    assert reconstruction.reasoner == "cold-start"
    assert reconstruction.similar_incidents == []
    assert reconstruction.recommended_actions == []
    assert reconstruction.confidence < 0.1
    assert "novel" in reconstruction.narrative
    assert [entry["node"] for entry in state["trace"]] == ["retrieve", "cold_start"]


def test_trace_records_every_node(service, pool_alert):
    state = service.agent.run(pool_alert)
    nodes = [entry["node"] for entry in state["trace"]]
    assert nodes == ["retrieve", "check_evidence", "gather_quotes", "reconstruct"]
    assert all(entry["ms"] >= 0 for entry in state["trace"])

    evidence_step = state["trace"][1]
    assert evidence_step["confirmed"] == 4
    assert evidence_step["provider"] == "fixture"


def test_quotes_are_human_commentary_from_the_matched_incidents(service, pool_alert):
    state = service.agent.run(pool_alert)
    quotes = state["quotes"]

    assert quotes
    assert {f.incident_id for f, _ in quotes} == {"INC-2197", "INC-1842"}
    # Every quote is something a person wrote. The highest-similarity fragment
    # is the old PagerDuty alert text, which is a near-copy of the query and
    # earns its place nowhere.
    assert all(fragment.author for fragment, _ in quotes)
    assert not any(fragment.text.startswith("PagerDuty") for fragment, _ in quotes)
    assert any("HikariPool" in fragment.text for fragment, _ in quotes)


def test_quote_selection_caps_each_incident(service, pool_alert):
    state = service.agent.run(pool_alert)
    counts: dict[str, int] = {}
    for fragment, _ in state["quotes"]:
        counts[fragment.incident_id] = counts.get(fragment.incident_id, 0) + 1
    assert max(counts.values()) <= 3
