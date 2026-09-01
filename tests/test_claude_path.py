"""The model-backed path, exercised against a fake Anthropic client.

These tests pin the request shape (model id, structured output, cached system
prompt) and -- more importantly -- the division of labor: the model writes
prose and picks actions, while similarity scores, evidence statuses and
incident ids are merged in by the system afterwards and cannot be overwritten
by whatever the model returns.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ops_recall.agent.evidence import EvidenceChecker
from ops_recall.agent.investigate import AgenticInvestigator
from ops_recall.agent.reconstruct import (
    ClaudeReconstructor,
    ReconstructionError,
    build_reconstructor,
)
from ops_recall.agent.tools import ToolContext, build_tools
from ops_recall.models import CheckStatus, ReconstructionDraft, RecommendedAction


class FakeMessages:
    def __init__(self, result):
        self._result = result
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class FakeClient:
    def __init__(self, result):
        self.messages = FakeMessages(result)


DRAFT = ReconstructionDraft(
    headline="Looks 75% like INC-2197: pool starved by a blocking VACUUM FULL.",
    root_cause_hypothesis="A maintenance statement is holding an exclusive lock.",
    narrative="The pool is saturated and a session has been blocking for six minutes.",
    recommended_actions=[
        RecommendedAction(
            action="Terminate the blocking backend",
            rationale="This resolved INC-2197 in 40 seconds.",
            command="SELECT pg_terminate_backend(51774)",
            from_incident="INC-2197",
            risk="high",
        )
    ],
    confidence=0.86,
    caveats=["Terminating the backend aborts the maintenance job."],
)


@pytest.fixture
def prepared(service, pool_alert):
    ranked = service.agent.ranker.rank_alert(pool_alert)
    evidence = EvidenceChecker(service.provider).check(ranked.results, pool_alert)
    return ranked.results, evidence


def test_request_shape_is_pinned(settings, prepared, pool_alert):
    retrieved, evidence = prepared
    client = FakeClient(SimpleNamespace(parsed_output=DRAFT, stop_reason="end_turn"))
    ClaudeReconstructor(client, settings).reconstruct(pool_alert, retrieved, evidence)

    call = client.messages.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["output_format"] is ReconstructionDraft
    assert call["max_tokens"] == settings.max_tokens
    # The system prompt is invariant across alerts, so it is cached.
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "incident reconstruction engine" in call["system"][0]["text"]


def test_prompt_carries_the_evidence_and_the_history(settings, prepared, pool_alert):
    retrieved, evidence = prepared
    client = FakeClient(SimpleNamespace(parsed_output=DRAFT, stop_reason="end_turn"))
    ClaudeReconstructor(client, settings).reconstruct(pool_alert, retrieved, evidence)

    prompt = client.messages.calls[0]["messages"][0]["content"]
    assert "INC-2197" in prompt and "INC-1842" in prompt
    assert "CONFIRMED" in prompt
    assert "VACUUM FULL orders" in prompt  # the live probe output
    assert "[FIX]" in prompt  # which step actually resolved it
    assert "recency weight" in prompt  # the model is told how stale the match is


def test_facts_are_merged_in_not_taken_from_the_model(settings, prepared, pool_alert):
    retrieved, evidence = prepared
    client = FakeClient(SimpleNamespace(parsed_output=DRAFT, stop_reason="end_turn"))
    result = ClaudeReconstructor(client, settings).reconstruct(
        pool_alert, retrieved, evidence
    )

    assert result.reasoner == "claude"
    assert result.headline == DRAFT.headline
    assert result.confidence == 0.86
    # These the model never saw a schema slot for:
    assert [b.incident_id for b in result.similar_incidents] == [
        item.incident.id for item in retrieved
    ]
    assert [b.similarity_pct for b in result.similar_incidents] == [
        item.similarity_pct for item in retrieved
    ]
    assert result.evidence == list(evidence)
    assert all(c.status is CheckStatus.CONFIRMED for c in result.evidence)


def test_refusal_is_surfaced_not_silently_empty(settings, prepared, pool_alert):
    retrieved, evidence = prepared
    client = FakeClient(
        SimpleNamespace(
            parsed_output=None,
            stop_reason="refusal",
            stop_details=SimpleNamespace(category="cyber", explanation="no"),
        )
    )
    with pytest.raises(ReconstructionError, match="declined"):
        ClaudeReconstructor(client, settings).reconstruct(pool_alert, retrieved, evidence)


def test_unparsable_response_is_an_error(settings, prepared, pool_alert):
    retrieved, evidence = prepared
    client = FakeClient(SimpleNamespace(parsed_output=None, stop_reason="end_turn"))
    with pytest.raises(ReconstructionError, match="no parsable"):
        ClaudeReconstructor(client, settings).reconstruct(pool_alert, retrieved, evidence)


def test_claude_reasoner_without_a_client_is_refused(settings):
    with pytest.raises(ReconstructionError, match="ANTHROPIC_API_KEY"):
        build_reconstructor(settings.model_copy(update={"reasoner": "claude"}), None)


# -- tools ----------------------------------------------------------------


def test_tools_are_read_only_and_record_what_they_observed(service):
    context = ToolContext(provider=service.provider, ranker=service.agent.ranker)
    tools = build_tools(context)
    by_name = {tool.name: tool for tool in tools}

    assert "check_db_locks" in by_name
    # Nothing in the tool surface can change the system: every tool reads.
    assert all(
        name.startswith(("check_", "search_", "get_")) for name in by_name
    )

    payload = json.loads(by_name["check_db_locks"].call({"database": "orders_db"}))
    assert payload["ok"] is True
    assert payload["data"]["holds"] is True
    assert context.observations[-1].tool == "check_db_locks"


def test_search_and_runbook_tools_reach_the_corpus(service):
    context = ToolContext(provider=service.provider, ranker=service.agent.ranker)
    by_name = {tool.name: tool for tool in build_tools(context)}

    hits = json.loads(by_name["search_incidents"].call({"query": "consumer lag climbing"}))
    assert hits[0]["incident_id"] == "INC-2088"

    runbook = json.loads(by_name["get_runbook"].call({"topic": "connection pool exhausted"}))
    assert any("pg_blocking_pids" in section["text"] for section in runbook)


# -- agentic investigation -------------------------------------------------


class FakeRunnerClient:
    """Stands in for `client.beta.messages.tool_runner`."""

    def __init__(self, messages=None, error: Exception | None = None):
        self._messages = messages or []
        self._error = error
        self.kwargs: dict = {}
        outer = self

        class Runner:
            def __init__(self, **kwargs):
                outer.kwargs = kwargs
                if outer._error:
                    raise outer._error

            def __iter__(self):
                return iter(outer._messages)

        self.beta = SimpleNamespace(messages=SimpleNamespace(tool_runner=Runner))


def _text_message(text: str):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def test_investigator_collects_notes_and_passes_tools(settings, service, prepared, pool_alert):
    retrieved, evidence = prepared
    client = FakeRunnerClient([_text_message("Nothing further worth checking.")])
    investigation = AgenticInvestigator(client, settings, service.agent.ranker).investigate(
        pool_alert, retrieved, evidence, service.provider
    )

    assert investigation.notes == "Nothing further worth checking."
    assert investigation.error is None
    assert client.kwargs["model"] == "claude-opus-5"
    tool_names = {tool.name for tool in client.kwargs["tools"]}
    assert "check_db_locks" in tool_names and "get_runbook" in tool_names
    assert "CONFIRMED" in client.kwargs["messages"][0]["content"]


def test_investigator_stops_at_the_iteration_cap(settings, service, prepared, pool_alert):
    retrieved, evidence = prepared
    capped = settings.model_copy(update={"max_agentic_iterations": 3})
    client = FakeRunnerClient([_text_message(f"step {i}") for i in range(50)])
    investigation = AgenticInvestigator(client, capped, None).investigate(
        pool_alert, retrieved, evidence, service.provider
    )
    assert investigation.iterations == 3


def test_investigation_failure_does_not_break_the_pipeline(
    settings, service, prepared, pool_alert
):
    client = FakeRunnerClient(error=RuntimeError("connection reset"))
    retrieved, evidence = prepared
    investigation = AgenticInvestigator(client, settings, None).investigate(
        pool_alert, retrieved, evidence, service.provider
    )
    assert investigation.error == "connection reset"
    assert investigation.observations == []
