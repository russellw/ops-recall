from pathlib import Path

import pytest

from ops_recall.config import REPO_ROOT
from ops_recall.ingest.pagerduty import alert_from_webhook, incident_to_fragments
from ops_recall.ingest.postmortems import (
    PostmortemParseError,
    normalize_prose,
    parse_postmortem,
    parse_remediation,
)
from ops_recall.ingest.slack import clean_slack_text, messages_to_fragments
from ops_recall.ingest.wiki import parse_wiki_page
from ops_recall.models import Severity, SourceKind

POSTMORTEMS = REPO_ROOT / "data" / "seed" / "postmortems"


def test_postmortem_parses_facts_prose_and_signals():
    incident = parse_postmortem(POSTMORTEMS / "INC-2197.md")
    assert incident.id == "INC-2197"
    assert incident.severity is Severity.SEV1
    assert "checkout-api" in incident.services
    assert incident.time_to_resolve_minutes == 86
    assert "err_5012" in incident.keywords
    assert {s.id for s in incident.signals} >= {
        "blocking-lock-orders-db",
        "pool-saturated-checkout",
    }
    probe = next(s for s in incident.signals if s.id == "blocking-lock-orders-db")
    assert probe.probe == "check_db_locks"
    assert probe.probe_args["database"] == "orders_db"


def test_remediation_keeps_order_actor_command_and_fix_marking():
    steps = parse_remediation(
        "1. [diagnose] Look at something `SELECT 1` — @priya\n"
        "2. [fix] Terminate the backend `SELECT pg_terminate_backend(1)` — @dan "
        "(outcome: recovered in 40s)\n"
    )
    assert [s.order for s in steps] == [1, 2]
    assert steps[0].is_fix is False
    assert steps[1].is_fix is True
    assert steps[1].actor == "dan"
    assert steps[1].command == "SELECT pg_terminate_backend(1)"
    assert steps[1].outcome == "recovered in 40s"
    assert "`" not in steps[1].action


def test_fix_steps_fall_back_to_all_steps_when_none_are_tagged():
    incident = parse_postmortem(POSTMORTEMS / "INC-2197.md")
    assert all(step.is_fix for step in incident.fix_steps())

    incident.remediation = [step for step in incident.remediation if not step.is_fix]
    assert incident.fix_steps() == incident.remediation


def test_matching_text_excludes_remediation_prose():
    incident = parse_postmortem(POSTMORTEMS / "INC-2197.md")
    assert "pg_terminate_backend" in incident.searchable_text()
    assert "pg_terminate_backend" not in incident.matching_text()


def test_missing_frontmatter_is_an_error(tmp_path: Path):
    path = tmp_path / "broken.md"
    path.write_text("# no frontmatter here\n", encoding="utf-8")
    with pytest.raises(PostmortemParseError):
        parse_postmortem(path)


def test_normalize_prose_unwraps_paragraphs():
    assert normalize_prose("one two\nthree\n\nfour\nfive") == "one two three\n\nfour five"


def test_slack_messages_link_by_thread_and_by_reference():
    messages = [
        {"type": "message", "user": "U1", "ts": "1780755120.000100",
         "thread_ts": "1780755120.000100", "text": "checkout is throwing 503s"},
        {"type": "message", "user": "U2", "ts": "1780755220.000100",
         "thread_ts": "1780755120.000100", "text": "pool pinned at max"},
        {"type": "message", "user": "U2", "ts": "1780755999.000100",
         "text": "unrelated chatter about lunch"},
        {"type": "message", "user": "U1", "ts": "1780756999.000100",
         "text": "see INC-1842 for the last time this happened"},
        {"type": "message", "subtype": "channel_join", "user": "U3",
         "ts": "1780757999.000100", "text": "joined"},
    ]
    fragments = messages_to_fragments(
        "C1", "ops", messages, users={"U1": "priya", "U2": "dan"},
        thread_index={"C1/1780755120.000100": "INC-2197"},
    )
    assert len(fragments) == 4  # the join message is dropped
    assert fragments[0].incident_id == "INC-2197"
    assert fragments[1].incident_id == "INC-2197"  # inherited from the thread
    assert fragments[2].incident_id is None
    assert fragments[3].incident_id == "INC-1842"  # matched by reference
    assert fragments[0].author == "priya"
    assert fragments[0].source.kind is SourceKind.SLACK


def test_slack_markup_is_stripped():
    text = "<@U1|alice> check <https://wiki.internal/db|the runbook> in <#C9|ops> &amp; retry"
    assert clean_slack_text(text) == "@alice check the runbook in #ops & retry"


def test_pagerduty_incident_becomes_trigger_and_log_fragments():
    fragments = incident_to_fragments(
        {
            "id": "PXK7N2QW",
            "title": "checkout-api 5xx ratio > 25%",
            "service": {"summary": "checkout-api"},
            "created_at": "2026-06-02T14:12:00Z",
            "urgency": "high",
            "body": {"details": {"error_code": "ERR_5012"}},
            "log_entries": [
                {"id": "L1", "type": "acknowledge_log_entry",
                 "created_at": "2026-06-02T14:15:00Z", "summary": "Acknowledged by Priya",
                 "agent": {"summary": "Priya"}},
                {"id": "L2", "type": "annotate_log_entry",
                 "created_at": "2026-06-02T14:31:00Z", "summary": "Note added",
                 "channel": {"type": "note", "notes": "blocking backend 48211"}},
            ],
        },
        incident_id="INC-2197",
    )
    # The trigger and the note survive; the bare acknowledgement is routing
    # bookkeeping and is dropped.
    assert len(fragments) == 2
    assert all(f.incident_id == "INC-2197" for f in fragments)
    assert "ERR_5012" in fragments[0].text
    assert "blocking backend 48211" in fragments[1].text


def test_pagerduty_webhook_normalizes_to_an_alert():
    alert = alert_from_webhook(
        {
            "event": {
                "data": {
                    "id": "PXK7N2QW",
                    "title": "checkout-api 5xx ratio > 25%",
                    "created_at": "2026-06-02T14:12:00Z",
                    "urgency": "high",
                    "service": {"summary": "checkout-api"},
                    "body": {"details": {"host": "db-prod-03", "error_code": "ERR_5012",
                                         "description": "pool timeout"}},
                }
            }
        }
    )
    assert alert.id == "PXK7N2QW"
    assert alert.service == "checkout-api"
    assert alert.host == "db-prod-03"
    assert alert.severity is Severity.SEV1
    assert "ERR_5012" in alert.query_text()


def test_wiki_page_splits_into_sections():
    fragments = parse_wiki_page(REPO_ROOT / "data" / "seed" / "wiki" / "database-runbook.md")
    titles = [f.source.title for f in fragments]
    assert any("Connection pool exhausted" in t for t in titles)
    assert all(f.incident_id is None for f in fragments)
    assert all(f.source.kind is SourceKind.WIKI for f in fragments)


def test_corpus_links_every_source_to_its_incident(corpus):
    assert corpus.counts()["incidents"] == 10
    linked = {f.incident_id for f in corpus.fragments if f.incident_id}
    assert "INC-2197" in linked and "INC-1842" in linked

    incident = next(i for i in corpus.incidents if i.id == "INC-2197")
    kinds = {s.kind for s in incident.sources}
    assert kinds == {SourceKind.POSTMORTEM, SourceKind.SLACK, SourceKind.PAGERDUTY}
