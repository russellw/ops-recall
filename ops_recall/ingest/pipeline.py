"""Corpus assembly: four sources in, one searchable index out.

The correlation step is the interesting part. Post-mortems declare their
PagerDuty incident and Slack thread in frontmatter; those declarations become
the key map that lets thousands of unlabeled Slack messages and PagerDuty log
entries attach themselves to the right incident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ops_recall.config import Settings, get_settings
from ops_recall.ingest.pagerduty import load_pagerduty_export
from ops_recall.ingest.postmortems import (
    frontmatter_links,
    iter_postmortem_paths,
    load_postmortems,
)
from ops_recall.ingest.slack import load_slack_export
from ops_recall.ingest.wiki import load_wiki
from ops_recall.models import Fragment, Incident, SourceKind, SourceRef
from ops_recall.retrieval.store import IncidentStore


@dataclass
class Corpus:
    incidents: list[Incident] = field(default_factory=list)
    fragments: list[Fragment] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        by_source: dict[str, int] = {}
        for fragment in self.fragments:
            key = fragment.source.kind.value
            by_source[key] = by_source.get(key, 0) + 1
        return {
            "incidents": len(self.incidents),
            "fragments": len(self.fragments),
            **by_source,
        }


def _cross_source_maps(postmortem_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    """(slack thread -> incident id, pagerduty id -> incident id)."""
    thread_index: dict[str, str] = {}
    pd_map: dict[str, str] = {}
    for path in iter_postmortem_paths(postmortem_dir):
        links = frontmatter_links(path)
        incident_id = links.get("id")
        if not incident_id:
            continue
        if links.get("slack_thread"):
            thread_index[str(links["slack_thread"])] = str(incident_id)
        if links.get("pagerduty_incident"):
            pd_map[str(links["pagerduty_incident"])] = str(incident_id)
    return thread_index, pd_map


def load_live_fragments(
    settings: Settings, thread_index: dict[str, str], pd_map: dict[str, str]
) -> list[Fragment]:
    """Pull from the Slack and PagerDuty APIs when credentials are configured.

    The exports on disk and the live APIs produce the same `Fragment` objects,
    so a team can start from a one-off export and switch to continuous ingest
    without touching anything downstream.
    """
    fragments: list[Fragment] = []
    if settings.slack_bot_token and settings.slack_channels:
        from ops_recall.ingest.slack import SlackClient

        fragments += SlackClient(settings.slack_bot_token).ingest(
            settings.slack_channels, thread_index=thread_index
        )
    if settings.pagerduty_api_key:
        from ops_recall.ingest.pagerduty import PagerDutyClient

        fragments += PagerDutyClient(settings.pagerduty_api_key).ingest(key_map=pd_map)
    return fragments


def load_corpus(settings: Settings | None = None, live: bool = False) -> Corpus:
    settings = settings or get_settings()
    root = Path(settings.data_dir)

    postmortem_dir = root / "postmortems"
    incidents = load_postmortems(postmortem_dir) if postmortem_dir.is_dir() else []
    thread_index, pd_map = _cross_source_maps(postmortem_dir)

    fragments: list[Fragment] = []
    if (root / "slack").is_dir():
        fragments += load_slack_export(root / "slack", thread_index=thread_index)
    if (root / "pagerduty").is_dir():
        fragments += load_pagerduty_export(root / "pagerduty", key_map=pd_map)
    if (root / "wiki").is_dir():
        fragments += load_wiki(root / "wiki")
    if live:
        fragments += load_live_fragments(settings, thread_index, pd_map)

    # A message can arrive from both an export and the live API; the fragment id
    # is derived from the source system's own ids, so deduplication is exact.
    fragments = list({fragment.id: fragment for fragment in fragments}.values())

    corpus = Corpus(incidents=incidents, fragments=fragments)
    attach_source_refs(corpus)
    return corpus


def attach_source_refs(corpus: Corpus) -> None:
    """Give each incident the list of sources that mention it, deduplicated by
    source system, so a reconstruction can cite provenance without walking the
    whole fragment set."""
    by_incident: dict[str, dict[SourceKind, SourceRef]] = {}
    for fragment in corpus.fragments:
        if not fragment.incident_id:
            continue
        by_incident.setdefault(fragment.incident_id, {}).setdefault(
            fragment.source.kind, fragment.source
        )
    for incident in corpus.incidents:
        for ref in by_incident.get(incident.id, {}).values():
            incident.sources.append(ref)


def build_index(
    settings: Settings | None = None,
    store: IncidentStore | None = None,
    live: bool = False,
) -> tuple[IncidentStore, dict[str, int]]:
    settings = settings or get_settings()
    corpus = load_corpus(settings, live=live)
    store = store or IncidentStore(settings)
    store.index(corpus.incidents, corpus.fragments, recreate=True)
    return store, corpus.counts()
