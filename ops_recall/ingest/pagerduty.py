"""PagerDuty ingestion.

Log entries are the highest-signal, lowest-noise source in the corpus: they
carry exact fire/ack/resolve timestamps, who was paged, and the raw alert
payload that fired -- which is the closest thing in the archive to the *query*
this system receives at runtime. Indexing them means a new alert can match the
old alert text directly, not just the prose someone wrote about it afterwards.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from ops_recall.models import Alert, Fragment, Severity, SourceKind, SourceRef


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


# Log entry types that carry content worth indexing. The rest -- assignment,
# acknowledgement, notification routing -- is bookkeeping: it has an author and
# a timestamp, so it looks like commentary to a retriever, but it says nothing
# about the incident.
CONTENTFUL_ENTRIES = frozenset(
    {"trigger", "annotate", "resolve", "escalate", "custom", "status_update"}
)


def _entry_kind(entry: dict[str, Any]) -> str:
    return str(entry.get("type", "log_entry")).replace("_log_entry", "")


def is_contentful(entry: dict[str, Any]) -> bool:
    if _entry_kind(entry) in CONTENTFUL_ENTRIES:
        return True
    # An acknowledgement with a note attached is a real observation.
    return bool((entry.get("channel") or {}).get("notes"))


def _summarize_entry(entry: dict[str, Any]) -> str:
    kind = _entry_kind(entry)
    agent = (entry.get("agent") or {}).get("summary", "")
    summary = entry.get("summary") or ""
    channel = entry.get("channel") or {}
    details = channel.get("details")
    parts = [f"[{kind}]", summary]
    if agent:
        parts.append(f"by {agent}")
    if isinstance(details, str) and details:
        parts.append(details)
    elif isinstance(details, dict) and details:
        parts.append(json.dumps(details, sort_keys=True))
    if channel.get("notes"):
        parts.append(f"note: {channel['notes']}")
    return " ".join(p for p in parts if p).strip()


def incident_to_fragments(
    pd_incident: dict[str, Any], incident_id: str | None = None
) -> list[Fragment]:
    """One fragment for the triggering alert, one per meaningful log entry."""
    pd_id = pd_incident.get("id", "unknown")
    incident_id = incident_id or pd_incident.get("incident_key")
    service = (pd_incident.get("service") or {}).get("summary", "")
    created = _parse_time(pd_incident.get("created_at"))

    def ref(suffix: str, when: datetime | None, title: str) -> SourceRef:
        return SourceRef(
            kind=SourceKind.PAGERDUTY,
            ref=f"{pd_id}{suffix}",
            title=title,
            url=pd_incident.get("html_url"),
            timestamp=when,
        )

    fragments = [
        Fragment(
            id=f"pagerduty:{pd_id}:trigger",
            incident_id=incident_id,
            text=" ".join(
                p
                for p in [
                    f"PagerDuty {pd_id} triggered:",
                    pd_incident.get("title", ""),
                    f"service {service}" if service else "",
                    f"urgency {pd_incident.get('urgency', '')}",
                    json.dumps(pd_incident.get("body", {}).get("details", {}), sort_keys=True)
                    if pd_incident.get("body")
                    else "",
                ]
                if p
            ).strip(),
            timestamp=created,
            source=ref("", created, pd_incident.get("title", pd_id)),
        )
    ]

    for index, entry in enumerate(pd_incident.get("log_entries", [])):
        if not is_contentful(entry):
            continue
        text = _summarize_entry(entry)
        if not text:
            continue
        when = _parse_time(entry.get("created_at"))
        fragments.append(
            Fragment(
                id=f"pagerduty:{pd_id}:log:{entry.get('id', index)}",
                incident_id=incident_id,
                text=f"PagerDuty {pd_id} {text}",
                author=(entry.get("agent") or {}).get("summary"),
                timestamp=when,
                source=ref(f"/log/{index}", when, f"{pd_id} log"),
            )
        )
    return fragments


def load_pagerduty_export(
    directory: Path, key_map: dict[str, str] | None = None
) -> list[Fragment]:
    """`key_map` maps PagerDuty incident id -> internal incident id, taken from
    post-mortem frontmatter."""
    key_map = key_map or {}
    fragments: list[Fragment] = []
    for path in sorted(Path(directory).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        incidents = payload if isinstance(payload, list) else payload.get("incidents", [])
        for pd_incident in incidents:
            fragments.extend(
                incident_to_fragments(
                    pd_incident, key_map.get(pd_incident.get("id", ""))
                )
            )
    return fragments


def alert_from_webhook(payload: dict[str, Any]) -> Alert:
    """Normalize a PagerDuty v3 webhook (or a bare incident object) into an
    `Alert`. Everything downstream speaks `Alert`, so this is the only place
    that has to know PagerDuty's shape."""
    event = payload.get("event", payload)
    data = event.get("data", event)

    body_details = (data.get("body") or {}).get("details") or {}
    if isinstance(body_details, str):
        body_details = {"details": body_details}

    labels = {
        str(k): str(v)
        for k, v in body_details.items()
        if isinstance(v, (str, int, float, bool))
    }
    urgency = data.get("urgency", "high")
    return Alert(
        id=data.get("id") or event.get("id") or "pd-unknown",
        title=data.get("title", "PagerDuty alert"),
        description=str(body_details.get("description") or body_details.get("details") or ""),
        service=(data.get("service") or {}).get("summary"),
        host=labels.get("host") or labels.get("instance"),
        severity=Severity.coerce("critical" if urgency == "high" else "warning"),
        fired_at=_parse_time(data.get("created_at")) or datetime.now(timezone.utc),
        labels=labels,
        raw=payload,
    )


class PagerDutyClient:
    """Live REST client. Auth is a REST API key: `Authorization: Token token=...`."""

    BASE = "https://api.pagerduty.com"

    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        if not api_key:
            raise ValueError("PagerDuty ingestion needs a REST API key")
        self._api_key = api_key
        self._timeout = timeout

    def _get(self, path: str, **params: Any) -> dict[str, Any]:
        import httpx

        response = httpx.get(
            f"{self.BASE}{path}",
            headers={
                "Authorization": f"Token token={self._api_key}",
                "Accept": "application/vnd.pagerduty+json;version=2",
            },
            params={k: v for k, v in params.items() if v is not None},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    def incidents(
        self, since: datetime | None = None, statuses: Sequence[str] = ("resolved",)
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        offset = 0
        while True:
            payload = self._get(
                "/incidents",
                since=since.isoformat() if since else None,
                limit=100,
                offset=offset,
                **{"statuses[]": list(statuses)},
            )
            results.extend(payload.get("incidents", []))
            if not payload.get("more"):
                return results
            offset += 100

    def log_entries(self, incident_id: str) -> list[dict[str, Any]]:
        payload = self._get(f"/incidents/{incident_id}/log_entries", is_overview=False, limit=100)
        return payload.get("log_entries", [])

    def ingest(
        self,
        since: datetime | None = None,
        key_map: dict[str, str] | None = None,
    ) -> list[Fragment]:
        key_map = key_map or {}
        fragments: list[Fragment] = []
        for pd_incident in self.incidents(since=since):
            pd_incident = dict(pd_incident)
            pd_incident["log_entries"] = self.log_entries(pd_incident["id"])
            fragments.extend(
                incident_to_fragments(pd_incident, key_map.get(pd_incident["id"]))
            )
        return fragments


def iter_paths(directory: Path) -> Iterable[Path]:
    return sorted(Path(directory).glob("*.json"))
