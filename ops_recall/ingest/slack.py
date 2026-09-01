"""Slack ingestion.

Two entry points with the same output shape:

* `load_slack_export(dir)` -- reads `conversations.history` JSON dumps from
  disk. This is what the seed corpus and the tests use, and what most teams
  can produce without wiring an app.
* `SlackClient.fetch_channel(...)` -- the live path against Slack's Web API.

Incident correlation is by explicit reference: a message is attached to an
incident when it names it (`INC-2197`) or when its thread is declared in a
post-mortem's frontmatter. Guessing by timestamp proximity was tempting and is
wrong -- ops channels run several conversations at once.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from ops_recall.models import Fragment, SourceKind, SourceRef

INCIDENT_REF = re.compile(r"\b(INC-\d+)\b", re.I)

# Slack markup that adds nothing once the text is embedded.
_USER_MENTION = re.compile(r"<@([UW][A-Z0-9]+)(?:\|([^>]+))?>")
_LINK = re.compile(r"<(https?://[^|>]+)(?:\|([^>]+))?>")
_CHANNEL = re.compile(r"<#[CG][A-Z0-9]+\|([^>]+)>")


def clean_slack_text(text: str, users: dict[str, str] | None = None) -> str:
    users = users or {}
    text = _USER_MENTION.sub(lambda m: "@" + (m.group(2) or users.get(m.group(1), m.group(1))), text)
    text = _LINK.sub(lambda m: m.group(2) or m.group(1), text)
    text = _CHANNEL.sub(lambda m: "#" + m.group(1), text)
    return text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").strip()


def _ts_to_datetime(ts: str) -> datetime:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc)


def messages_to_fragments(
    channel_id: str,
    channel_name: str,
    messages: Sequence[dict[str, Any]],
    users: dict[str, str] | None = None,
    thread_index: dict[str, str] | None = None,
) -> list[Fragment]:
    """Convert raw Slack messages into fragments, resolving incident links.

    `thread_index` maps `"<channel>/<thread_ts>"` to an incident id, so replies
    in a declared incident thread inherit the link even when they never name
    the incident.
    """
    users = users or {}
    thread_index = thread_index or {}
    fragments: list[Fragment] = []

    for message in messages:
        if message.get("subtype") in {"channel_join", "channel_leave"}:
            continue
        text = clean_slack_text(message.get("text", ""), users)
        if not text:
            continue
        ts = str(message.get("ts"))
        thread_ts = str(message.get("thread_ts") or ts)

        incident_id = thread_index.get(f"{channel_id}/{thread_ts}")
        if incident_id is None:
            match = INCIDENT_REF.search(text)
            incident_id = match.group(1).upper() if match else None

        author = users.get(message.get("user", ""), message.get("user_name") or message.get("user"))
        fragments.append(
            Fragment(
                id=f"slack:{channel_id}:{ts}",
                incident_id=incident_id,
                text=text,
                author=author,
                timestamp=_ts_to_datetime(ts),
                source=SourceRef(
                    kind=SourceKind.SLACK,
                    ref=f"{channel_id}/{ts}",
                    title=f"#{channel_name}",
                    url=f"https://slack.com/archives/{channel_id}/p{ts.replace('.', '')}",
                    timestamp=_ts_to_datetime(ts),
                ),
            )
        )
    return fragments


def load_slack_export(
    directory: Path, thread_index: dict[str, str] | None = None
) -> list[Fragment]:
    fragments: list[Fragment] = []
    for path in sorted(Path(directory).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        channel = payload.get("channel", {})
        users = {u["id"]: u.get("name", u["id"]) for u in payload.get("users", [])}
        fragments.extend(
            messages_to_fragments(
                channel.get("id", path.stem),
                channel.get("name", path.stem),
                payload.get("messages", []),
                users=users,
                thread_index=thread_index,
            )
        )
    return fragments


class SlackClient:
    """Minimal Web API client for the live ingest path.

    Scopes needed: `channels:history`, `groups:history`, `users:read`.
    """

    BASE = "https://slack.com/api"

    def __init__(self, token: str, timeout: float = 30.0) -> None:
        if not token:
            raise ValueError("Slack ingestion needs a bot token")
        self._token = token
        self._timeout = timeout

    def _call(self, method: str, **params: Any) -> dict[str, Any]:
        import httpx

        response = httpx.get(
            f"{self.BASE}/{method}",
            headers={"Authorization": f"Bearer {self._token}"},
            params={k: v for k, v in params.items() if v is not None},
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(f"Slack {method} failed: {payload.get('error')}")
        return payload

    def users(self) -> dict[str, str]:
        names: dict[str, str] = {}
        cursor = None
        while True:
            payload = self._call("users.list", cursor=cursor, limit=200)
            for user in payload.get("members", []):
                names[user["id"]] = user.get("profile", {}).get("display_name") or user.get("name", user["id"])
            cursor = payload.get("response_metadata", {}).get("next_cursor") or None
            if not cursor:
                return names

    def fetch_channel(
        self,
        channel_id: str,
        oldest: datetime | None = None,
        include_threads: bool = True,
    ) -> list[dict[str, Any]]:
        """Channel history plus thread replies, oldest-first."""
        messages: list[dict[str, Any]] = []
        cursor = None
        while True:
            payload = self._call(
                "conversations.history",
                channel=channel_id,
                cursor=cursor,
                limit=200,
                oldest=oldest.timestamp() if oldest else None,
            )
            messages.extend(payload.get("messages", []))
            cursor = payload.get("response_metadata", {}).get("next_cursor") or None
            if not cursor:
                break

        if include_threads:
            parents = [m for m in messages if int(m.get("reply_count", 0)) > 0]
            for parent in parents:
                payload = self._call(
                    "conversations.replies", channel=channel_id, ts=parent["ts"], limit=200
                )
                # The parent is repeated in the replies payload; skip it.
                messages.extend(m for m in payload.get("messages", []) if m["ts"] != parent["ts"])

        return sorted(messages, key=lambda m: float(m["ts"]))

    def ingest(
        self,
        channels: Iterable[str],
        oldest: datetime | None = None,
        thread_index: dict[str, str] | None = None,
    ) -> list[Fragment]:
        users = self.users()
        fragments: list[Fragment] = []
        for channel_id in channels:
            info = self._call("conversations.info", channel=channel_id).get("channel", {})
            fragments.extend(
                messages_to_fragments(
                    channel_id,
                    info.get("name", channel_id),
                    self.fetch_channel(channel_id, oldest=oldest),
                    users=users,
                    thread_index=thread_index,
                )
            )
        return fragments
