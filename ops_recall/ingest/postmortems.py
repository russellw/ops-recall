"""Parse post-mortem markdown into `Incident` records.

The format is the one most teams already write: YAML frontmatter for the facts
a machine needs, prose for everything else. Two things are extracted with more
care than usual:

* **Remediation steps** keep their order and are tagged `[fix]` / `[mitigate]` /
  `[diagnose]`, because "what finally worked" and "what we tried first" are
  different answers to a page at 3am.
* **Signals** are the conditions that held during the incident, each bound to a
  telemetry probe so the same condition can be re-tested against the live
  system later.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

from ops_recall.models import (
    Incident,
    RemediationStep,
    Severity,
    Signal,
    SourceKind,
    SourceRef,
)
from ops_recall.retrieval.text import extract_identifiers

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.S)
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.M)
_STEP = re.compile(r"^\s*\d+\.\s+(.*)$")
_TAG = re.compile(r"^\[(fix|mitigate|diagnose|rollback)\]\s*", re.I)
_COMMAND = re.compile(r"`([^`]+)`")
_ACTOR = re.compile(r"\s+[-—]{1,2}\s+@([\w.-]+)")
_OUTCOME = re.compile(r"\(outcome:\s*(.+?)\)\s*$", re.I)


class PostmortemParseError(ValueError):
    pass


def normalize_prose(text: str) -> str:
    """Unwrap hard-wrapped markdown paragraphs.

    Post-mortems are written wrapped at 80 columns. Those line breaks are an
    artifact of the editor, and carrying them into an API response or a
    terminal render makes the text ragged, so paragraphs are rejoined and only
    blank-line breaks survive.
    """
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return "\n\n".join(" ".join(p.split()) for p in paragraphs if p.strip())


def split_sections(body: str) -> dict[str, str]:
    """Map lowercased heading -> section text."""
    sections: dict[str, str] = {}
    matches = list(_HEADING.finditer(body))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections[match.group(1).strip().lower()] = body[start:end].strip()
    return sections


def parse_remediation(section: str) -> list[RemediationStep]:
    steps: list[RemediationStep] = []
    for line in section.splitlines():
        match = _STEP.match(line)
        if not match:
            continue
        text = match.group(1).strip()

        tag_match = _TAG.match(text)
        tag = (tag_match.group(1).lower() if tag_match else "diagnose")
        text = _TAG.sub("", text)

        outcome_match = _OUTCOME.search(text)
        outcome = outcome_match.group(1).strip() if outcome_match else None
        text = _OUTCOME.sub("", text).strip()

        actor_match = _ACTOR.search(text)
        actor = actor_match.group(1) if actor_match else None
        text = _ACTOR.sub("", text).strip()

        commands = _COMMAND.findall(text)
        command = commands[0].strip() if commands else None
        # A single trailing code span is the step's command and lives in its own
        # field, so drop it from the prose rather than repeating a whole SQL
        # statement. Code spans woven into a sentence stay where they are.
        action = text
        if len(commands) == 1 and text.rstrip().endswith("`"):
            action = _COMMAND.sub("", text)
        action = action.replace("`", "").strip().rstrip(" ,:.")

        steps.append(
            RemediationStep(
                order=len(steps) + 1,
                action=action + ("." if action else ""),
                command=command,
                actor=actor,
                outcome=outcome,
                is_fix=tag == "fix",
            )
        )
    return steps


def _bullets(section: str) -> list[str]:
    return [
        line.strip().lstrip("-*").strip()
        for line in section.splitlines()
        if line.strip().startswith(("-", "*"))
    ]


def _parse_time(value: Any, field: str, path: Path) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    raise PostmortemParseError(f"{path}: `{field}` must be a timestamp")


def parse_postmortem(path: Path) -> Incident:
    raw = path.read_text(encoding="utf-8")
    match = _FRONTMATTER.match(raw)
    if not match:
        raise PostmortemParseError(f"{path}: missing YAML frontmatter")
    meta: dict[str, Any] = yaml.safe_load(match.group(1)) or {}
    body = match.group(2)
    sections = split_sections(body)

    if not meta.get("id"):
        raise PostmortemParseError(f"{path}: frontmatter needs an `id`")

    signals = [
        Signal(
            id=item["id"],
            statement=item["statement"],
            probe=item.get("probe"),
            probe_args=item.get("probe_args") or {},
        )
        for item in meta.get("signals") or []
    ]

    summary = normalize_prose(sections.get("summary", ""))
    root_cause = normalize_prose(sections.get("root cause", ""))
    symptoms = _bullets(sections.get("symptoms", ""))
    remediation = parse_remediation(sections.get("remediation", ""))

    sources = [
        SourceRef(
            kind=SourceKind.POSTMORTEM,
            ref=path.name,
            title=meta.get("title", path.stem),
            url=meta.get("url"),
        )
    ]

    declared = [str(k) for k in (meta.get("keywords") or [])]
    discovered = extract_identifiers(
        "\n".join([meta.get("title", ""), summary, root_cause, *symptoms])
    )
    keywords = list(dict.fromkeys([k.lower() for k in declared] + discovered))

    return Incident(
        id=str(meta["id"]),
        title=meta.get("title", path.stem),
        summary=summary,
        root_cause=root_cause,
        severity=Severity.coerce(meta.get("severity")),
        services=[str(s) for s in (meta.get("services") or [])],
        started_at=_parse_time(meta.get("started_at"), "started_at", path),
        resolved_at=(
            _parse_time(meta["resolved_at"], "resolved_at", path)
            if meta.get("resolved_at")
            else None
        ),
        symptoms=symptoms,
        keywords=keywords,
        signals=signals,
        remediation=remediation,
        sources=sources,
        tags=[str(t) for t in (meta.get("tags") or [])],
    )


def load_postmortems(directory: Path) -> list[Incident]:
    incidents: list[Incident] = []
    for path in sorted(Path(directory).glob("*.md")):
        incidents.append(parse_postmortem(path))
    return incidents


def frontmatter_links(path: Path) -> dict[str, Any]:
    """Cross-source ids declared by a post-mortem (`pagerduty_incident`,
    `slack_thread`), used to attach fragments to the right incident."""
    match = _FRONTMATTER.match(path.read_text(encoding="utf-8"))
    if not match:
        return {}
    meta = yaml.safe_load(match.group(1)) or {}
    return {
        "id": meta.get("id"),
        "pagerduty_incident": meta.get("pagerduty_incident"),
        "slack_thread": meta.get("slack_thread"),
    }


def iter_postmortem_paths(directory: Path) -> Iterable[Path]:
    return sorted(Path(directory).glob("*.md"))
