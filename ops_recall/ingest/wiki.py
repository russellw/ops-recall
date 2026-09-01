"""Internal wiki / runbook ingestion.

Runbooks are indexed per section rather than per page: a responder needs "how
do I find the blocking session" now, not a 4000-word database operations page.
Sections keep a `topic` so the agent's `get_runbook` tool can look them up
directly.
"""

from __future__ import annotations

import re
from pathlib import Path

from ops_recall.models import Fragment, SourceKind, SourceRef

_H1 = re.compile(r"^#\s+(.+?)\s*$", re.M)
_SECTION = re.compile(r"^##\s+(.+?)\s*$", re.M)


def parse_wiki_page(path: Path) -> list[Fragment]:
    raw = path.read_text(encoding="utf-8")
    title_match = _H1.search(raw)
    page_title = title_match.group(1).strip() if title_match else path.stem

    fragments: list[Fragment] = []
    matches = list(_SECTION.finditer(raw))
    if not matches:
        matches_body = [(page_title, raw[title_match.end() :] if title_match else raw)]
    else:
        matches_body = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
            matches_body.append((match.group(1).strip(), raw[match.end() : end]))

    for index, (heading, body) in enumerate(matches_body):
        body = body.strip()
        if not body:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
        fragments.append(
            Fragment(
                id=f"wiki:{path.stem}#{slug}",
                incident_id=None,
                text=f"{page_title} - {heading}\n{body}",
                source=SourceRef(
                    kind=SourceKind.WIKI,
                    ref=f"{path.stem}#{slug}",
                    title=f"{page_title}: {heading}",
                    url=f"https://wiki.internal/{path.stem}#{slug}",
                ),
            )
        )
    return fragments


def load_wiki(directory: Path) -> list[Fragment]:
    fragments: list[Fragment] = []
    for path in sorted(Path(directory).glob("*.md")):
        fragments.extend(parse_wiki_page(path))
    return fragments
