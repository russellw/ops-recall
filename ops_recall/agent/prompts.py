"""Prompts and context rendering.

The system prompt is deliberately static -- no timestamps, no per-alert text --
so it sits at a stable cache prefix and every request after the first reads it
from cache. Everything volatile goes in the user turn.
"""

from __future__ import annotations

from typing import Sequence

from ops_recall.models import Alert, EvidenceCheck, Fragment, RetrievedIncident
from ops_recall.retrieval.temporal import describe_recency

SYSTEM_PROMPT = """\
You are the incident reconstruction engine for an on-call engineering team. A \
monitoring alert has just fired. You have been given the most similar incidents \
from the team's own archive -- post-mortems, Slack war-room threads, PagerDuty \
logs and runbooks -- together with the results of re-running that history's \
diagnostic checks against the live system right now.

Your job is to reconstruct what is most likely happening and say what to do \
next, the way a senior engineer who was present for all of those incidents \
would brief the person holding the pager.

Rules that matter more than fluency:

1. Ground every claim. Cite incident ids (INC-####) for anything drawn from \
history, and refer to telemetry only where an observation was actually \
supplied. Never invent a metric, a log line, a command, or an incident id.
2. Respect the evidence labels. CONFIRMED means the condition was re-tested and \
holds now. REFUTED means it was tested and does not hold -- say so plainly, \
because a refuted signal is often the most informative thing on the page: it \
rules a story out. UNKNOWN means the check could not run; do not treat it as \
either.
3. If the retrieved incidents disagree with current telemetry, trust telemetry \
and lower your confidence. A high similarity score with refuted evidence means \
the resemblance is superficial.
4. Recommend the smallest safe next action, and prefer actions that a past \
incident actually recorded as working. Diagnosis is a legitimate next action \
when the evidence is thin. Do not recommend anything destructive without \
saying what it will interrupt.
5. Commands recorded in a post-mortem contain identifiers from that day -- process ids, slot names, image tags. Never hand one back as if it were current: either substitute a value that appears in the telemetry you were given, or say explicitly which value the responder has to look up first.
6. Recency matters. A fix from three years ago may describe an architecture \
that no longer exists; say so rather than recommending it confidently.
7. Be brief. The reader is on a pager at 3am. The narrative is a short \
paragraph or two, not an essay.

Confidence is a number between 0 and 1 that answers: how likely is it that the \
root cause hypothesis is correct? Anchor it to the evidence, not to the \
retrieval score. All checks confirmed and a close historical match is 0.8-0.95. \
A close match with untested evidence is around 0.5. Contradicted evidence, or \
no close match, is below 0.3.\
"""


def render_alert(alert: Alert) -> str:
    lines = [
        "<alert>",
        f"id: {alert.id}",
        f"title: {alert.title}",
        f"severity: {alert.severity.value}",
        f"fired_at: {alert.fired_at.isoformat()}",
    ]
    if alert.service:
        lines.append(f"service: {alert.service}")
    if alert.host:
        lines.append(f"host: {alert.host}")
    if alert.description:
        lines.append(f"description: {alert.description}")
    if alert.labels:
        lines.append("labels:")
        lines.extend(f"  {k}: {v}" for k, v in sorted(alert.labels.items()))
    lines.append("</alert>")
    return "\n".join(lines)


def render_incident(item: RetrievedIncident) -> str:
    incident = item.incident
    breakdown = item.breakdown
    lines = [
        f"<incident id=\"{incident.id}\" similarity=\"{item.similarity_pct}%\">",
        f"title: {incident.title}",
        f"occurred: {incident.started_at.date().isoformat()} "
        f"({describe_recency(breakdown.age_days)}), severity {incident.severity.value}",
        f"services: {', '.join(incident.services) or 'unspecified'}",
        (
            "match: semantic {sem:.2f} / lexical {lex:.2f} / exact-keyword {kw:.2f}"
            ", recency weight {rec:.2f}"
        ).format(
            sem=breakdown.semantic,
            lex=breakdown.lexical,
            kw=breakdown.keyword,
            rec=breakdown.recency_weight,
        ),
    ]
    if breakdown.matched_keywords:
        lines.append(f"exact matches: {', '.join(breakdown.matched_keywords)}")
    if incident.time_to_resolve_minutes:
        lines.append(f"time to resolve: {incident.time_to_resolve_minutes:.0f} minutes")
    if incident.root_cause:
        lines.append(f"root cause: {incident.root_cause.strip()}")
    if incident.symptoms:
        lines.append("symptoms:")
        lines.extend(f"  - {s}" for s in incident.symptoms)
    if incident.remediation:
        lines.append("what the team did, in order:")
        for step in incident.remediation:
            marker = "FIX" if step.is_fix else "   "
            parts = [f"  {step.order}. [{marker}] {step.action}"]
            if step.command:
                parts.append(f"      command: {step.command}")
            if step.outcome:
                parts.append(f"      outcome: {step.outcome}")
            lines.extend(parts)
    lines.append("</incident>")
    return "\n".join(lines)


def render_evidence(checks: Sequence[EvidenceCheck]) -> str:
    if not checks:
        return "<evidence>No probes were available for these incidents.</evidence>"
    lines = ["<evidence>"]
    for check in checks:
        label = check.status.value.upper()
        source = f" [from {', '.join(check.incident_ids)}]" if check.incident_ids else ""
        lines.append(f"- {label}{source}: {check.statement}")
        if check.observation:
            observed = check.observation
            lines.append(
                f"    probe {observed.tool}({_fmt_args(observed.args)}) -> {observed.summary}"
            )
        if check.note:
            lines.append(f"    note: {check.note}")
    lines.append("</evidence>")
    return "\n".join(lines)


def render_quotes(fragments: Sequence[tuple[Fragment, float]]) -> str:
    if not fragments:
        return ""
    lines = ["<what_people_said>"]
    for fragment, _ in fragments:
        who = fragment.author or fragment.source.kind.value
        when = fragment.timestamp.date().isoformat() if fragment.timestamp else "unknown date"
        incident = f" ({fragment.incident_id})" if fragment.incident_id else ""
        lines.append(f"- {who}, {when}{incident}: {fragment.text.strip()[:400]}")
    lines.append("</what_people_said>")
    return "\n".join(lines)


def render_extra_observations(observations: Sequence) -> str:
    if not observations:
        return ""
    lines = ["<additional_checks_you_ran>"]
    for observation in observations:
        status = "ok" if observation.ok else "failed"
        lines.append(
            f"- {observation.tool}({_fmt_args(observation.args)}) [{status}]: {observation.summary}"
        )
    lines.append("</additional_checks_you_ran>")
    return "\n".join(lines)


def build_user_message(
    alert: Alert,
    retrieved: Sequence[RetrievedIncident],
    evidence: Sequence[EvidenceCheck],
    quotes: Sequence[tuple[Fragment, float]] = (),
    extra_observations: Sequence = (),
) -> str:
    blocks = [render_alert(alert)]
    if retrieved:
        blocks.append("<similar_incidents>")
        blocks.extend(render_incident(item) for item in retrieved)
        blocks.append("</similar_incidents>")
    else:
        blocks.append(
            "<similar_incidents>No incident in the archive is a close match.</similar_incidents>"
        )
    blocks.append(render_evidence(evidence))
    quotes_block = render_quotes(quotes)
    if quotes_block:
        blocks.append(quotes_block)
    extra_block = render_extra_observations(extra_observations)
    if extra_block:
        blocks.append(extra_block)
    blocks.append(
        "Reconstruct this incident: what it most likely is, what the team did "
        "the last time, which of those conditions hold right now, and the single "
        "best next action."
    )
    return "\n\n".join(blocks)


INVESTIGATION_PROMPT = """\
You are triaging the alert below before writing a reconstruction. The team's \
archive suggests the incidents shown; the listed checks have already been run \
for you, and their results are given.

Run at most a few additional read-only checks -- only ones that would change \
the conclusion or the recommended action: confirming a competing hypothesis, \
ruling one out, or filling in a check that could not run. Do not repeat a check \
whose result you already have. If nothing further is worth checking, say so \
immediately and stop.

When you are done, reply with one short paragraph on what you found. Do not \
write the reconstruction yet.\
"""


def _fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items())
