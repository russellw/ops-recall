"""Turning retrieval + evidence into a reconstruction.

Two implementations of one interface:

* `ClaudeReconstructor` -- the real one. Claude writes the narrative, the root
  cause hypothesis, and the recommended actions, constrained to a schema
  (`ReconstructionDraft`) that deliberately excludes anything the system already
  knows: similarity scores, evidence statuses, incident ids and timestamps are
  merged in afterwards, so the model cannot invent a retrieval score or claim a
  telemetry result that was never observed.
* `TemplateReconstructor` -- deterministic assembly, no model call. It exists so
  the retrieval and evidence layers can be tested end to end, and so a
  degraded-mode answer is available when the API is unreachable. Its output is
  labeled `reasoner="template"` and never silently substituted for the model.
"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence

from ops_recall.agent.prompts import SYSTEM_PROMPT, build_user_message
from ops_recall.config import Settings
from ops_recall.models import (
    Alert,
    CheckStatus,
    EvidenceCheck,
    Fragment,
    Reconstruction,
    ReconstructionDraft,
    RecommendedAction,
    RetrievedIncident,
    SimilarIncidentBrief,
    TelemetryObservation,
)
from ops_recall.retrieval.temporal import describe_recency

logger = logging.getLogger(__name__)

DESTRUCTIVE_HINTS = (
    "terminate",
    "kill",
    "drop ",
    "delete",
    "restart",
    "roll back",
    "rollback",
    "failover",
    "truncate",
    "cancel",
)


class ReconstructionError(RuntimeError):
    pass


class Reconstructor(Protocol):
    name: str

    def reconstruct(
        self,
        alert: Alert,
        retrieved: Sequence[RetrievedIncident],
        evidence: Sequence[EvidenceCheck],
        quotes: Sequence[tuple[Fragment, float]] = (),
        extra_observations: Sequence[TelemetryObservation] = (),
    ) -> Reconstruction: ...


def briefs(retrieved: Sequence[RetrievedIncident]) -> list[SimilarIncidentBrief]:
    return [
        SimilarIncidentBrief(
            incident_id=item.incident.id,
            title=item.incident.title,
            similarity_pct=item.similarity_pct,
            occurred=item.incident.started_at,
            root_cause=item.incident.root_cause,
            actions_taken=[step.action for step in item.incident.remediation],
            resolution_minutes=item.incident.time_to_resolve_minutes,
        )
        for item in retrieved
    ]


def _risk_for(text: str) -> str:
    lowered = text.lower()
    return "high" if any(hint in lowered for hint in DESTRUCTIVE_HINTS) else "low"


class ClaudeReconstructor:
    name = "claude"

    def __init__(self, client, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    def reconstruct(
        self,
        alert: Alert,
        retrieved: Sequence[RetrievedIncident],
        evidence: Sequence[EvidenceCheck],
        quotes: Sequence[tuple[Fragment, float]] = (),
        extra_observations: Sequence[TelemetryObservation] = (),
    ) -> Reconstruction:
        user_message = build_user_message(
            alert, retrieved, evidence, quotes, extra_observations
        )
        response = self.client.messages.parse(
            model=self.settings.model,
            max_tokens=self.settings.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    # The system prompt is identical on every alert, so it sits
                    # at a stable cache prefix.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_message}],
            output_format=ReconstructionDraft,
        )
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise ReconstructionError(
                f"model declined to answer ({getattr(details, 'category', 'unknown')})"
            )
        draft: ReconstructionDraft | None = getattr(response, "parsed_output", None)
        if draft is None:
            raise ReconstructionError("model returned no parsable reconstruction")

        return Reconstruction(
            alert_id=alert.id,
            headline=draft.headline,
            similar_incidents=briefs(retrieved),
            root_cause_hypothesis=draft.root_cause_hypothesis,
            narrative=draft.narrative,
            evidence=list(evidence),
            recommended_actions=draft.recommended_actions,
            confidence=draft.confidence,
            caveats=draft.caveats,
            reasoner=self.name,
        )


class TemplateReconstructor:
    """Deterministic reconstruction. No model call, no invention -- it can only
    restate retrieved facts and observed telemetry."""

    name = "template"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings

    def reconstruct(
        self,
        alert: Alert,
        retrieved: Sequence[RetrievedIncident],
        evidence: Sequence[EvidenceCheck],
        quotes: Sequence[tuple[Fragment, float]] = (),
        extra_observations: Sequence[TelemetryObservation] = (),
    ) -> Reconstruction:
        confirmed = [c for c in evidence if c.status is CheckStatus.CONFIRMED]
        refuted = [c for c in evidence if c.status is CheckStatus.REFUTED]

        if not retrieved:
            return Reconstruction(
                alert_id=alert.id,
                headline=f"No close historical match for {alert.title}.",
                narrative=(
                    "Nothing in the incident archive resembles this alert closely "
                    "enough to reconstruct from. Treat it as novel and follow "
                    "standard triage."
                ),
                evidence=list(evidence),
                confidence=0.05,
                caveats=["No incident scored above the similarity threshold."],
                reasoner=self.name,
            )

        best = retrieved[0]
        others = retrieved[1:]
        ids = ", ".join(
            f"{item.incident.id} ({item.similarity_pct}%)" for item in retrieved
        )
        headline = (
            f"This alert looks {best.similarity_pct}% similar to "
            f"{best.incident.id}: {best.incident.title}."
        )

        fix_steps = best.incident.fix_steps()
        paragraphs = [
            f"This alert looks similar to {ids}. "
            f"{best.incident.id} happened "
            f"{describe_recency(best.breakdown.age_days)}"
            + (
                f" and took {best.incident.time_to_resolve_minutes:.0f} minutes to resolve"
                if best.incident.time_to_resolve_minutes
                else ""
            )
            + f". Root cause: {best.incident.root_cause.strip()}"
        ]
        if best.incident.remediation:
            actions = "; ".join(
                f"{step.order}. {step.action}" for step in best.incident.remediation
            )
            paragraphs.append(f"What the team did in {best.incident.id}: {actions}")
        if others:
            paragraphs.append(
                "Also similar: "
                + "; ".join(
                    f"{item.incident.id} ({item.similarity_pct}%, "
                    f"{describe_recency(item.breakdown.age_days)}) - "
                    f"{item.incident.root_cause.strip()[:180]}"
                    for item in others
                )
            )
        if confirmed:
            paragraphs.append(
                "Current telemetry confirms: "
                + " ".join(
                    f"{c.statement} ({c.observation.summary if c.observation else 'observed'})"
                    for c in confirmed
                )
            )
        if refuted:
            paragraphs.append(
                "Current telemetry does NOT show: "
                + " ".join(
                    f"{c.statement} ({c.observation.summary if c.observation else 'not observed'})"
                    for c in refuted
                )
            )

        recommended = [
            RecommendedAction(
                action=step.action,
                rationale=(
                    f"This is what resolved {best.incident.id}"
                    + (f" ({step.outcome})" if step.outcome else "")
                ),
                command=step.command,
                from_incident=best.incident.id,
                risk=_risk_for(f"{step.action} {step.command or ''}"),
                requires_human_approval=True,
            )
            for step in fix_steps[:2]
        ]

        testable = [c for c in evidence if c.status is not CheckStatus.UNKNOWN]
        evidence_ratio = (len(confirmed) / len(testable)) if testable else 0.0
        confidence = round(
            best.similarity * (0.4 + 0.6 * evidence_ratio) if testable else best.similarity * 0.4,
            2,
        )

        caveats = [
            "Generated without the language model: facts are restated, not reasoned about."
        ]
        if any(action.command for action in recommended):
            caveats.append(
                "Commands are copied verbatim from the historical incident. "
                "Identifiers inside them -- pids, slot names, versions -- are the "
                "old ones; re-derive them from current output before running."
            )
        if not testable:
            caveats.append("No signal could be re-tested against live telemetry.")
        if refuted:
            caveats.append(
                f"{len(refuted)} historical signal(s) do not hold now, so the "
                "resemblance may be superficial."
            )
        if best.breakdown.age_days > 365:
            caveats.append(
                f"{best.incident.id} is "
                f"{describe_recency(best.breakdown.age_days)}; the system may have changed."
            )

        return Reconstruction(
            alert_id=alert.id,
            headline=headline,
            similar_incidents=briefs(retrieved),
            root_cause_hypothesis=best.incident.root_cause.strip(),
            narrative="\n\n".join(paragraphs),
            evidence=list(evidence),
            recommended_actions=recommended,
            confidence=confidence,
            caveats=caveats,
            reasoner=self.name,
        )


def build_reconstructor(settings: Settings, client=None) -> Reconstructor:
    if settings.reasoner == "template":
        return TemplateReconstructor(settings)
    if client is None:
        raise ReconstructionError(
            "reasoner='claude' needs an Anthropic client; set ANTHROPIC_API_KEY "
            "(or run `ant auth login`), or set OPS_RECALL_REASONER=template for "
            "the deterministic fallback."
        )
    return ClaudeReconstructor(client, settings)
