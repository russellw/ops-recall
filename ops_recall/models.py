"""Domain models shared by ingestion, retrieval, the agent, and the API.

The vocabulary here is deliberately incident-shaped rather than document-shaped:
the system's job is not "find relevant text" but "reconstruct what happened last
time and check whether it is happening again".
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    """Normalize to an aware UTC datetime so age arithmetic never crashes."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class Severity(str, Enum):
    SEV1 = "sev1"
    SEV2 = "sev2"
    SEV3 = "sev3"
    SEV4 = "sev4"

    @classmethod
    def coerce(cls, value: Any) -> "Severity":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower().replace("-", "").replace("_", "")
        aliases = {
            "critical": cls.SEV1,
            "p1": cls.SEV1,
            "high": cls.SEV2,
            "p2": cls.SEV2,
            "warning": cls.SEV3,
            "medium": cls.SEV3,
            "p3": cls.SEV3,
            "low": cls.SEV4,
            "info": cls.SEV4,
            "p4": cls.SEV4,
        }
        if text in aliases:
            return aliases[text]
        try:
            return cls(text)
        except ValueError:
            return cls.SEV3


class SourceKind(str, Enum):
    POSTMORTEM = "postmortem"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WIKI = "wiki"


class SourceRef(BaseModel):
    """Where a piece of an incident record came from, for citation."""

    kind: SourceKind
    ref: str = Field(description="Native id in the source system (thread ts, PD id, page slug).")
    title: str = ""
    url: str | None = None
    timestamp: datetime | None = None

    @field_validator("timestamp")
    @classmethod
    def _tz(cls, v: datetime | None) -> datetime | None:
        return _as_utc(v) if v else None


class Signal(BaseModel):
    """A condition that was observed to hold during an incident.

    `probe` names a telemetry tool that can decide whether the same condition
    holds *right now* -- this is the hinge the whole product turns on. A signal
    with no probe is still useful narrative context, it just cannot be verified.
    """

    id: str
    statement: str
    probe: str | None = None
    probe_args: dict[str, Any] = Field(default_factory=dict)


class RemediationStep(BaseModel):
    """One action the team actually took, in the order they took it."""

    order: int
    action: str
    command: str | None = None
    actor: str | None = None
    outcome: str | None = None
    is_fix: bool = Field(
        default=False,
        description="True for the step(s) that actually resolved the incident, "
        "as distinct from diagnosis or mitigation.",
    )


class Incident(BaseModel):
    """A historical incident, reconstructed from every source that mentions it."""

    id: str
    title: str
    summary: str = ""
    root_cause: str = ""
    severity: Severity = Severity.SEV3
    services: list[str] = Field(default_factory=list)
    started_at: datetime
    resolved_at: datetime | None = None
    symptoms: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(
        default_factory=list,
        description="Exact-match tokens: error codes, host ids, exception classes.",
    )
    signals: list[Signal] = Field(default_factory=list)
    remediation: list[RemediationStep] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @field_validator("started_at", "resolved_at")
    @classmethod
    def _tz(cls, v: datetime | None) -> datetime | None:
        return _as_utc(v) if v else None

    @property
    def time_to_resolve_minutes(self) -> float | None:
        if not self.resolved_at:
            return None
        return (self.resolved_at - self.started_at).total_seconds() / 60.0

    def age_days(self, now: datetime | None = None) -> float:
        return ((now or utcnow()) - self.started_at).total_seconds() / 86400.0

    def matching_text(self) -> str:
        """The text the dense vector is built from.

        Deliberately *not* the whole record: an alert can only ever describe how
        something presented, so the semantic lane matches symptom against
        symptom. Including the remediation prose pulls the vector toward how the
        incident was fixed, which is exactly the part a firing alert cannot
        resemble -- and measurably blurs the distinction between unrelated
        incidents that happened to be fixed the same way.
        """
        parts = [
            self.title,
            " ".join(self.symptoms),
            " ".join(self.keywords),
            " ".join(self.services),
            " ".join(self.tags),
        ]
        return "\n".join(p for p in parts if p)

    def searchable_text(self) -> str:
        """The text the sparse (BM25) vector is built from: everything.

        The lexical lane is where an exact token should be findable wherever it
        appears -- including in a command inside a remediation step.
        """
        parts = [
            self.title,
            " ".join(self.symptoms),
            self.root_cause,
            self.summary,
            " ".join(s.statement for s in self.signals),
            " ".join(
                " ".join(filter(None, (step.action, step.command, step.outcome)))
                for step in self.remediation
            ),
            " ".join(self.services),
            " ".join(self.tags),
            " ".join(self.keywords),
        ]
        return "\n".join(p for p in parts if p)

    def fix_steps(self) -> list[RemediationStep]:
        steps = [s for s in self.remediation if s.is_fix]
        return steps or self.remediation


class Fragment(BaseModel):
    """A raw snippet of source material (Slack message, PD log entry, wiki
    section) kept alongside incidents so answers can quote what people said."""

    id: str
    incident_id: str | None = None
    text: str
    source: SourceRef
    author: str | None = None
    timestamp: datetime | None = None

    @field_validator("timestamp")
    @classmethod
    def _tz(cls, v: datetime | None) -> datetime | None:
        return _as_utc(v) if v else None


class Alert(BaseModel):
    """A newly firing alert -- the query side of the system."""

    id: str
    title: str
    description: str = ""
    service: str | None = None
    host: str | None = None
    severity: Severity = Severity.SEV3
    fired_at: datetime = Field(default_factory=utcnow)
    labels: dict[str, str] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)

    @field_validator("fired_at")
    @classmethod
    def _tz(cls, v: datetime) -> datetime:
        return _as_utc(v)

    def query_text(self) -> str:
        parts = [self.title, self.description]
        if self.service:
            parts.append(f"service {self.service}")
        if self.host:
            parts.append(f"host {self.host}")
        parts.extend(f"{k} {v}" for k, v in sorted(self.labels.items()))
        return "\n".join(p for p in parts if p)


class ScoreBreakdown(BaseModel):
    """Every term that fed the headline similarity number.

    Exposed on the API because "87% similar" is only actionable if a responder
    can see *why* -- and can tell a lexical error-code hit from a vague
    semantic resemblance.
    """

    semantic: float = 0.0
    lexical: float = 0.0
    keyword: float = 0.0
    matched_keywords: list[str] = Field(default_factory=list)
    base: float = 0.0
    recency_weight: float = 1.0
    age_days: float = 0.0
    final: float = 0.0


class RetrievedIncident(BaseModel):
    incident: Incident
    similarity: float = Field(description="0-1 after fusion and time decay.")
    breakdown: ScoreBreakdown

    @property
    def similarity_pct(self) -> int:
        return int(round(self.similarity * 100))


class CheckStatus(str, Enum):
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    UNKNOWN = "unknown"


class TelemetryObservation(BaseModel):
    """The result of asking a live system a question, right now."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    summary: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    observed_at: datetime = Field(default_factory=utcnow)


class EvidenceCheck(BaseModel):
    """A historical signal, re-tested against current telemetry."""

    signal_id: str
    statement: str
    incident_ids: list[str] = Field(default_factory=list)
    status: CheckStatus = CheckStatus.UNKNOWN
    observation: TelemetryObservation | None = None
    note: str = ""


class RecommendedAction(BaseModel):
    action: str
    rationale: str = ""
    command: str | None = None
    from_incident: str | None = None
    risk: Literal["low", "medium", "high"] = "medium"
    requires_human_approval: bool = True


class SimilarIncidentBrief(BaseModel):
    """The per-incident half of a reconstruction."""

    incident_id: str
    title: str
    similarity_pct: int
    occurred: datetime
    root_cause: str
    actions_taken: list[str] = Field(default_factory=list)
    resolution_minutes: float | None = None


class Reconstruction(BaseModel):
    """The central artifact: what this alert most likely is, based on history,
    and what to do about it next."""

    alert_id: str
    headline: str = Field(
        description="One line a responder can read in a pager buzz, e.g. "
        "'Looks 87% like INC-2197: connection pool starved by a long-running migration.'"
    )
    similar_incidents: list[SimilarIncidentBrief] = Field(default_factory=list)
    root_cause_hypothesis: str = ""
    narrative: str = Field(
        default="",
        description="The reconstruction prose: what happened before, what the team did, "
        "and which of those conditions telemetry says are true again now.",
    )
    evidence: list[EvidenceCheck] = Field(default_factory=list)
    recommended_actions: list[RecommendedAction] = Field(default_factory=list)
    confidence: float = 0.0
    caveats: list[str] = Field(default_factory=list)
    reasoner: str = "claude"
    generated_at: datetime = Field(default_factory=utcnow)


class ReconstructionDraft(BaseModel):
    """The slice of a reconstruction the model is allowed to author.

    Deliberately narrower than `Reconstruction`: similarity scores, evidence
    statuses and timestamps are computed by the system and merged in afterwards,
    so the model can never invent a retrieval score or claim a telemetry result
    that was not actually observed.
    """

    headline: str
    root_cause_hypothesis: str
    narrative: str
    recommended_actions: list[RecommendedAction] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    caveats: list[str] = Field(default_factory=list)
