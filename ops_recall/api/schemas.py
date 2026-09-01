"""Request and response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ops_recall.models import (
    Alert,
    Reconstruction,
    RetrievedIncident,
    Severity,
    utcnow,
)


class AlertRequest(BaseModel):
    """A firing alert, in whatever shape the monitoring stack emits."""

    id: str | None = None
    title: str
    description: str = ""
    service: str | None = None
    host: str | None = None
    severity: str = "sev3"
    fired_at: datetime | None = None
    labels: dict[str, str] = Field(default_factory=dict)

    def to_alert(self) -> Alert:
        return Alert(
            id=self.id or f"alert-{int(utcnow().timestamp())}",
            title=self.title,
            description=self.description,
            service=self.service,
            host=self.host,
            severity=Severity.coerce(self.severity),
            fired_at=self.fired_at or utcnow(),
            labels=self.labels,
        )


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    services: list[str] = Field(default_factory=list)
    max_age_days: float | None = None


class ScoredIncidentResponse(BaseModel):
    incident_id: str
    title: str
    similarity_pct: int
    occurred: datetime
    severity: Severity
    services: list[str]
    root_cause: str
    breakdown: dict[str, Any]

    @classmethod
    def from_retrieved(cls, item: RetrievedIncident) -> "ScoredIncidentResponse":
        return cls(
            incident_id=item.incident.id,
            title=item.incident.title,
            similarity_pct=item.similarity_pct,
            occurred=item.incident.started_at,
            severity=item.incident.severity,
            services=item.incident.services,
            root_cause=item.incident.root_cause,
            breakdown=item.breakdown.model_dump(),
        )


class SearchResponse(BaseModel):
    query: str
    keywords: list[str]
    considered: int
    results: list[ScoredIncidentResponse]


class ReconstructionResponse(BaseModel):
    reconstruction: Reconstruction
    retrieved: list[ScoredIncidentResponse]
    trace: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-node timings and counts, so a surprising answer can be "
        "traced back to the retrieval or evidence step that produced it.",
    )
    investigation_notes: str | None = None
