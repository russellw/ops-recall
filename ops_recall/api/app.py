"""FastAPI surface.

The important endpoint is `POST /v1/alerts`: give it a firing alert, get back a
reconstruction. `POST /v1/alerts/pagerduty` is the same thing behind a webhook
adapter, so PagerDuty can be pointed straight at it.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request

from ops_recall.api.schemas import (
    AlertRequest,
    ReconstructionResponse,
    ScoredIncidentResponse,
    SearchRequest,
    SearchResponse,
)
from ops_recall.ingest.pagerduty import alert_from_webhook
from ops_recall.ingest.pipeline import build_index
from ops_recall.models import Alert
from ops_recall.retrieval.store import SearchFilters
from ops_recall.service import Service, build_service

logger = logging.getLogger(__name__)


def create_app(service: Service | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Indexing the corpus is startup work, not per-request work.
        app.state.service = service or build_service()
        logger.info("ops-recall ready: %s", app.state.service.stats())
        yield

    app = FastAPI(
        title="ops-recall",
        version="0.1.0",
        summary="Reconstructs a firing alert from the team's own incident history.",
        lifespan=lifespan,
    )

    def current(request: Request) -> Service:
        svc: Service | None = getattr(request.app.state, "service", None)
        if svc is None:  # pragma: no cover - lifespan always sets it
            raise HTTPException(503, "service not ready")
        return svc

    def respond(svc: Service, alert: Alert, filters: SearchFilters | None = None):
        state = svc.agent.run(alert, filters)
        investigation = state.get("investigation")
        return ReconstructionResponse(
            reconstruction=state["reconstruction"],
            retrieved=[
                ScoredIncidentResponse.from_retrieved(item)
                for item in state.get("retrieved", [])
            ],
            trace=state.get("trace", []),
            investigation_notes=investigation.notes if investigation else None,
        )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/stats")
    def stats(request: Request) -> dict[str, Any]:
        return current(request).stats()

    @app.post("/v1/alerts", response_model=ReconstructionResponse)
    def reconstruct_alert(
        request: Request, payload: AlertRequest
    ) -> ReconstructionResponse:
        return respond(current(request), payload.to_alert())

    @app.post("/v1/alerts/pagerduty", response_model=ReconstructionResponse)
    def reconstruct_pagerduty(
        request: Request, payload: dict[str, Any] = Body(...)
    ) -> ReconstructionResponse:
        try:
            alert = alert_from_webhook(payload)
        except Exception as exc:
            raise HTTPException(422, f"unrecognized PagerDuty payload: {exc}") from exc
        return respond(current(request), alert)

    @app.post("/v1/search", response_model=SearchResponse)
    def search(request: Request, payload: SearchRequest) -> SearchResponse:
        svc = current(request)
        filters = SearchFilters(
            services=payload.services, max_age_days=payload.max_age_days
        )
        ranked = svc.agent.ranker.rank(
            payload.query, top_k=payload.top_k, filters=filters
        )
        return SearchResponse(
            query=ranked.query_text,
            keywords=ranked.query_keywords,
            considered=ranked.considered,
            results=[ScoredIncidentResponse.from_retrieved(i) for i in ranked.results],
        )

    @app.get("/v1/incidents")
    def list_incidents(request: Request) -> list[dict[str, Any]]:
        return [
            {
                "id": incident.id,
                "title": incident.title,
                "severity": incident.severity,
                "services": incident.services,
                "started_at": incident.started_at,
                "sources": [s.kind for s in incident.sources],
            }
            for incident in current(request).store.all_incidents()
        ]

    @app.get("/v1/incidents/{incident_id}")
    def get_incident(request: Request, incident_id: str) -> Any:
        incident = current(request).store.get_incident(incident_id.upper())
        if incident is None:
            raise HTTPException(404, f"unknown incident {incident_id}")
        return incident

    @app.get("/v1/probes")
    def list_probes(request: Request) -> list[dict[str, Any]]:
        """What the agent can ask the live system. Useful when writing the
        `signals:` block of a new post-mortem."""
        return [
            {"name": p.name, "description": p.description, "schema": p.json_schema()}
            for p in current(request).provider.available_probes()
        ]

    @app.post("/v1/reindex")
    def reindex(request: Request) -> dict[str, Any]:
        svc = current(request)
        _, counts = build_index(svc.settings, store=svc.store)
        svc.counts = counts
        return {"reindexed": counts, "index": svc.store.stats()}

    return app


app = create_app()


def main() -> None:  # pragma: no cover - entry point
    import uvicorn

    uvicorn.run(
        "ops_recall.api.app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
    )
