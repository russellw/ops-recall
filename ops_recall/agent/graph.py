"""The LangGraph pipeline.

    retrieve ──► no close match ──► cold_start ──► END
        │
        └─► check_evidence ──► gather_quotes ──► [investigate] ──► reconstruct ──► END

Retrieval and evidence checking are deterministic and always run; the model is
reached only at the end, with everything it needs already assembled. That
ordering is the point of the design -- the expensive, non-deterministic step
gets facts rather than being asked to go find them, and the whole state at each
hop is inspectable when a recommendation later turns out to be wrong.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence, TypedDict

from langgraph.graph import END, START, StateGraph

from ops_recall.agent.evidence import EvidenceChecker
from ops_recall.agent.investigate import AgenticInvestigator, Investigation
from ops_recall.agent.reconstruct import (
    Reconstructor,
    TemplateReconstructor,
    build_reconstructor,
)
from ops_recall.config import Settings, get_settings
from ops_recall.models import (
    Alert,
    EvidenceCheck,
    Fragment,
    Reconstruction,
    RetrievedIncident,
)
from ops_recall.retrieval.ranker import HybridRanker, RankedResults
from ops_recall.retrieval.store import IncidentStore, SearchFilters
from ops_recall.telemetry.provider import TelemetryProvider

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    alert: Alert
    filters: SearchFilters | None
    ranked: RankedResults
    retrieved: list[RetrievedIncident]
    evidence: list[EvidenceCheck]
    quotes: list[tuple[Fragment, float]]
    investigation: Investigation | None
    reconstruction: Reconstruction
    trace: list[dict[str, Any]]


class IncidentAgent:
    """Wires the pieces together and compiles the graph once, at construction."""

    def __init__(
        self,
        store: IncidentStore,
        provider: TelemetryProvider,
        settings: Settings | None = None,
        client: Any = None,
        reconstructor: Reconstructor | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store
        self.provider = provider
        self.ranker = HybridRanker(store, self.settings)
        self.checker = EvidenceChecker(provider)
        self.client = client
        self.reconstructor = reconstructor or build_reconstructor(self.settings, client)
        self.investigator = (
            AgenticInvestigator(client, self.settings, self.ranker)
            if client is not None and self.settings.agentic_checks
            else None
        )
        self.graph = self._build().compile()

    # -- nodes ----------------------------------------------------------

    def _retrieve(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        ranked = self.ranker.rank_alert(state["alert"], filters=state.get("filters"))
        return {
            "ranked": ranked,
            "retrieved": ranked.results,
            "trace": _trace(
                state,
                "retrieve",
                started,
                considered=ranked.considered,
                kept=len(ranked.results),
                keywords=ranked.query_keywords,
                top=[f"{r.incident.id}:{r.similarity_pct}%" for r in ranked.results],
            ),
        }

    def _check_evidence(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        evidence = self.checker.check(state["retrieved"], state["alert"])
        return {
            "evidence": evidence,
            "trace": _trace(
                state,
                "check_evidence",
                started,
                probes=len(evidence),
                confirmed=sum(1 for c in evidence if c.status.value == "confirmed"),
                refuted=sum(1 for c in evidence if c.status.value == "refuted"),
                unknown=sum(1 for c in evidence if c.status.value == "unknown"),
                provider=self.provider.name,
            ),
        }

    def _gather_quotes(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        incident_ids = [item.incident.id for item in state["retrieved"]]
        candidates = self.store.search_fragments(
            state["alert"].query_text(), incident_ids=incident_ids, limit=24
        )
        quotes = select_quotes(candidates, per_incident=3, total=6)
        return {
            "quotes": quotes,
            "trace": _trace(state, "gather_quotes", started, quotes=len(quotes)),
        }

    def _investigate(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        assert self.investigator is not None
        investigation = self.investigator.investigate(
            state["alert"], state["retrieved"], state.get("evidence", []), self.provider
        )
        return {
            "investigation": investigation,
            "trace": _trace(
                state,
                "investigate",
                started,
                extra_checks=len(investigation.observations),
                iterations=investigation.iterations,
                error=investigation.error,
            ),
        }

    def _reconstruct(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        investigation = state.get("investigation")
        reconstruction = self.reconstructor.reconstruct(
            state["alert"],
            state["retrieved"],
            state.get("evidence", []),
            state.get("quotes", []),
            investigation.observations if investigation else (),
        )
        return {
            "reconstruction": reconstruction,
            "trace": _trace(
                state,
                "reconstruct",
                started,
                reasoner=reconstruction.reasoner,
                confidence=reconstruction.confidence,
            ),
        }

    def _cold_start(self, state: AgentState) -> dict[str, Any]:
        """No incident cleared the similarity threshold.

        Answering anyway is the failure mode that destroys trust in a system
        like this, so the cold path returns an explicit "this is novel" with
        whatever telemetry could still be gathered.
        """
        started = time.perf_counter()
        reconstruction = TemplateReconstructor(self.settings).reconstruct(
            state["alert"], [], state.get("evidence", [])
        )
        reconstruction.reasoner = "cold-start"
        return {
            "reconstruction": reconstruction,
            "trace": _trace(state, "cold_start", started, considered=state["ranked"].considered),
        }

    # -- wiring ---------------------------------------------------------

    def _build(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("check_evidence", self._check_evidence)
        graph.add_node("gather_quotes", self._gather_quotes)
        graph.add_node("reconstruct", self._reconstruct)
        graph.add_node("cold_start", self._cold_start)

        graph.add_edge(START, "retrieve")
        graph.add_conditional_edges(
            "retrieve",
            lambda state: "cold_start" if not state["retrieved"] else "check_evidence",
            {"cold_start": "cold_start", "check_evidence": "check_evidence"},
        )
        graph.add_edge("check_evidence", "gather_quotes")

        if self.investigator is not None:
            graph.add_node("investigate", self._investigate)
            graph.add_edge("gather_quotes", "investigate")
            graph.add_edge("investigate", "reconstruct")
        else:
            graph.add_edge("gather_quotes", "reconstruct")

        graph.add_edge("reconstruct", END)
        graph.add_edge("cold_start", END)
        return graph

    # -- entry points ---------------------------------------------------

    def run(self, alert: Alert, filters: SearchFilters | None = None) -> AgentState:
        return self.graph.invoke({"alert": alert, "filters": filters, "trace": []})

    def reconstruct(self, alert: Alert, filters: SearchFilters | None = None) -> Reconstruction:
        return self.run(alert, filters)["reconstruction"]


def select_quotes(
    candidates: Sequence[tuple[Fragment, float]],
    per_incident: int = 3,
    total: int = 6,
) -> list[tuple[Fragment, float]]:
    """Pick the supporting quotes worth showing.

    Pure similarity picks badly here. The closest fragment to a firing alert is
    always the *old alert text* from PagerDuty -- a near-duplicate of the query
    that tells a responder nothing they are not already looking at. What helps
    is what a person said while working the incident, so authored fragments are
    taken first and machine-generated ones only fill any remaining space. A
    per-incident cap keeps one loud incident from crowding out the second-best
    match.
    """
    def take(pool: Sequence[tuple[Fragment, float]], seen: dict[str, int]) -> list:
        picked = []
        for fragment, score in sorted(pool, key=lambda pair: -pair[1]):
            key = fragment.incident_id or ""
            if seen.get(key, 0) >= per_incident:
                continue
            seen[key] = seen.get(key, 0) + 1
            picked.append((fragment, score))
        return picked

    seen: dict[str, int] = {}
    authored = take([c for c in candidates if c[0].author], seen)
    machine = take([c for c in candidates if not c[0].author], seen)
    return (authored + machine)[:total]


def _trace(
    state: AgentState, node: str, started: float, **details: Any
) -> list[dict[str, Any]]:
    entry = {
        "node": node,
        "ms": round((time.perf_counter() - started) * 1000, 1),
        **{k: v for k, v in details.items() if v is not None},
    }
    return [*state.get("trace", []), entry]
