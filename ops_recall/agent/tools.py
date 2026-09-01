"""Tools Claude can call while investigating an alert.

These are the agentic half of the system. Retrieval and the replayed signal
probes are deterministic and always run; these tools let the model chase a
hypothesis the corpus did not anticipate -- "the pool is saturated but nothing
is blocking, is the service itself leaking connections?" -- by asking the live
system a question of its own.

Everything the model can call is read-only. Nothing here terminates a backend,
rolls back a deploy or edits configuration: remediation is proposed to a human,
never executed. That is a deliberate boundary, not an unimplemented feature.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from anthropic import beta_tool

from ops_recall.models import TelemetryObservation
from ops_recall.retrieval.ranker import HybridRanker
from ops_recall.telemetry.provider import TelemetryProvider


@dataclass
class ToolContext:
    """Bound per request, so tools stay pure functions of their arguments plus
    an explicit context rather than reaching for globals."""

    provider: TelemetryProvider
    ranker: HybridRanker | None = None
    observations: list[TelemetryObservation] = field(default_factory=list)

    def run(self, probe: str, **args: Any) -> str:
        observation = self.provider.probe(probe, **args)
        self.observations.append(observation)
        payload: dict[str, Any] = {"ok": observation.ok, "summary": observation.summary}
        if observation.data:
            payload["data"] = observation.data
        return json.dumps(payload, default=str)


def build_tools(context: ToolContext) -> list[Callable[..., str]]:
    """Decorated tool functions closed over `context`."""

    @beta_tool
    def check_db_locks(database: str, min_seconds: float = 60) -> str:
        """Check whether any session is currently blocking others on a database.

        Returns the blocking query, how long it has been holding the lock, and
        how many sessions are waiting behind it. This is the first thing to
        check when a service reports connection timeouts.

        Args:
            database: Database name, for example orders_db.
            min_seconds: Only count blocks that have been held at least this long.
        """
        return context.run("check_db_locks", database=database, min_seconds=min_seconds)

    @beta_tool
    def check_connection_pool(service: str, saturation: float = 0.9) -> str:
        """Read a service's database connection pool utilization and queue depth.

        Args:
            service: Service name, for example checkout-api.
            saturation: Utilization ratio (0-1) treated as saturated.
        """
        return context.run("check_connection_pool", service=service, saturation=saturation)

    @beta_tool
    def check_recent_deploys(service: str, hours: float = 6, expect: str = "any") -> str:
        """List recent deploys of a service.

        Use this to separate "something we shipped" from "something the
        environment did". Pass expect='none' when testing the hypothesis that
        the incident is not deploy-related.

        Args:
            service: Service name.
            hours: How far back to look.
            expect: 'any' or 'none' -- which outcome counts as confirming.
        """
        return context.run("check_recent_deploys", service=service, hours=hours, expect=expect)

    @beta_tool
    def check_error_rate(service: str, threshold: float = 0.05) -> str:
        """Current error rate for a service, as a fraction of requests.

        Args:
            service: Service name.
            threshold: Fraction above which the rate counts as elevated.
        """
        return context.run("check_error_rate", service=service, threshold=threshold)

    @beta_tool
    def check_service_metric(
        service: str, metric: str, threshold: float = 0, comparison: str = "gte"
    ) -> str:
        """Read one gauge for a service and compare it against a threshold.

        Args:
            service: Service name.
            metric: One of heap_used_pct, cpu_pct, p99_ms, rps, restarts_1h.
            threshold: Value to compare against.
            comparison: 'gte' or 'lte'.
        """
        return context.run(
            "check_service_metric",
            service=service,
            metric=metric,
            threshold=threshold,
            comparison=comparison,
        )

    @beta_tool
    def check_consumer_lag(group: str, threshold: float = 1000) -> str:
        """Kafka consumer group lag, partition count and rebalance state.

        Args:
            group: Consumer group id.
            threshold: Lag counted as unhealthy.
        """
        return context.run("check_consumer_lag", group=group, threshold=threshold)

    @beta_tool
    def check_certificate_expiry(host: str, within_days: float = 14) -> str:
        """Days until the TLS certificate served by a host expires.

        Args:
            host: Hostname.
            within_days: Window treated as urgent.
        """
        return context.run("check_certificate_expiry", host=host, within_days=within_days)

    @beta_tool
    def check_disk_usage(host: str, mount: str = "/", threshold_pct: float = 85) -> str:
        """Disk utilization for a mount on a host.

        Args:
            host: Host name.
            mount: Mount point, for example /var/lib/postgresql.
            threshold_pct: Percentage treated as full.
        """
        return context.run(
            "check_disk_usage", host=host, mount=mount, threshold_pct=threshold_pct
        )

    @beta_tool
    def check_replication_slots(
        database: str, max_retained_bytes: float = 10_000_000_000
    ) -> str:
        """Replication slot backlog for a database, in bytes of retained WAL.

        An inactive slot pins every WAL segment it has not consumed, which is
        the usual cause of a database volume filling up with no change in write
        volume.

        Args:
            database: Database name.
            max_retained_bytes: Backlog treated as stalled.
        """
        return context.run(
            "check_replication_slots",
            database=database,
            max_retained_bytes=max_retained_bytes,
        )

    tools: list[Callable[..., str]] = [
        check_db_locks,
        check_connection_pool,
        check_recent_deploys,
        check_error_rate,
        check_service_metric,
        check_consumer_lag,
        check_certificate_expiry,
        check_disk_usage,
        check_replication_slots,
    ]

    if context.ranker is not None:
        ranker = context.ranker

        @beta_tool
        def search_incidents(query: str, limit: int = 3) -> str:
            """Search the incident archive for historical incidents matching a
            description. Use this to follow up on a hypothesis that the original
            alert text did not cover.

            Args:
                query: Free-text description, including any error codes or host ids.
                limit: Maximum number of incidents to return.
            """
            ranked = ranker.rank(query, top_k=min(int(limit), 5))
            return json.dumps(
                [
                    {
                        "incident_id": item.incident.id,
                        "title": item.incident.title,
                        "similarity_pct": item.similarity_pct,
                        "occurred": item.incident.started_at.date().isoformat(),
                        "root_cause": item.incident.root_cause[:400],
                    }
                    for item in ranked.results
                ]
            )

        @beta_tool
        def get_runbook(topic: str) -> str:
            """Look up the internal runbook section for a topic, for example
            'connection pool exhausted' or 'consumer lag'.

            Args:
                topic: What you need the procedure for.
            """
            hits = ranker.store.search_fragments(topic, source_kinds=["wiki"], limit=3)
            return json.dumps(
                [
                    {"title": fragment.source.title, "url": fragment.source.url,
                     "text": fragment.text[:1500]}
                    for fragment, _ in hits
                ]
            )

        tools.extend([search_incidents, get_runbook])

    return tools
