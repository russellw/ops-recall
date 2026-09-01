"""Live system probes.

A probe answers one yes/no question about the system *right now*, and returns
the numbers behind the answer. That shape is deliberate: a historical signal
("a session was holding a lock for over a minute") is only useful if the same
question can be re-asked today and get a comparable answer. Every probe result
carries `data["holds"]`, which is what turns a past condition into confirmed or
refuted present-day evidence.

`FixtureTelemetryProvider` (see `mock.py`) reads a JSON snapshot of the world,
so demos and tests are deterministic. A production provider subclasses
`TelemetryProvider` and overrides the `probe_*` methods to hit Prometheus,
`pg_stat_activity`, the deploy tracker, and so on -- the probe names, argument
names and return shape stay identical, so nothing downstream changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ops_recall.models import TelemetryObservation, utcnow


@dataclass(frozen=True)
class ProbeArg:
    name: str
    type: str
    description: str
    required: bool = False
    default: Any = None


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    description: str
    args: tuple[ProbeArg, ...] = ()

    def json_schema(self) -> dict[str, Any]:
        properties = {
            arg.name: {"type": arg.type, "description": arg.description}
            for arg in self.args
        }
        return {
            "type": "object",
            "properties": properties,
            "required": [a.name for a in self.args if a.required],
            "additionalProperties": False,
        }


PROBES: tuple[ProbeSpec, ...] = (
    ProbeSpec(
        "check_db_locks",
        "Check whether any session is currently blocking others on a database, "
        "and for how long. Returns the blocking query and the number of waiters.",
        (
            ProbeArg("database", "string", "Database name, e.g. orders_db", required=True),
            ProbeArg("min_seconds", "number", "Only count blocks held at least this long", default=60),
        ),
    ),
    ProbeSpec(
        "check_connection_pool",
        "Check a service's database connection pool utilization and the number "
        "of requests queued waiting for a connection.",
        (
            ProbeArg("service", "string", "Service name, e.g. checkout-api", required=True),
            ProbeArg("saturation", "number", "Utilization ratio counted as saturated", default=0.9),
        ),
    ),
    ProbeSpec(
        "check_recent_deploys",
        "List deploys of a service in the recent past. Use expect='none' to test "
        "the hypothesis that the incident is NOT deploy-related.",
        (
            ProbeArg("service", "string", "Service name", required=True),
            ProbeArg("hours", "number", "Look-back window in hours", default=6),
            ProbeArg("expect", "string", "'any' or 'none' -- which outcome counts as holding", default="any"),
        ),
    ),
    ProbeSpec(
        "check_error_rate",
        "Current error rate for a service as a fraction of requests.",
        (
            ProbeArg("service", "string", "Service name", required=True),
            ProbeArg("threshold", "number", "Error-rate fraction counted as elevated", default=0.05),
        ),
    ),
    ProbeSpec(
        "check_service_metric",
        "Read one gauge for a service (heap_used_pct, cpu_pct, p99_ms, rps, "
        "restarts_1h) and compare it against a threshold.",
        (
            ProbeArg("service", "string", "Service name", required=True),
            ProbeArg("metric", "string", "Metric name", required=True),
            ProbeArg("threshold", "number", "Value to compare against", default=0),
            ProbeArg("comparison", "string", "'gte' or 'lte'", default="gte"),
        ),
    ),
    ProbeSpec(
        "check_consumer_lag",
        "Kafka consumer group lag in messages.",
        (
            ProbeArg("group", "string", "Consumer group id", required=True),
            ProbeArg("threshold", "number", "Lag counted as unhealthy", default=1000),
        ),
    ),
    ProbeSpec(
        "check_certificate_expiry",
        "Days until a TLS certificate expires.",
        (
            ProbeArg("host", "string", "Hostname", required=True),
            ProbeArg("within_days", "number", "Expiry window counted as urgent", default=14),
        ),
    ),
    ProbeSpec(
        "check_disk_usage",
        "Disk utilization for a mount on a host.",
        (
            ProbeArg("host", "string", "Host name", required=True),
            ProbeArg("mount", "string", "Mount point", default="/"),
            ProbeArg("threshold_pct", "number", "Percentage counted as full", default=85),
        ),
    ),
    ProbeSpec(
        "check_replication_slots",
        "Replication slot backlog for a database, in bytes of retained WAL.",
        (
            ProbeArg("database", "string", "Database name", required=True),
            ProbeArg("max_retained_bytes", "number", "Backlog counted as stalled", default=10_000_000_000),
        ),
    ),
)

PROBES_BY_NAME: dict[str, ProbeSpec] = {p.name: p for p in PROBES}


class TelemetryError(RuntimeError):
    pass


class TelemetryProvider:
    """Dispatches probe names to `probe_<name>` methods."""

    name: str = "base"

    def available_probes(self) -> tuple[ProbeSpec, ...]:
        return tuple(p for p in PROBES if hasattr(self, f"probe_{p.name}"))

    def probe(self, name: str, **args: Any) -> TelemetryObservation:
        spec = PROBES_BY_NAME.get(name)
        handler: Callable[..., TelemetryObservation] | None = getattr(
            self, f"probe_{name}", None
        )
        if spec is None or handler is None:
            return TelemetryObservation(
                tool=name,
                args=args,
                ok=False,
                summary=f"No such probe: {name}",
            )
        merged = {a.name: a.default for a in spec.args if a.default is not None}
        merged.update({k: v for k, v in args.items() if v is not None})
        # Arguments arrive from JSON tool calls and from stored signal metadata,
        # so a numeric argument may show up as a string. Coerce against the
        # declared schema rather than making every probe defensive.
        for arg in spec.args:
            if arg.type == "number" and arg.name in merged:
                try:
                    merged[arg.name] = float(merged[arg.name])
                except (TypeError, ValueError):
                    return TelemetryObservation(
                        tool=name,
                        args=merged,
                        ok=False,
                        summary=f"Argument `{arg.name}` must be a number, got {merged[arg.name]!r}",
                    )
        missing = [a.name for a in spec.args if a.required and a.name not in merged]
        if missing:
            return TelemetryObservation(
                tool=name,
                args=merged,
                ok=False,
                summary=f"Missing required argument(s): {', '.join(missing)}",
            )
        try:
            return handler(**merged)
        except TelemetryError as exc:
            return TelemetryObservation(tool=name, args=merged, ok=False, summary=str(exc))

    @staticmethod
    def _observation(
        name: str,
        args: dict[str, Any],
        holds: bool,
        summary: str,
        **data: Any,
    ) -> TelemetryObservation:
        return TelemetryObservation(
            tool=name,
            args=args,
            ok=True,
            summary=summary,
            data={"holds": holds, **data},
            observed_at=utcnow(),
        )


@dataclass
class NullTelemetryProvider(TelemetryProvider):
    """Used when no telemetry backend is configured: every check comes back
    unknown, and the reconstruction says so rather than guessing."""

    name: str = "null"
    reason: str = "no telemetry provider configured"
    _unused: dict[str, Any] = field(default_factory=dict, repr=False)

    def probe(self, name: str, **args: Any) -> TelemetryObservation:
        return TelemetryObservation(tool=name, args=args, ok=False, summary=self.reason)

    def available_probes(self) -> tuple[ProbeSpec, ...]:
        return ()
