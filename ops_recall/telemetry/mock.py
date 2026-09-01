"""Fixture-backed telemetry.

Reads a JSON snapshot of "the system right now". Editing that file changes what
the agent concludes, which makes the whole reconstruction path demoable and
testable without a staging environment: flip `blocking_sessions` to empty and
the same alert produces a different recommended action.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ops_recall.models import TelemetryObservation, utcnow
from ops_recall.telemetry.provider import TelemetryProvider


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


class FixtureTelemetryProvider(TelemetryProvider):
    name = "fixture"

    def __init__(self, state: dict[str, Any] | None = None, path: Path | None = None) -> None:
        if state is None:
            if path is None:
                raise ValueError("FixtureTelemetryProvider needs `state` or `path`")
            state = json.loads(Path(path).read_text(encoding="utf-8"))
        self.state = state

    @classmethod
    def from_path(cls, path: Path) -> "FixtureTelemetryProvider":
        return cls(path=path)

    @staticmethod
    def _when(record: dict[str, Any], absolute_key: str) -> datetime | None:
        """Resolve a timestamp that may be absolute or relative.

        Fixtures written with absolute timestamps go stale -- a deploy dated
        last March stops being "recent" and the demo quietly changes meaning.
        Entries may therefore say `minutes_ago` / `days_from_now` instead, which
        are resolved against the clock at read time.
        """
        if record.get("minutes_ago") is not None:
            return utcnow() - timedelta(minutes=float(record["minutes_ago"]))
        if record.get("days_from_now") is not None:
            return utcnow() + timedelta(days=float(record["days_from_now"]))
        return _parse_time(record.get(absolute_key))

    def _section(self, *keys: str) -> dict[str, Any]:
        node: Any = self.state
        for key in keys:
            node = (node or {}).get(key, {})
        return node or {}

    # -- probes ---------------------------------------------------------

    def probe_check_db_locks(self, database: str, min_seconds: float = 60) -> TelemetryObservation:
        db = self._section("databases", database)
        sessions = [
            s for s in db.get("blocking_sessions", [])
            if float(s.get("duration_seconds", 0)) >= float(min_seconds)
        ]
        if not sessions:
            return self._observation(
                "check_db_locks",
                {"database": database, "min_seconds": min_seconds},
                False,
                f"No session has blocked others on {database} for {min_seconds:g}s or more.",
                blocking_sessions=[],
            )
        worst = max(sessions, key=lambda s: float(s.get("duration_seconds", 0)))
        return self._observation(
            "check_db_locks",
            {"database": database, "min_seconds": min_seconds},
            True,
            f"{len(sessions)} blocking session(s) on {database}; longest is pid "
            f"{worst.get('pid')} at {worst.get('duration_seconds')}s running "
            f"{worst.get('query', 'unknown query')!r}, {worst.get('waiters', 0)} waiters.",
            blocking_sessions=sessions,
            longest_seconds=float(worst.get("duration_seconds", 0)),
            blocking_query=worst.get("query"),
            blocking_pid=worst.get("pid"),
        )

    def probe_check_connection_pool(
        self, service: str, saturation: float = 0.9
    ) -> TelemetryObservation:
        pool = self._section("services", service).get("pool")
        if not pool:
            return TelemetryObservation(
                tool="check_connection_pool",
                args={"service": service},
                ok=False,
                summary=f"No pool metrics reported for {service}.",
            )
        maximum = float(pool.get("max", 0)) or 1.0
        active = float(pool.get("active", 0))
        utilization = active / maximum
        pending = int(pool.get("pending", 0))
        holds = utilization >= float(saturation)
        return self._observation(
            "check_connection_pool",
            {"service": service, "saturation": saturation},
            holds,
            f"{service} pool at {active:.0f}/{maximum:.0f} "
            f"({utilization:.0%}) with {pending} request(s) queued.",
            utilization=round(utilization, 3),
            active=active,
            max=maximum,
            pending=pending,
        )

    def probe_check_recent_deploys(
        self, service: str, hours: float = 6, expect: str = "any"
    ) -> TelemetryObservation:
        now = utcnow()
        deploys = []
        for deploy in self._section("services", service).get("deploys", []):
            when = self._when(deploy, "at")
            if when and (now - when).total_seconds() <= float(hours) * 3600:
                deploys.append({**deploy, "age_minutes": round((now - when).total_seconds() / 60, 1)})
        found = bool(deploys)
        holds = (not found) if str(expect).lower() == "none" else found
        if found:
            latest = deploys[0]
            summary = (
                f"{len(deploys)} deploy(s) of {service} in the last {hours:g}h; "
                f"most recent {latest.get('version', '?')} "
                f"{latest['age_minutes']:.0f} min ago by {latest.get('by', 'unknown')}."
            )
        else:
            summary = f"No deploys of {service} in the last {hours:g}h."
        return self._observation(
            "check_recent_deploys",
            {"service": service, "hours": hours, "expect": expect},
            holds,
            summary,
            deploys=deploys,
        )

    def probe_check_error_rate(
        self, service: str, threshold: float = 0.05
    ) -> TelemetryObservation:
        metrics = self._section("services", service)
        if "error_rate" not in metrics:
            return TelemetryObservation(
                tool="check_error_rate",
                args={"service": service},
                ok=False,
                summary=f"No error-rate metric for {service}.",
            )
        rate = float(metrics["error_rate"])
        return self._observation(
            "check_error_rate",
            {"service": service, "threshold": threshold},
            rate >= float(threshold),
            f"{service} error rate {rate:.1%} (threshold {float(threshold):.1%}).",
            error_rate=rate,
        )

    def probe_check_service_metric(
        self,
        service: str,
        metric: str,
        threshold: float = 0,
        comparison: str = "gte",
    ) -> TelemetryObservation:
        metrics = self._section("services", service)
        if metric not in metrics:
            return TelemetryObservation(
                tool="check_service_metric",
                args={"service": service, "metric": metric},
                ok=False,
                summary=f"{service} does not report {metric}.",
            )
        value = float(metrics[metric])
        holds = value >= float(threshold) if comparison == "gte" else value <= float(threshold)
        return self._observation(
            "check_service_metric",
            {"service": service, "metric": metric, "threshold": threshold, "comparison": comparison},
            holds,
            f"{service} {metric} = {value:g} ({comparison} {float(threshold):g} -> {holds}).",
            value=value,
        )

    def probe_check_consumer_lag(
        self, group: str, threshold: float = 1000
    ) -> TelemetryObservation:
        groups = self._section("kafka", "consumer_groups")
        if group not in groups:
            return TelemetryObservation(
                tool="check_consumer_lag",
                args={"group": group},
                ok=False,
                summary=f"Unknown consumer group {group}.",
            )
        info = groups[group]
        lag = float(info.get("lag", 0))
        return self._observation(
            "check_consumer_lag",
            {"group": group, "threshold": threshold},
            lag >= float(threshold),
            f"Consumer group {group} lag {lag:.0f} messages across "
            f"{info.get('partitions', '?')} partition(s); state {info.get('state', 'unknown')}.",
            lag=lag,
            state=info.get("state"),
        )

    def probe_check_certificate_expiry(
        self, host: str, within_days: float = 14
    ) -> TelemetryObservation:
        cert = self._section("certificates", host)
        not_after = self._when(cert, "not_after")
        if not not_after:
            return TelemetryObservation(
                tool="check_certificate_expiry",
                args={"host": host},
                ok=False,
                summary=f"No certificate record for {host}.",
            )
        days = (not_after - utcnow()).total_seconds() / 86400
        return self._observation(
            "check_certificate_expiry",
            {"host": host, "within_days": within_days},
            days <= float(within_days),
            f"{host} certificate expires in {days:.1f} days ({not_after.date()}).",
            days_remaining=round(days, 2),
            not_after=not_after.isoformat(),
        )

    def probe_check_disk_usage(
        self, host: str, mount: str = "/", threshold_pct: float = 85
    ) -> TelemetryObservation:
        disks = self._section("hosts", host).get("disks", {})
        if mount not in disks:
            return TelemetryObservation(
                tool="check_disk_usage",
                args={"host": host, "mount": mount},
                ok=False,
                summary=f"No disk metrics for {host}:{mount}.",
            )
        used = float(disks[mount].get("used_pct", 0))
        return self._observation(
            "check_disk_usage",
            {"host": host, "mount": mount, "threshold_pct": threshold_pct},
            used >= float(threshold_pct),
            f"{host}:{mount} is {used:.0f}% full (threshold {float(threshold_pct):.0f}%).",
            used_pct=used,
        )

    def probe_check_replication_slots(
        self, database: str, max_retained_bytes: float = 10_000_000_000
    ) -> TelemetryObservation:
        slots = self._section("databases", database).get("replication_slots", [])
        stalled = [
            s for s in slots
            if float(s.get("retained_bytes", 0)) >= float(max_retained_bytes)
        ]
        if not slots:
            return TelemetryObservation(
                tool="check_replication_slots",
                args={"database": database},
                ok=False,
                summary=f"No replication slot data for {database}.",
            )
        worst = max(slots, key=lambda s: float(s.get("retained_bytes", 0)))
        return self._observation(
            "check_replication_slots",
            {"database": database, "max_retained_bytes": max_retained_bytes},
            bool(stalled),
            f"{len(stalled)} of {len(slots)} slot(s) on {database} over the backlog "
            f"threshold; worst is {worst.get('slot_name')} retaining "
            f"{float(worst.get('retained_bytes', 0)) / 1e9:.1f} GB "
            f"(active={worst.get('active')}).",
            slots=slots,
            worst_retained_bytes=float(worst.get("retained_bytes", 0)),
        )


def build_provider(settings) -> TelemetryProvider:
    """Fixture provider when the snapshot exists, null provider otherwise."""
    from ops_recall.telemetry.provider import NullTelemetryProvider

    path = Path(settings.telemetry_fixture)
    if path.is_file():
        return FixtureTelemetryProvider.from_path(path)
    return NullTelemetryProvider(reason=f"telemetry fixture not found at {path}")
