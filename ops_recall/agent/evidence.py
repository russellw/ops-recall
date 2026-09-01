"""Re-testing history against the present.

Retrieval says "this looks like INC-2197". That is a claim about the past. The
question a responder actually has is whether the *conditions* that made
INC-2197 what it was hold right now -- and that is what turns a similarity
score into a recommendation worth following.

Each retrieved incident carries signals bound to telemetry probes. This module
deduplicates those probes across incidents (three incidents in the same family
usually share signals), retargets them at the service the current alert names,
runs each one once, and labels the result confirmed / refuted / unknown.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from ops_recall.models import (
    Alert,
    CheckStatus,
    EvidenceCheck,
    RetrievedIncident,
    Signal,
)
from ops_recall.telemetry.provider import PROBES_BY_NAME, TelemetryProvider

# Probe arguments that name the thing under investigation, and can therefore be
# retargeted from the historical subject to the current one.
RETARGETABLE = {"service": "service", "host": "host"}


def _args_key(args: dict[str, Any]) -> tuple:
    return tuple(sorted((k, str(v)) for k, v in args.items()))


def retarget(
    signal: Signal, incident_services: Sequence[str], alert: Alert | None
) -> tuple[dict[str, Any], str]:
    """Point a historical probe at the currently affected subject.

    A signal recorded as "checkout-api's pool is saturated" generalizes to "the
    affected service's pool is saturated". Retargeting only happens when the
    historical value is one of that incident's own services (so it really was
    the subject) and the alert names a different one -- otherwise the probe runs
    exactly as it was written.
    """
    args = dict(signal.probe_args)
    if alert is None:
        return args, ""
    notes: list[str] = []
    historical = {s.lower() for s in incident_services}
    for arg_name, alert_attr in RETARGETABLE.items():
        if arg_name not in args:
            continue
        current = getattr(alert, alert_attr, None)
        if not current:
            continue
        old = str(args[arg_name])
        if old.lower() == current.lower():
            continue
        if arg_name == "service" and old.lower() not in historical:
            continue
        args[arg_name] = current
        notes.append(f"retargeted {arg_name} {old} -> {current}")
    return args, "; ".join(notes)


class EvidenceChecker:
    def __init__(self, provider: TelemetryProvider) -> None:
        self.provider = provider

    def check(
        self,
        retrieved: Iterable[RetrievedIncident],
        alert: Alert | None = None,
    ) -> list[EvidenceCheck]:
        pending: dict[tuple, EvidenceCheck] = {}
        probe_args: dict[tuple, dict[str, Any]] = {}
        order: list[tuple] = []

        for item in retrieved:
            incident = item.incident
            for signal in incident.signals:
                if not signal.probe or signal.probe not in PROBES_BY_NAME:
                    continue
                args, note = retarget(signal, incident.services, alert)
                key = (signal.probe, _args_key(args))
                existing = pending.get(key)
                if existing is None:
                    pending[key] = EvidenceCheck(
                        signal_id=signal.id,
                        statement=signal.statement,
                        incident_ids=[incident.id],
                        note=note,
                    )
                    probe_args[key] = args
                    order.append(key)
                elif incident.id not in existing.incident_ids:
                    # The same condition seen in several incidents is stronger
                    # evidence, not a duplicate check.
                    existing.incident_ids.append(incident.id)

        results: list[EvidenceCheck] = []
        for key in order:
            probe_name, _ = key
            check = pending[key]
            observation = self.provider.probe(probe_name, **probe_args[key])
            check.observation = observation
            if not observation.ok:
                check.status = CheckStatus.UNKNOWN
            else:
                check.status = (
                    CheckStatus.CONFIRMED
                    if observation.data.get("holds")
                    else CheckStatus.REFUTED
                )
            results.append(check)

        # Confirmed evidence first: that is what a responder reads.
        rank = {CheckStatus.CONFIRMED: 0, CheckStatus.REFUTED: 1, CheckStatus.UNKNOWN: 2}
        results.sort(key=lambda c: (rank[c.status], -len(c.incident_ids)))
        return results


def evidence_summary(checks: Sequence[EvidenceCheck]) -> dict[str, int]:
    summary = {status.value: 0 for status in CheckStatus}
    for check in checks:
        summary[check.status.value] += 1
    return summary
