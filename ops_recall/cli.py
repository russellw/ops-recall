"""Command line entry point.

    python -m ops_recall.cli demo               # the flagship reconstruction
    python -m ops_recall.cli ingest [--live]    # rebuild the index
    python -m ops_recall.cli alert "..." --service checkout-api
    python -m ops_recall.cli search "connection pool exhausted"
    python -m ops_recall.cli probe check_db_locks database=orders_db
    python -m ops_recall.cli stats
    python -m ops_recall.cli serve
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Sequence

from ops_recall.config import get_settings
from ops_recall.models import Alert, CheckStatus, Reconstruction, Severity, utcnow
from ops_recall.retrieval.temporal import describe_recency
from ops_recall.service import Service, build_service

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"

DEMO_ALERT = Alert(
    id="ALRT-88213",
    title="checkout-api 5xx ratio > 25% (ERR_5012 connection timeout)",
    description=(
        "HikariPool-1 - Connection is not available, request timed out after "
        "30000ms. java.sql.SQLTransientConnectionException on POST /checkout."
    ),
    service="checkout-api",
    host="db-prod-03",
    severity=Severity.SEV1,
    labels={"alertname": "HighErrorRatio", "error_code": "ERR_5012", "http_status": "503"},
)


def _color(text: str, code: str, plain: bool) -> str:
    return text if plain else f"{code}{text}{RESET}"


def print_reconstruction(state: dict[str, Any], plain: bool = False) -> None:
    reconstruction: Reconstruction = state["reconstruction"]
    out = sys.stdout.write

    out("\n" + _color(reconstruction.headline, BOLD, plain) + "\n\n")

    if reconstruction.similar_incidents:
        out(_color("Similar incidents", BOLD, plain) + "\n")
        for item, brief in zip(state.get("retrieved", []), reconstruction.similar_incidents):
            out(
                f"  {brief.incident_id}  {brief.similarity_pct:3d}%  "
                f"{describe_recency(item.breakdown.age_days):>14}  {brief.title[:70]}\n"
            )
            b = item.breakdown
            out(
                _color(
                    f"           semantic {b.semantic:.2f} | lexical {b.lexical:.2f} | "
                    f"exact-keyword {b.keyword:.2f} | recency x{b.recency_weight:.2f}\n",
                    DIM,
                    plain,
                )
            )
        out("\n")

    if reconstruction.evidence:
        out(_color("Live telemetry", BOLD, plain) + "\n")
        colors = {
            CheckStatus.CONFIRMED: GREEN,
            CheckStatus.REFUTED: RED,
            CheckStatus.UNKNOWN: YELLOW,
        }
        for check in reconstruction.evidence:
            label = _color(f"{check.status.value.upper():9}", colors[check.status], plain)
            out(f"  {label} {check.statement}\n")
            if check.observation:
                out(_color(f"            {check.observation.summary}\n", DIM, plain))
        out("\n")

    if reconstruction.root_cause_hypothesis:
        out(_color("Root cause hypothesis", BOLD, plain) + "\n")
        out(f"  {reconstruction.root_cause_hypothesis.strip()}\n\n")

    out(_color("Reconstruction", BOLD, plain) + "\n")
    for paragraph in reconstruction.narrative.split("\n\n"):
        out(f"  {paragraph.strip()}\n\n")

    if reconstruction.recommended_actions:
        out(_color("Recommended next action", BOLD, plain) + "\n")
        for index, action in enumerate(reconstruction.recommended_actions, start=1):
            out(f"  {index}. {action.action}\n")
            if action.command:
                out(_color(f"     $ {action.command}\n", DIM, plain))
            if action.rationale:
                out(_color(f"     {action.rationale}\n", DIM, plain))
            out(
                _color(
                    f"     risk {action.risk}"
                    + (" | needs human approval" if action.requires_human_approval else "")
                    + "\n",
                    DIM,
                    plain,
                )
            )
        out("\n")

    if reconstruction.caveats:
        out(_color("Caveats", BOLD, plain) + "\n")
        for caveat in reconstruction.caveats:
            out(f"  - {caveat}\n")
        out("\n")

    out(
        _color(
            f"confidence {reconstruction.confidence:.2f} | reasoner "
            f"{reconstruction.reasoner} | "
            + " -> ".join(f"{e['node']} {e['ms']}ms" for e in state.get("trace", []))
            + "\n\n",
            DIM,
            plain,
        )
    )


def _run_alert(service: Service, alert: Alert, args: argparse.Namespace) -> int:
    state = service.agent.run(alert)
    if args.json:
        print(
            json.dumps(
                {
                    "reconstruction": state["reconstruction"].model_dump(mode="json"),
                    "trace": state.get("trace", []),
                },
                indent=2,
            )
        )
    else:
        print_reconstruction(state, plain=args.plain)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    # The shared flags are attached to every subparser as well as the top
    # level, so both `cli --plain demo` and `cli demo --plain` work.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="machine-readable output")
    common.add_argument("--plain", action="store_true", help="disable color")
    common.add_argument("--verbose", action="store_true", help="log at INFO level")

    parser = argparse.ArgumentParser(
        prog="ops-recall",
        description=__doc__,
        parents=[common],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("demo", parents=[common], help="reconstruct the bundled example alert")
    ingest_cmd = sub.add_parser(
        "ingest", parents=[common], help="rebuild the index from the configured sources"
    )
    ingest_cmd.add_argument(
        "--live",
        action="store_true",
        help="also pull from the Slack and PagerDuty APIs (needs "
        "OPS_RECALL_SLACK_BOT_TOKEN / OPS_RECALL_PAGERDUTY_API_KEY)",
    )
    sub.add_parser("stats", parents=[common], help="show corpus and index statistics")
    sub.add_parser("serve", parents=[common], help="run the HTTP API")

    alert_cmd = sub.add_parser(
        "alert", parents=[common], help="reconstruct an alert given on the command line"
    )
    alert_cmd.add_argument("title")
    alert_cmd.add_argument("--description", default="")
    alert_cmd.add_argument("--service")
    alert_cmd.add_argument("--host")
    alert_cmd.add_argument("--severity", default="sev2")

    search_cmd = sub.add_parser(
        "search", parents=[common], help="rank incidents against free text"
    )
    search_cmd.add_argument("query")
    search_cmd.add_argument("--top-k", type=int, default=5)

    probe_cmd = sub.add_parser(
        "probe", parents=[common], help="run one telemetry probe directly"
    )
    probe_cmd.add_argument("name")
    probe_cmd.add_argument("args", nargs="*", help="key=value pairs")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "serve":  # pragma: no cover - long running
        from ops_recall.api.app import main as serve

        serve()
        return 0

    if args.command == "ingest":
        from ops_recall.ingest.pipeline import build_index

        store, counts = build_index(get_settings(), live=args.live)
        print(json.dumps({"ingested": counts, "index": store.stats()}, indent=2))
        return 0

    service = build_service()

    if args.command == "demo":
        return _run_alert(service, DEMO_ALERT, args)

    if args.command == "alert":
        alert = Alert(
            id=f"cli-{int(utcnow().timestamp())}",
            title=args.title,
            description=args.description,
            service=args.service,
            host=args.host,
            severity=Severity.coerce(args.severity),
        )
        return _run_alert(service, alert, args)

    if args.command == "search":
        ranked = service.agent.ranker.rank(args.query, top_k=args.top_k)
        if args.json:
            print(
                json.dumps(
                    [
                        {
                            "incident_id": item.incident.id,
                            "similarity_pct": item.similarity_pct,
                            "title": item.incident.title,
                            "breakdown": item.breakdown.model_dump(),
                        }
                        for item in ranked.results
                    ],
                    indent=2,
                )
            )
            return 0
        print(f"\nkeywords: {', '.join(ranked.query_keywords) or '(none)'}")
        print(f"{len(ranked.results)} of {ranked.considered} candidates above threshold\n")
        for item in ranked.results:
            b = item.breakdown
            print(f"  {item.incident.id}  {item.similarity_pct:3d}%  {item.incident.title[:66]}")
            print(
                f"           semantic {b.semantic:.2f} | lexical {b.lexical:.2f} | "
                f"keyword {b.keyword:.2f} | recency x{b.recency_weight:.2f} "
                f"({describe_recency(b.age_days)})"
            )
        print()
        return 0

    if args.command == "probe":
        probe_args: dict[str, Any] = {}
        for pair in args.args:
            if "=" not in pair:
                parser.error(f"probe arguments must be key=value, got {pair!r}")
            key, value = pair.split("=", 1)
            probe_args[key] = value
        observation = service.provider.probe(args.name, **probe_args)
        print(json.dumps(observation.model_dump(mode="json"), indent=2))
        return 0 if observation.ok else 1

    if args.command == "stats":
        print(json.dumps(service.stats(), indent=2, default=str))
        return 0

    return 0  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
