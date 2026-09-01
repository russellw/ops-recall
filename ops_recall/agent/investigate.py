"""The agentic step: let Claude run its own read-only checks.

Everything before this point is deterministic -- retrieval, then replaying the
probes attached to the retrieved incidents. That covers the case where the
archive already knows what to look at. It does not cover the case where the
archive is *nearly* right: the alert resembles a lock incident, the lock probe
comes back refuted, and the useful next question ("is the service leaking
connections instead?") is one no post-mortem wrote down.

So the model gets the same telemetry surface as a tool loop, capped, read-only,
and with every observation it collects recorded for the audit trail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from ops_recall.agent.prompts import (
    INVESTIGATION_PROMPT,
    build_user_message,
)
from ops_recall.agent.tools import ToolContext, build_tools
from ops_recall.config import Settings
from ops_recall.models import Alert, EvidenceCheck, RetrievedIncident, TelemetryObservation

logger = logging.getLogger(__name__)


@dataclass
class Investigation:
    observations: list[TelemetryObservation]
    notes: str = ""
    iterations: int = 0
    error: str | None = None


class AgenticInvestigator:
    def __init__(self, client, settings: Settings, ranker=None) -> None:
        self.client = client
        self.settings = settings
        self.ranker = ranker

    def investigate(
        self,
        alert: Alert,
        retrieved: Sequence[RetrievedIncident],
        evidence: Sequence[EvidenceCheck],
        provider,
    ) -> Investigation:
        context = ToolContext(provider=provider, ranker=self.ranker)
        tools = build_tools(context)

        prompt = "\n\n".join(
            [
                INVESTIGATION_PROMPT,
                build_user_message(alert, retrieved, evidence),
            ]
        )

        notes: list[str] = []
        iterations = 0
        try:
            runner = self.client.beta.messages.tool_runner(
                model=self.settings.model,
                max_tokens=self.settings.max_tokens,
                system=self.settings.investigation_system or None,
                tools=tools,
                messages=[{"role": "user", "content": prompt}],
            )
            for message in runner:
                iterations += 1
                notes.extend(
                    block.text for block in message.content if block.type == "text"
                )
                if iterations >= self.settings.max_agentic_iterations:
                    # The cap is a cost and latency guard, not a correctness one:
                    # whatever was observed so far still reaches the
                    # reconstruction.
                    logger.info(
                        "agentic investigation hit the iteration cap (%s)", iterations
                    )
                    break
        except Exception as exc:  # investigation is best-effort by design
            logger.warning("agentic investigation failed: %s", exc)
            return Investigation(
                observations=context.observations,
                notes="\n".join(notes).strip(),
                iterations=iterations,
                error=str(exc),
            )

        return Investigation(
            observations=context.observations,
            notes="\n".join(n for n in notes if n).strip(),
            iterations=iterations,
        )
