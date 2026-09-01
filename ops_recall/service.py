"""Composition root.

One place that knows how to build every component from settings, so the API,
the CLI and the tests all get an identically wired system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ops_recall.agent.graph import IncidentAgent
from ops_recall.agent.reconstruct import build_reconstructor
from ops_recall.config import Settings, get_settings
from ops_recall.ingest.pipeline import build_index
from ops_recall.retrieval.store import IncidentStore
from ops_recall.telemetry.mock import build_provider
from ops_recall.telemetry.provider import TelemetryProvider

logger = logging.getLogger(__name__)


def build_client(settings: Settings) -> Any | None:
    """An Anthropic client, or None when no credentials are resolvable.

    The SDK resolves credentials from `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`
    or an `ant auth login` profile, and constructs happily with none of them --
    the failure surfaces on the first request. Checking `auth_headers` here
    turns that into a startup-time decision, so the service can report which
    reasoner it is actually running instead of erroring per alert.
    """
    if settings.reasoner != "claude":
        return None
    try:
        import anthropic

        client = anthropic.Anthropic()
        if not client.auth_headers:
            logger.warning(
                "no Anthropic credentials resolved (ANTHROPIC_API_KEY, "
                "ANTHROPIC_AUTH_TOKEN or `ant auth login`)"
            )
            return None
        return client
    except Exception as exc:
        logger.warning("no Anthropic client available: %s", exc)
        return None


@dataclass
class Service:
    settings: Settings
    store: IncidentStore
    provider: TelemetryProvider
    agent: IncidentAgent
    counts: dict[str, int]

    @property
    def reasoner(self) -> str:
        return self.agent.reconstructor.name

    def stats(self) -> dict[str, Any]:
        return {
            "corpus": self.counts,
            "index": self.store.stats(),
            "telemetry": self.provider.name,
            "reasoner": self.reasoner,
            "model": self.settings.model if self.reasoner == "claude" else None,
            "agentic_checks": self.agent.investigator is not None,
            "half_life_days": self.settings.half_life_days,
        }


def build_service(
    settings: Settings | None = None,
    store: IncidentStore | None = None,
    provider: TelemetryProvider | None = None,
    client: Any = None,
) -> Service:
    settings = settings or get_settings()
    if store is None:
        store, counts = build_index(settings)
    else:
        counts = dict(store.index_counts)
    provider = provider or build_provider(settings)
    client = client if client is not None else build_client(settings)

    reconstructor = None
    if settings.reasoner == "claude" and client is None:
        # Degrade loudly rather than failing every request: retrieval and
        # evidence are still worth serving, and the response says which
        # reasoner produced it.
        from ops_recall.agent.reconstruct import TemplateReconstructor

        logger.warning(
            "reasoner=claude but no credentials resolved; falling back to the "
            "template reconstructor. Responses will be labeled reasoner=template."
        )
        reconstructor = TemplateReconstructor(settings)
    elif settings.reasoner != "claude":
        reconstructor = build_reconstructor(settings, client)

    agent = IncidentAgent(
        store=store,
        provider=provider,
        settings=settings,
        client=client,
        reconstructor=reconstructor,
    )
    return Service(
        settings=settings, store=store, provider=provider, agent=agent, counts=counts
    )
