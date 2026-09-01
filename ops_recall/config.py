"""Runtime configuration.

Every knob is settable through the environment with an `OPS_RECALL_` prefix,
e.g. `OPS_RECALL_HALF_LIFE_DAYS=90`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPS_RECALL_", env_file=".env", extra="ignore"
    )

    # --- storage -------------------------------------------------------
    qdrant_url: str | None = Field(
        default=None,
        description="http://host:6333 for a real cluster. Unset uses an embedded "
        "local Qdrant, which is enough for a single-team corpus.",
    )
    qdrant_api_key: str | None = None
    qdrant_path: Path | None = Field(
        default=None,
        description="On-disk path for embedded Qdrant. Unset means in-memory "
        "(rebuilt from sources on every start).",
    )
    collection: str = "incidents"

    # --- embeddings ----------------------------------------------------
    embedder: Literal["hashing", "voyage"] = "hashing"
    embedding_dim: int = 512
    voyage_model: str = "voyage-3"
    voyage_api_key: str | None = None

    # --- retrieval -----------------------------------------------------
    candidate_limit: int = 50
    top_k: int = 3
    min_similarity: float = Field(
        default=0.30,
        description="Below this the system says 'no close historical match' "
        "instead of forcing a bad analogy onto a novel incident.",
    )
    weight_semantic: float = 0.5
    weight_lexical: float = 0.3
    weight_keyword: float = 0.2

    # --- temporal awareness --------------------------------------------
    half_life_days: float = Field(
        default=365.0,
        description="An incident one half-life old carries half the *decayable* "
        "part of its score. Tuned for infrastructure knowledge, which goes stale "
        "over quarters rather than weeks.",
    )
    recency_floor: float = Field(
        default=0.6,
        description="Floor on the decay multiplier. Decay is meant to break ties "
        "between comparable matches, not to bury history: a three-year-old exact "
        "match is stale, not irrelevant.",
    )

    # --- reasoning -----------------------------------------------------
    reasoner: Literal["claude", "template"] = "claude"
    model: str = "claude-opus-5"
    max_tokens: int = 16000
    agentic_checks: bool = Field(
        default=True,
        description="Let the model run extra telemetry tools of its own choosing "
        "on top of the probes replayed from history.",
    )
    max_agentic_iterations: int = 6
    investigation_system: str = Field(
        default="",
        description="Optional extra system prompt for the investigation step; "
        "empty means the tool descriptions carry the instructions.",
    )

    # --- data sources --------------------------------------------------
    data_dir: Path = REPO_ROOT / "data" / "seed"
    telemetry_fixture: Path = REPO_ROOT / "data" / "telemetry" / "current_state.json"
    slack_bot_token: str | None = None
    slack_channels: list[str] = Field(default_factory=list)
    pagerduty_api_key: str | None = None

    def qdrant_location(self) -> dict:
        """Kwargs for `QdrantClient(...)` covering all three deployment shapes."""
        if self.qdrant_url:
            return {"url": self.qdrant_url, "api_key": self.qdrant_api_key}
        if self.qdrant_path:
            return {"path": str(self.qdrant_path)}
        return {"location": ":memory:"}


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_settings(settings: Settings) -> None:
    """Used by tests and the CLI to install an explicit configuration."""
    global _settings
    _settings = settings
