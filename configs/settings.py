"""
configs/settings.py
-------------------
Central configuration for the raguard project.

All model names, API keys, and URLs are read exclusively from environment
variables (or a local .env file).  No literal values appear here.
"""

import functools
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Project-root-relative path constants (not env-driven)
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_RAW_DIR: Path = _PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = _PROJECT_ROOT / "data" / "processed"
RUNS_DIR: Path = _PROJECT_ROOT / "runs"

RUNS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Secrets (LLM_API_KEY) are stored as ``SecretStr`` and are intentionally
    omitted from ``repr()``, ``str()``, and :meth:`to_loggable_dict`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Pydantic-settings will only read the listed .env file; it will not
        # fall back to the system environment unless we keep populate_by_name.
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Required fields — no defaults; missing or empty → validation error
    # ------------------------------------------------------------------

    GENERATOR_MODEL: str
    JUDGE_MODEL: str
    NLI_MODEL: str
    EMBEDDING_MODEL: str
    LLM_API_KEY: SecretStr
    LLM_BASE_URL: str

    # ------------------------------------------------------------------
    # Optional fields with safe defaults
    # ------------------------------------------------------------------

    RANDOM_SEED: int = 42

    # ------------------------------------------------------------------
    # Validators: reject empty strings for every str model field
    # ------------------------------------------------------------------

    @field_validator("GENERATOR_MODEL", "JUDGE_MODEL", "NLI_MODEL", "EMBEDDING_MODEL", "LLM_BASE_URL")
    @classmethod
    def _reject_empty(cls, value: str, info) -> str:  # noqa: ANN001
        if not value.strip():
            raise ValueError(
                f"Environment variable '{info.field_name}' must not be empty. "
                "Set it in your shell or in a .env file."
            )
        return value

    @field_validator("LLM_API_KEY")
    @classmethod
    def _reject_empty_secret(cls, value: SecretStr, info) -> SecretStr:  # noqa: ANN001
        if not value.get_secret_value().strip():
            raise ValueError(
                f"Environment variable '{info.field_name}' must not be empty. "
                "Set it in your shell or in a .env file."
            )
        return value

    # ------------------------------------------------------------------
    # Safe serialisation — never expose the API key
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        safe = self.to_loggable_dict()
        pairs = ", ".join(f"{k}={v!r}" for k, v in safe.items())
        return f"Settings({pairs})"

    def __str__(self) -> str:  # pragma: no cover
        return self.__repr__()

    def to_loggable_dict(self) -> dict:
        """
        Return a copy of the settings suitable for experiment logs.

        ``LLM_API_KEY`` is excluded so the dict can be saved next to results
        without leaking credentials.
        """
        return {
            "GENERATOR_MODEL": self.GENERATOR_MODEL,
            "JUDGE_MODEL": self.JUDGE_MODEL,
            "NLI_MODEL": self.NLI_MODEL,
            "EMBEDDING_MODEL": self.EMBEDDING_MODEL,
            "LLM_BASE_URL": self.LLM_BASE_URL,
            "RANDOM_SEED": self.RANDOM_SEED,
        }


# ---------------------------------------------------------------------------
# Cached accessor
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application settings, loading them once and caching."""
    return Settings()
