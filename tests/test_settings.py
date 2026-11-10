"""
tests/test_settings.py
----------------------
Unit tests for configs/settings.py.

All tests use monkeypatch to set or remove environment variables so they
never depend on a real .env file.  The lru_cache is cleared before each
test to force a fresh Settings() construction.
"""

import os
import sys
import pytest

# Make sure the project root is importable without installation
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from configs.settings import Settings, get_settings  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_VARS = {
    "GENERATOR_MODEL": "test-model",
    "JUDGE_MODEL": "judge-test",
    "NLI_MODEL": "nli-test",
    "EMBEDDING_MODEL": "embed-test",
    "LLM_API_KEY": "test-api-key",
    "LLM_BASE_URL": "https://api.test.example.com/v1",
}


def _set_all(monkeypatch):
    """Set all required env vars plus the default-bearing RANDOM_SEED."""
    for k, v in _REQUIRED_VARS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("RANDOM_SEED", "42")


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear lru_cache before every test so Settings is re-constructed."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _no_env_file(monkeypatch):
    """
    Prevent pydantic-settings from reading a real .env file by pointing it at
    a nonexistent path.  Override Settings.model_config for each test.
    """
    monkeypatch.setattr(
        "configs.settings.Settings.model_config",
        {
            "env_file": "/nonexistent/.env.does_not_exist",
            "env_file_encoding": "utf-8",
            "extra": "ignore",
            "case_sensitive": False,
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSettingsHappyPath:
    def test_all_variables_set_loads_correctly(self, monkeypatch):
        """All required variables present → Settings loads and values match."""
        _set_all(monkeypatch)

        s = Settings()

        assert s.GENERATOR_MODEL == "test-model"
        assert s.JUDGE_MODEL == "judge-test"
        assert s.NLI_MODEL == "nli-test"
        assert s.EMBEDDING_MODEL == "embed-test"
        assert s.LLM_API_KEY.get_secret_value() == "test-api-key"
        assert s.LLM_BASE_URL == "https://api.test.example.com/v1"
        assert s.RANDOM_SEED == 42

    def test_random_seed_defaults_to_42_when_unset(self, monkeypatch):
        """RANDOM_SEED should default to 42 if the env var is absent."""
        _set_all(monkeypatch)
        monkeypatch.delenv("RANDOM_SEED", raising=False)

        s = Settings()

        assert s.RANDOM_SEED == 42


class TestSettingsValidation:
    def test_missing_model_variable_raises_with_name(self, monkeypatch):
        """Omitting a required model variable → ValidationError naming the var."""
        _set_all(monkeypatch)
        monkeypatch.delenv("NLI_MODEL", raising=False)

        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        assert "NLI_MODEL" in str(exc_info.value).upper() or "nli_model" in str(exc_info.value).lower()

    def test_empty_string_model_variable_raises(self, monkeypatch):
        """An empty-string value for a required field → ValidationError."""
        _set_all(monkeypatch)
        monkeypatch.setenv("GENERATOR_MODEL", "")

        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings()


class TestSettingsSecurity:
    def test_repr_does_not_contain_api_key(self, monkeypatch):
        """repr(settings) must not expose the raw API key value."""
        _set_all(monkeypatch)
        s = Settings()

        # The key value must not appear in repr
        assert "test-api-key" not in repr(s)

    def test_to_loggable_dict_does_not_contain_api_key(self, monkeypatch):
        """to_loggable_dict() must not include LLM_API_KEY at all."""
        _set_all(monkeypatch)
        s = Settings()

        loggable = s.to_loggable_dict()

        assert "LLM_API_KEY" not in loggable
        assert "test-api-key" not in loggable.values()
