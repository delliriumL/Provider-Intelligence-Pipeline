"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from provider_intelligence.config import load_config


@pytest.fixture(autouse=True)
def reset_config_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure env overrides apply and LLM stays offline in tests."""
    monkeypatch.setenv("LLM_MODE", "off")
    load_config.cache_clear()
    yield
    load_config.cache_clear()
