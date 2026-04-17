"""Tests for response caching."""

from __future__ import annotations

import os

import pytest

from litmux.cache import _cache_key, get_cached, set_cached, clear_cache, CACHE_DIR


@pytest.fixture(autouse=True)
def _clean_cache(tmp_path, monkeypatch):
    """Use a temp directory for cache during tests."""
    import litmux.cache as cache_module
    monkeypatch.setattr(cache_module, "CACHE_DIR", tmp_path / ".litmux_cache")
    monkeypatch.delenv("LITMUX_NO_CACHE", raising=False)
    yield


class TestCacheKey:
    def test_deterministic(self) -> None:
        k1 = _cache_key("gpt-4o", "hello", 0.0, 1024)
        k2 = _cache_key("gpt-4o", "hello", 0.0, 1024)
        assert k1 == k2

    def test_different_inputs(self) -> None:
        k1 = _cache_key("gpt-4o", "hello", 0.0, 1024)
        k2 = _cache_key("gpt-4o", "goodbye", 0.0, 1024)
        assert k1 != k2

    def test_different_models(self) -> None:
        k1 = _cache_key("gpt-4o", "hello", 0.0, 1024)
        k2 = _cache_key("gpt-4o-mini", "hello", 0.0, 1024)
        assert k1 != k2


class TestCacheGetSet:
    def test_miss_returns_none(self) -> None:
        assert get_cached("model", "prompt", 0.0, 1024) is None

    def test_set_then_get(self) -> None:
        result = {"output": "hello", "input_tokens": 5, "output_tokens": 3}
        set_cached("model", "prompt", 0.0, 1024, result)
        cached = get_cached("model", "prompt", 0.0, 1024)
        assert cached is not None
        assert cached["output"] == "hello"

    def test_no_cache_env_disables(self, monkeypatch) -> None:
        monkeypatch.setenv("LITMUX_NO_CACHE", "1")
        result = {"output": "hello", "input_tokens": 5, "output_tokens": 3}
        set_cached("model", "prompt", 0.0, 1024, result)
        assert get_cached("model", "prompt", 0.0, 1024) is None


class TestClearCache:
    def test_clear_empty(self) -> None:
        assert clear_cache() == 0

    def test_clear_with_entries(self) -> None:
        result = {"output": "hello", "input_tokens": 5, "output_tokens": 3}
        set_cached("m1", "p1", 0.0, 1024, result)
        set_cached("m2", "p2", 0.0, 1024, result)
        count = clear_cache()
        assert count == 2
        assert get_cached("m1", "p1", 0.0, 1024) is None
