"""Response cache for Litmux — avoids redundant API calls during iteration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

CACHE_DIR = Path(".litmux_cache")


def _cache_key(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    """Generate a deterministic cache key from request parameters."""
    raw = f"{model}|{prompt}|{temperature}|{max_tokens}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_cached(model: str, prompt: str, temperature: float, max_tokens: int) -> dict[str, Any] | None:
    """Return cached response if available, else None."""
    if os.environ.get("LITMUX_NO_CACHE"):
        return None

    key = _cache_key(model, prompt, temperature, max_tokens)
    cache_file = CACHE_DIR / f"{key}.json"

    if cache_file.exists():
        try:
            with open(cache_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def set_cached(
    model: str, prompt: str, temperature: float, max_tokens: int,
    result: dict[str, Any],
) -> None:
    """Store a response in the cache."""
    if os.environ.get("LITMUX_NO_CACHE"):
        return

    CACHE_DIR.mkdir(exist_ok=True)
    key = _cache_key(model, prompt, temperature, max_tokens)
    cache_file = CACHE_DIR / f"{key}.json"

    try:
        with open(cache_file, "w") as f:
            json.dump(result, f)
    except OSError:
        pass  # Cache write failure is non-fatal


def clear_cache() -> int:
    """Clear all cached responses. Returns number of entries removed."""
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for f in CACHE_DIR.glob("*.json"):
        f.unlink()
        count += 1
    return count
