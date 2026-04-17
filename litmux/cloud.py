"""Litmux Cloud — HTTP client for syncing results to app.litmux.dev."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from litmux.models import EvalResult, TestResult

DEFAULT_API_URL = "https://api.litmux.dev"
CONFIG_DIR = Path.home() / ".litmux"
CONFIG_FILE = CONFIG_DIR / "config.json"

# Network errors that should be silently swallowed during best-effort sync.
_NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    httpx.HTTPError,
    httpx.InvalidURL,
    ValueError,
    OSError,
)


class CloudConfigError(Exception):
    """Raised for misconfigured LITMUX_API_URL."""


def get_api_url() -> str:
    """Return the configured API URL, validated for HTTPS.

    Raises CloudConfigError if a non-HTTPS URL is set without
    LITMUX_API_URL_ALLOW_INSECURE=1.
    """
    url = os.environ.get("LITMUX_API_URL", DEFAULT_API_URL).rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise CloudConfigError(
            f"LITMUX_API_URL is not a valid URL: {url!r}"
        )
    if parsed.scheme != "https" and os.environ.get("LITMUX_API_URL_ALLOW_INSECURE") != "1":
        raise CloudConfigError(
            f"LITMUX_API_URL must use HTTPS (got {url!r}). "
            f"Set LITMUX_API_URL_ALLOW_INSECURE=1 to override for local development."
        )
    return url


def _api_host(url: str) -> str:
    return urlparse(url).netloc


def _read_config() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(CONFIG_FILE.read_text()) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def _get_token_for(host: str) -> str | None:
    """Return saved token only if it was minted for this host."""
    cfg = _read_config()
    if cfg.get("host") and cfg.get("host") != host:
        return None
    return cfg.get("token")


def is_logged_in() -> bool:
    """Check if user is authenticated with Litmux Cloud."""
    return bool(_read_config().get("token"))


def save_token(token: str) -> None:
    """Save auth token to ~/.litmux/config.json with 0600 perms.

    The token is bound to the host of the currently configured API URL.
    """
    try:
        host = _api_host(get_api_url())
    except CloudConfigError:
        host = _api_host(DEFAULT_API_URL)

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        CONFIG_DIR.chmod(stat.S_IRWXU)  # 0700
    except OSError:
        pass

    CONFIG_FILE.write_text(json.dumps({"token": token, "host": host}))
    try:
        CONFIG_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600
    except OSError:
        pass


def remove_token() -> None:
    """Remove saved auth token."""
    if CONFIG_FILE.exists():
        CONFIG_FILE.unlink()


def _get_git_info() -> dict[str, str | None]:
    """Get current git branch, commit, and message."""
    info: dict[str, str | None] = {"branch": None, "commit": None, "message": None}
    try:
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["message"] = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%s"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info


def _auth_headers(api_url: str) -> dict[str, str] | None:
    """Build auth headers, ensuring token host matches the API URL."""
    token = _get_token_for(_api_host(api_url))
    if not token:
        return None
    return {"Authorization": f"Bearer {token}"}


def sync_run(
    results: list[TestResult],
    duration_ms: float,
    trigger: str = "cli",
) -> str | None:
    """Send run results to Litmux Cloud. Returns run ID or None on failure."""
    try:
        api_url = get_api_url()
    except CloudConfigError:
        return None

    headers = _auth_headers(api_url)
    if not headers:
        return None

    git = _get_git_info()
    project = os.path.basename(os.getcwd())
    run_id = str(uuid.uuid4())

    test_rows = []
    for tr in results:
        for mr in tr.model_results:
            assertion_details = [
                {
                    "type": ar.assertion.type.value,
                    "passed": ar.passed,
                    "message": ar.message,
                    "value": str(ar.assertion.value) if ar.assertion.value is not None else None,
                }
                for ar in mr.assertion_results
            ]
            test_rows.append({
                "test_name": tr.test_case.name,
                "model": mr.model_name,
                "provider": mr.model_config_obj.provider.value,
                "passed": mr.passed,
                "latency_ms": round(mr.latency_ms, 1),
                "cost_usd": mr.cost_usd,
                "input_tokens": mr.input_tokens,
                "output_tokens": mr.output_tokens,
                "output": mr.output[:5000],
                "error": mr.error,
                "assertions_passed": mr.pass_count,
                "assertions_total": mr.total_assertions,
                "assertion_details": assertion_details,
            })

    try:
        resp = httpx.post(
            f"{api_url}/v1/runs",
            json={
                "id": run_id,
                "project": project,
                "git": git,
                "duration_ms": duration_ms,
                "trigger": trigger,
                "total_tests": len(results),
                "passed_tests": sum(1 for r in results if r.all_passed),
                "results": test_rows,
            },
            headers=headers,
            timeout=10,
        )
        if resp.status_code == 200:
            return run_id
    except _NETWORK_ERRORS:
        pass
    return None


def sync_eval(
    eval_results: list[EvalResult],
    run_id: str,
) -> None:
    """Send eval results to Litmux Cloud."""
    try:
        api_url = get_api_url()
    except CloudConfigError:
        return

    headers = _auth_headers(api_url)
    if not headers:
        return

    for er in eval_results:
        row_data = [
            {
                "row_index": rr.row_index,
                "inputs": rr.inputs,
                "expected": rr.expected,
                "actual_output": rr.actual_output[:5000],
                "passed": rr.passed,
                "judge_score": rr.judge_score,
                "latency_ms": round(rr.latency_ms, 1),
                "cost_usd": rr.cost_usd,
                "assertion_details": [
                    {"type": a.assertion.type.value, "passed": a.passed, "message": a.message}
                    for a in rr.assertion_results
                ],
            }
            for rr in er.row_results
        ]

        try:
            httpx.post(
                f"{api_url}/v1/evals",
                json={
                    "run_id": run_id,
                    "eval_name": er.eval_case.name,
                    "model": er.model_name,
                    "dataset_name": er.eval_case.dataset,
                    "total_rows": len(er.row_results),
                    "passed_rows": sum(1 for r in er.row_results if r.passed),
                    "avg_score": er.avg_score,
                    "avg_latency_ms": er.avg_latency_ms,
                    "total_cost_usd": er.total_cost_usd,
                    "rows": row_data,
                },
                headers=headers,
                timeout=10,
            )
        except _NETWORK_ERRORS:
            pass


def get_history(limit: int = 50, project: str | None = None) -> list[dict[str, Any]]:
    """Fetch recent runs from Litmux Cloud."""
    try:
        api_url = get_api_url()
    except CloudConfigError:
        return []

    headers = _auth_headers(api_url)
    if not headers:
        return []

    params: dict[str, Any] = {"limit": limit}
    if project:
        params["project"] = project

    try:
        resp = httpx.get(
            f"{api_url}/v1/runs",
            params=params,
            headers=headers,
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data if isinstance(data, list) else data.get("runs", [])
    except _NETWORK_ERRORS:
        pass
    return []
