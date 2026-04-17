"""YAML config parser and validation for Litmux."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from litmux.models import (
    DefaultTest,
    EvalCase,
    LitmuxConfig,
    ModelConfig,
    Provider,
    TestCase,
)


def find_config(path: str | None = None) -> Path:
    """Find the litmux config file."""
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        return p

    for name in ("litmux.yaml", "litmux.yml"):
        if Path(name).exists():
            return Path(name)

    raise FileNotFoundError(
        "No litmux.yaml found. Run 'litmux init' to create one."
    )


def load_config(path: str | None = None) -> LitmuxConfig:
    """Load and validate a litmux config file."""
    config_path = find_config(path)

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ValueError("Config file is empty")

    # Parse models
    raw_models = raw.get("models", [])
    if not raw_models:
        raise ValueError("No models defined in config")

    models = []
    for m in raw_models:
        if "provider" not in m:
            m["provider"] = _infer_provider(m.get("model", ""))
        models.append(ModelConfig(**m))

    # Parse defaultTest
    default_test = None
    if "defaultTest" in raw:
        default_test = DefaultTest(**raw["defaultTest"])

    # Parse tests
    tests = []
    for t in raw.get("tests", []):
        prompt_text, prompt_source = _resolve_prompt(t, config_path.parent)
        t["prompt"] = prompt_text
        t["prompt_source"] = prompt_source
        if "inputs" in t:
            t["prompt"] = _apply_inputs(t["prompt"], t["inputs"])
        tc = TestCase(**t)
        # Merge defaultTest assertions (prepend, test can override)
        if default_test:
            if default_test.inputs:
                merged_inputs = {**default_test.inputs, **tc.inputs}
                tc = tc.model_copy(update={"inputs": merged_inputs})
            if default_test.assertions:
                merged = list(default_test.assertions) + list(tc.assertions)
                tc = tc.model_copy(update={"assertions": merged})
        tests.append(tc)

    # Parse evals
    evals = []
    for e in raw.get("evals", []):
        prompt_text, prompt_source = _resolve_prompt(e, config_path.parent)
        e["prompt"] = prompt_text
        e["prompt_source"] = prompt_source
        # Resolve dataset path relative to config file
        if "dataset" in e:
            try:
                ds_path = _safe_join(config_path.parent, e["dataset"])
            except ValueError:
                raise
            if ds_path.exists():
                e["dataset"] = str(ds_path)
        evals.append(EvalCase(**e))

    return LitmuxConfig(models=models, tests=tests, evals=evals, default_test=default_test)


def _safe_join(base_dir: Path, rel: str) -> Path:
    """Resolve `rel` relative to `base_dir` and reject paths that escape it."""
    base = base_dir.resolve()
    candidate = (base / rel).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise ValueError(
            f"Path '{rel}' resolves outside the config directory; refusing to read."
        )
    return candidate


def _resolve_prompt(test_def: dict, base_dir: Path) -> tuple[str, str]:
    """Resolve prompt from file path or inline text."""
    if "prompt" in test_def:
        raw = test_def["prompt"]
        # Heuristic: treat as a path only if it looks like one.
        looks_like_path = isinstance(raw, str) and (
            raw.endswith((".txt", ".md", ".tmpl")) or "/" in raw or "\\" in raw
        ) and "\n" not in raw
        if looks_like_path:
            prompt_path = _safe_join(base_dir, raw)
            if prompt_path.exists():
                return prompt_path.read_text(), str(raw)
        # Otherwise treat as inline text
        return raw, "inline"
    raise ValueError(f"Test/eval is missing 'prompt' field: {test_def.get('name', '?')}")


def _apply_inputs(prompt: str, inputs: dict[str, str]) -> str:
    """Replace {{variable}} placeholders with input values."""
    for key, value in inputs.items():
        prompt = prompt.replace(f"{{{{{key}}}}}", str(value))
    return prompt


def _infer_provider(model: str) -> str:
    """Auto-detect provider from model name."""
    model_lower = model.lower()
    if any(k in model_lower for k in ("gpt", "o1", "o3", "davinci")):
        return Provider.OPENAI.value
    if any(k in model_lower for k in ("claude",)):
        return Provider.ANTHROPIC.value
    if any(k in model_lower for k in ("gemini",)):
        return Provider.GOOGLE.value
    # HuggingFace models use org/model format (e.g. meta-llama/Llama-3.1-8B-Instruct)
    hf_orgs = (
        "meta-llama", "mistralai", "qwen", "microsoft", "huggingfaceh4",
        "tiiuae", "bigcode", "codellama", "deepseek", "stabilityai",
        "nvidia", "01-ai", "openchat", "teknium", "nousresearch",
    )
    if "/" in model and any(model_lower.startswith(org) for org in hf_orgs):
        return Provider.HUGGINGFACE.value
    raise ValueError(
        f"Cannot infer provider for model '{model}'. "
        f"Specify 'provider' explicitly (openai, anthropic, google, huggingface)."
    )
