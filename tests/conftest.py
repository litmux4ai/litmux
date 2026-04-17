"""Shared fixtures for Litmux tests."""

from __future__ import annotations

import pytest
from pathlib import Path

from litmux.models import (
    Assertion,
    AssertionType,
    ModelConfig,
    ModelRunResult,
    Provider,
    TestCase,
)


@pytest.fixture
def sample_model_config() -> ModelConfig:
    return ModelConfig(provider=Provider.OPENAI, model="gpt-4o-mini")


@pytest.fixture
def sample_run_result(sample_model_config: ModelConfig) -> ModelRunResult:
    return ModelRunResult(
        model_config_obj=sample_model_config,
        output="Hello world! This is a test response.",
        latency_ms=250.0,
        input_tokens=10,
        output_tokens=20,
        cost_usd=0.000015,
    )


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    """Create a temporary litmux.yaml for testing."""
    config = tmp_path / "litmux.yaml"
    config.write_text(
        """\
models:
  - provider: openai
    model: gpt-4o-mini
    temperature: 0.0
    max_tokens: 256

tests:
  - name: greeting-test
    prompt: "Say hello"
    assert:
      - type: contains
        value: hello
"""
    )
    return config


@pytest.fixture
def tmp_config_with_prompt_file(tmp_path: Path) -> Path:
    """Create a config that references a prompt file."""
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Classify this ticket: {{ticket}}")

    config = tmp_path / "litmux.yaml"
    config.write_text(
        """\
models:
  - model: gpt-4o-mini

tests:
  - name: classify-test
    prompt: prompt.txt
    inputs:
      ticket: "My internet is down"
    assert:
      - type: contains
        value: network
"""
    )
    return config
