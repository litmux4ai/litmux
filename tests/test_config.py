"""Tests for YAML config parsing and validation."""

from __future__ import annotations

import pytest
from pathlib import Path

from litmux.config import (
    _apply_inputs,
    _infer_provider,
    _resolve_prompt,
    find_config,
    load_config,
)
from litmux.models import Provider


class TestInferProvider:
    def test_openai_models(self) -> None:
        assert _infer_provider("gpt-4o") == "openai"
        assert _infer_provider("gpt-4o-mini") == "openai"
        assert _infer_provider("o1") == "openai"
        assert _infer_provider("o3-mini") == "openai"

    def test_anthropic_models(self) -> None:
        assert _infer_provider("claude-3-5-sonnet-20241022") == "anthropic"
        assert _infer_provider("claude-3-haiku-20240307") == "anthropic"

    def test_google_models(self) -> None:
        assert _infer_provider("gemini-1.5-pro") == "google"
        assert _infer_provider("gemini-2.0-flash") == "google"

    def test_huggingface_models(self) -> None:
        assert _infer_provider("meta-llama/Llama-3.1-8B-Instruct") == "huggingface"
        assert _infer_provider("mistralai/Mistral-7B-Instruct-v0.3") == "huggingface"
        assert _infer_provider("Qwen/Qwen2.5-72B-Instruct") == "huggingface"

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot infer provider"):
            _infer_provider("some-unknown-model")


class TestApplyInputs:
    def test_single_variable(self) -> None:
        result = _apply_inputs("Hello {{name}}", {"name": "World"})
        assert result == "Hello World"

    def test_multiple_variables(self) -> None:
        result = _apply_inputs(
            "{{greeting}} {{name}}!",
            {"greeting": "Hi", "name": "Alice"},
        )
        assert result == "Hi Alice!"

    def test_no_variables(self) -> None:
        result = _apply_inputs("No variables here", {})
        assert result == "No variables here"

    def test_missing_variable_untouched(self) -> None:
        result = _apply_inputs("Hello {{name}}", {})
        assert result == "Hello {{name}}"


class TestResolvePrompt:
    def test_inline_prompt(self, tmp_path: Path) -> None:
        text, source = _resolve_prompt({"prompt": "Say hello"}, tmp_path)
        assert text == "Say hello"
        assert source == "inline"

    def test_file_prompt(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "my_prompt.txt"
        prompt_file.write_text("This is from a file")
        text, source = _resolve_prompt({"prompt": "my_prompt.txt"}, tmp_path)
        assert text == "This is from a file"
        assert source == "my_prompt.txt"

    def test_missing_prompt_key(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="missing 'prompt' field"):
            _resolve_prompt({"name": "test"}, tmp_path)


class TestFindConfig:
    def test_explicit_path(self, tmp_path: Path) -> None:
        cfg = tmp_path / "custom.yaml"
        cfg.write_text("models: []")
        assert find_config(str(cfg)) == cfg

    def test_missing_explicit_path(self) -> None:
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            find_config("/nonexistent/path.yaml")

    def test_no_config_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="No litmux.yaml found"):
            find_config()


class TestLoadConfig:
    def test_load_valid_config(self, tmp_config: Path) -> None:
        config = load_config(str(tmp_config))
        assert len(config.models) == 1
        assert config.models[0].model == "gpt-4o-mini"
        assert len(config.tests) == 1
        assert config.tests[0].name == "greeting-test"

    def test_load_config_with_prompt_file(self, tmp_config_with_prompt_file: Path) -> None:
        config = load_config(str(tmp_config_with_prompt_file))
        assert len(config.tests) == 1
        assert "My internet is down" in config.tests[0].prompt

    def test_load_empty_config(self, tmp_path: Path) -> None:
        cfg = tmp_path / "litmux.yaml"
        cfg.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_config(str(cfg))

    def test_load_config_no_models(self, tmp_path: Path) -> None:
        cfg = tmp_path / "litmux.yaml"
        cfg.write_text("tests: []")
        with pytest.raises(ValueError, match="No models"):
            load_config(str(cfg))

    def test_provider_auto_inferred(self, tmp_path: Path) -> None:
        cfg = tmp_path / "litmux.yaml"
        cfg.write_text(
            """\
models:
  - model: claude-3-5-sonnet-20241022
tests: []
"""
        )
        config = load_config(str(cfg))
        assert config.models[0].provider == Provider.ANTHROPIC


class TestDefaultTest:
    def test_default_assertions_merged(self, tmp_path: Path) -> None:
        cfg = tmp_path / "litmux.yaml"
        cfg.write_text(
            """\
models:
  - model: gpt-4o-mini

defaultTest:
  assert:
    - type: latency-less-than
      value: 5000

tests:
  - name: test1
    prompt: "Say hello"
    assert:
      - type: contains
        value: hello
"""
        )
        config = load_config(str(cfg))
        # Should have the default assertion (latency) + test-specific one (contains)
        assert len(config.tests[0].assertions) == 2
        types = [a.type.value for a in config.tests[0].assertions]
        assert "latency-less-than" in types
        assert "contains" in types

    def test_default_test_none_when_absent(self, tmp_path: Path) -> None:
        cfg = tmp_path / "litmux.yaml"
        cfg.write_text(
            """\
models:
  - model: gpt-4o-mini
tests: []
"""
        )
        config = load_config(str(cfg))
        assert config.default_test is None
