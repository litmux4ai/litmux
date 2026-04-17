"""Tests for dataset loading and generation."""

from __future__ import annotations

import json
import pytest

from litmux.dataset import (
    extract_variables,
    apply_input_mapping,
)


# ─── Tests for _parse_json_response ─────────────────────────────


class TestParseJsonResponse:
    """Tests for the robust JSON parser."""

    def test_clean_json_array(self) -> None:
        from litmux.dataset import _parse_json_response

        content = json.dumps([{"ticket": "help", "difficulty": "easy"}])
        result = _parse_json_response(content)
        assert len(result) == 1
        assert result[0]["ticket"] == "help"

    def test_json_with_markdown_fences(self) -> None:
        from litmux.dataset import _parse_json_response

        content = '```json\n[{"ticket": "help"}]\n```'
        result = _parse_json_response(content)
        assert len(result) == 1
        assert result[0]["ticket"] == "help"

    def test_json_with_plain_fences(self) -> None:
        from litmux.dataset import _parse_json_response

        content = '```\n[{"ticket": "help"}]\n```'
        result = _parse_json_response(content)
        assert len(result) == 1

    def test_json_with_leading_prose(self) -> None:
        from litmux.dataset import _parse_json_response

        content = 'Here are the scenarios:\n[{"ticket": "help"}]'
        result = _parse_json_response(content)
        assert len(result) == 1

    def test_json_with_trailing_prose(self) -> None:
        from litmux.dataset import _parse_json_response

        content = '[{"ticket": "help"}]\n\nLet me know if you need more!'
        result = _parse_json_response(content)
        assert len(result) == 1

    def test_empty_array(self) -> None:
        from litmux.dataset import _parse_json_response

        result = _parse_json_response("[]")
        assert result == []

    def test_invalid_content_raises(self) -> None:
        from litmux.dataset import _parse_json_response

        with pytest.raises(ValueError, match="Failed to parse"):
            _parse_json_response("This is not JSON at all, no arrays here.")

    def test_non_array_json_raises(self) -> None:
        from litmux.dataset import _parse_json_response

        with pytest.raises(ValueError, match="Failed to parse"):
            _parse_json_response('{"key": "value"}')

    def test_wrapped_scenarios_object(self) -> None:
        from litmux.dataset import _parse_json_response

        content = json.dumps({"scenarios": [{"ticket": "help", "difficulty": "easy"}]})
        result = _parse_json_response(content)
        assert len(result) == 1
        assert result[0]["ticket"] == "help"

    def test_fences_with_whitespace(self) -> None:
        from litmux.dataset import _parse_json_response

        content = '  ```json\n  [{"a": 1}]  \n```  '
        result = _parse_json_response(content)
        assert len(result) == 1


# ─── Tests for provider routing ──────────────────────────────────


class TestInferGenerationProvider:
    """Tests for provider detection in generation context."""

    def test_anthropic_haiku(self) -> None:
        from litmux.dataset import _infer_generation_provider

        assert _infer_generation_provider("claude-haiku-4-5-20251001") == "anthropic"

    def test_anthropic_sonnet(self) -> None:
        from litmux.dataset import _infer_generation_provider

        assert _infer_generation_provider("claude-sonnet-4-6") == "anthropic"

    def test_openai_gpt4o(self) -> None:
        from litmux.dataset import _infer_generation_provider

        assert _infer_generation_provider("gpt-4o-mini") == "openai"

    def test_openai_gpt4(self) -> None:
        from litmux.dataset import _infer_generation_provider

        assert _infer_generation_provider("gpt-4o") == "openai"

    def test_unsupported_raises(self) -> None:
        from litmux.dataset import _infer_generation_provider

        with pytest.raises(ValueError, match="not supported for dataset generation"):
            _infer_generation_provider("gemini-2.0-flash")

    def test_unknown_raises(self) -> None:
        from litmux.dataset import _infer_generation_provider

        with pytest.raises(ValueError, match="not supported for dataset generation"):
            _infer_generation_provider("totally-unknown-model")


# ─── Tests for generate_dataset signature ────────────────────────


class TestGenerateDatasetSignature:
    """Verify the default model is claude-3-5-haiku-20241022."""

    def test_default_model(self) -> None:
        import inspect
        from litmux.dataset import generate_dataset

        sig = inspect.signature(generate_dataset)
        default = sig.parameters["model"].default
        assert default == "claude-haiku-4-5-20251001"


# ─── Tests for meta prompt building ─────────────────────────────


class TestBuildMetaPrompt:
    """Tests for meta prompt construction."""

    def test_contains_variables(self) -> None:
        from litmux.dataset import _build_meta_prompt

        prompt = _build_meta_prompt(
            template="Classify: {{ticket}}",
            variables=["ticket"],
            description="classifier",
            n=10,
        )
        assert "ticket" in prompt
        assert "10" in prompt
        assert "classifier" in prompt

    def test_contains_distribution(self) -> None:
        from litmux.dataset import _build_meta_prompt

        prompt = _build_meta_prompt(
            template="{{text}}",
            variables=["text"],
            description="test",
            n=5,
        )
        assert "happy_path" in prompt
        assert "edge_case" in prompt
        assert "adversarial" in prompt

    def test_contains_json_instruction(self) -> None:
        from litmux.dataset import _build_meta_prompt

        prompt = _build_meta_prompt(
            template="{{x}}",
            variables=["x"],
            description="test",
            n=5,
        )
        assert "JSON" in prompt


# ─── Existing function tests ────────────────────────────────────


class TestExtractVariables:
    def test_single(self) -> None:
        assert extract_variables("Hello {{name}}") == ["name"]

    def test_multiple(self) -> None:
        assert extract_variables("{{a}} and {{b}}") == ["a", "b"]

    def test_none(self) -> None:
        assert extract_variables("no variables") == []


class TestApplyInputMapping:
    def test_mapping(self) -> None:
        result = apply_input_mapping(
            "Classify: {{ticket}}",
            {"ticket": "text"},
            {"text": "My internet is down"},
        )
        assert result == "Classify: My internet is down"

    def test_missing_column(self) -> None:
        result = apply_input_mapping(
            "Classify: {{ticket}}",
            {"ticket": "text"},
            {"other_col": "value"},
        )
        assert result == "Classify: "
