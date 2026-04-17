"""Tests for assertion evaluators."""

from __future__ import annotations

import json

import pytest

from litmux.evaluator import evaluate_assertions, _evaluate_single, _strip_markdown_fences
from litmux.models import (
    Assertion,
    AssertionType,
    ModelConfig,
    ModelRunResult,
    Provider,
)


def _make_result(output: str = "", **kwargs) -> ModelRunResult:
    mc = ModelConfig(provider=Provider.OPENAI, model="gpt-4o-mini")
    return ModelRunResult(model_config_obj=mc, output=output, **kwargs)


class TestContains:
    def test_passes(self) -> None:
        result = _make_result("Hello world")
        a = Assertion(type=AssertionType.CONTAINS, value="hello")
        ar = _evaluate_single(result, a)
        assert ar.passed is True

    def test_fails(self) -> None:
        result = _make_result("Goodbye world")
        a = Assertion(type=AssertionType.CONTAINS, value="hello")
        ar = _evaluate_single(result, a)
        assert ar.passed is False
        assert "does not contain" in ar.message

    def test_case_insensitive(self) -> None:
        result = _make_result("HELLO WORLD")
        a = Assertion(type=AssertionType.CONTAINS, value="hello")
        ar = _evaluate_single(result, a)
        assert ar.passed is True


class TestNotContains:
    def test_passes(self) -> None:
        result = _make_result("Hello world")
        a = Assertion(type=AssertionType.NOT_CONTAINS, value="goodbye")
        ar = _evaluate_single(result, a)
        assert ar.passed is True

    def test_fails(self) -> None:
        result = _make_result("Hello world")
        a = Assertion(type=AssertionType.NOT_CONTAINS, value="hello")
        ar = _evaluate_single(result, a)
        assert ar.passed is False


class TestRegex:
    def test_passes(self) -> None:
        result = _make_result("Order #12345 confirmed")
        a = Assertion(type=AssertionType.REGEX, value=r"#\d{5}")
        ar = _evaluate_single(result, a)
        assert ar.passed is True

    def test_fails(self) -> None:
        result = _make_result("No order here")
        a = Assertion(type=AssertionType.REGEX, value=r"#\d{5}")
        ar = _evaluate_single(result, a)
        assert ar.passed is False


class TestJsonValid:
    def test_valid_json(self) -> None:
        result = _make_result('{"key": "value"}')
        a = Assertion(type=AssertionType.JSON_VALID)
        ar = _evaluate_single(result, a)
        assert ar.passed is True

    def test_invalid_json(self) -> None:
        result = _make_result("not json at all")
        a = Assertion(type=AssertionType.JSON_VALID)
        ar = _evaluate_single(result, a)
        assert ar.passed is False
        assert "Invalid JSON" in ar.message


class TestJsonSchema:
    def test_required_keys_present(self) -> None:
        result = _make_result(json.dumps({"name": "Alice", "age": 30}))
        a = Assertion(
            type=AssertionType.JSON_SCHEMA,
            value={"required": ["name", "age"]},
        )
        ar = _evaluate_single(result, a)
        assert ar.passed is True

    def test_required_keys_missing(self) -> None:
        result = _make_result(json.dumps({"name": "Alice"}))
        a = Assertion(
            type=AssertionType.JSON_SCHEMA,
            value={"required": ["name", "age"]},
        )
        ar = _evaluate_single(result, a)
        assert ar.passed is False
        assert "age" in ar.message


class TestCostAssertion:
    def test_under_threshold(self) -> None:
        result = _make_result("ok", cost_usd=0.001)
        a = Assertion(type=AssertionType.COST_LESS_THAN, value=0.01)
        ar = _evaluate_single(result, a)
        assert ar.passed is True

    def test_over_threshold(self) -> None:
        result = _make_result("ok", cost_usd=0.05)
        a = Assertion(type=AssertionType.COST_LESS_THAN, value=0.01)
        ar = _evaluate_single(result, a)
        assert ar.passed is False


class TestLatencyAssertion:
    def test_under_threshold(self) -> None:
        result = _make_result("ok", latency_ms=100.0)
        a = Assertion(type=AssertionType.LATENCY_LESS_THAN, value=500)
        ar = _evaluate_single(result, a)
        assert ar.passed is True

    def test_over_threshold(self) -> None:
        result = _make_result("ok", latency_ms=1000.0)
        a = Assertion(type=AssertionType.LATENCY_LESS_THAN, value=500)
        ar = _evaluate_single(result, a)
        assert ar.passed is False


class TestEvaluateAssertions:
    def test_multiple_assertions(self) -> None:
        result = _make_result("Hello world", cost_usd=0.001, latency_ms=200)
        assertions = [
            Assertion(type=AssertionType.CONTAINS, value="hello"),
            Assertion(type=AssertionType.NOT_CONTAINS, value="error"),
            Assertion(type=AssertionType.COST_LESS_THAN, value=0.01),
        ]
        results = evaluate_assertions(result, assertions)
        assert len(results) == 3
        assert all(r.passed for r in results)

    def test_empty_assertions(self) -> None:
        result = _make_result("anything")
        results = evaluate_assertions(result, [])
        assert results == []


class TestStripMarkdownFences:
    def test_plain_text_unchanged(self) -> None:
        assert _strip_markdown_fences('{"key": "value"}') == '{"key": "value"}'

    def test_strips_json_fence(self) -> None:
        text = '```json\n{"key": "value"}\n```'
        assert _strip_markdown_fences(text) == '{"key": "value"}'

    def test_strips_plain_fence(self) -> None:
        text = '```\n{"key": "value"}\n```'
        assert _strip_markdown_fences(text) == '{"key": "value"}'

    def test_strips_with_surrounding_whitespace(self) -> None:
        text = '  ```json\n{"key": "value"}\n```  '
        assert _strip_markdown_fences(text) == '{"key": "value"}'

    def test_json_valid_with_fences(self) -> None:
        result = _make_result('```json\n{"sentiment": "bullish"}\n```')
        a = Assertion(type=AssertionType.JSON_VALID)
        ar = _evaluate_single(result, a)
        assert ar.passed is True

    def test_json_schema_with_fences(self) -> None:
        result = _make_result('```json\n{"name": "Alice", "age": 30}\n```')
        a = Assertion(
            type=AssertionType.JSON_SCHEMA,
            value={"required": ["name", "age"]},
        )
        ar = _evaluate_single(result, a)
        assert ar.passed is True
