"""Tests for Pydantic data models."""

from __future__ import annotations

import pytest

from litmux.models import (
    Assertion,
    AssertionResult,
    AssertionType,
    DefaultTest,
    EvalCase,
    EvalResult,
    EvalRowResult,
    JudgeConfig,
    LitmuxConfig,
    ModelConfig,
    ModelRunResult,
    Provider,
    TestCase,
    TestResult,
)


class TestProvider:
    def test_enum_values(self) -> None:
        assert Provider.OPENAI == "openai"
        assert Provider.ANTHROPIC == "anthropic"
        assert Provider.GOOGLE == "google"
        assert Provider.HUGGINGFACE == "huggingface"

    def test_from_string(self) -> None:
        assert Provider("openai") == Provider.OPENAI
        assert Provider("anthropic") == Provider.ANTHROPIC
        assert Provider("huggingface") == Provider.HUGGINGFACE


class TestModelConfig:
    def test_defaults(self) -> None:
        mc = ModelConfig(provider=Provider.OPENAI, model="gpt-4o")
        assert mc.temperature == 0.0
        assert mc.max_tokens == 1024

    def test_custom_values(self) -> None:
        mc = ModelConfig(
            provider=Provider.ANTHROPIC,
            model="claude-3-5-sonnet-20241022",
            temperature=0.7,
            max_tokens=2048,
        )
        assert mc.temperature == 0.7
        assert mc.max_tokens == 2048


class TestAssertion:
    def test_contains(self) -> None:
        a = Assertion(type=AssertionType.CONTAINS, value="hello")
        assert a.type == AssertionType.CONTAINS
        assert a.threshold == 7.0  # default

    def test_llm_judge(self) -> None:
        a = Assertion(
            type=AssertionType.LLM_JUDGE,
            criteria="Is it polite?",
            threshold=8.0,
        )
        assert a.criteria == "Is it polite?"
        assert a.threshold == 8.0


class TestTestCase:
    def test_alias_field(self) -> None:
        tc = TestCase(
            name="test1",
            prompt="Say hello",
            **{"assert": [{"type": "contains", "value": "hi"}]},
        )
        assert len(tc.assertions) == 1
        assert tc.assertions[0].type == AssertionType.CONTAINS

    def test_populate_by_name(self) -> None:
        tc = TestCase(
            name="test1",
            prompt="Say hello",
            assertions=[Assertion(type=AssertionType.CONTAINS, value="hi")],
        )
        assert len(tc.assertions) == 1

    def test_empty_defaults(self) -> None:
        tc = TestCase(name="t", prompt="p")
        assert tc.inputs == {}
        assert tc.assertions == []
        assert tc.prompt_source == ""


class TestModelRunResult:
    def test_passed_all_assertions(self) -> None:
        mc = ModelConfig(provider=Provider.OPENAI, model="gpt-4o")
        result = ModelRunResult(
            model_config_obj=mc,
            output="hello world",
            assertion_results=[
                AssertionResult(
                    assertion=Assertion(type=AssertionType.CONTAINS, value="hello"),
                    passed=True,
                )
            ],
        )
        assert result.passed is True
        assert result.pass_count == 1
        assert result.total_assertions == 1

    def test_failed_assertion(self) -> None:
        mc = ModelConfig(provider=Provider.OPENAI, model="gpt-4o")
        result = ModelRunResult(
            model_config_obj=mc,
            output="goodbye",
            assertion_results=[
                AssertionResult(
                    assertion=Assertion(type=AssertionType.CONTAINS, value="hello"),
                    passed=False,
                    message='Output does not contain "hello"',
                )
            ],
        )
        assert result.passed is False

    def test_error_means_not_passed(self) -> None:
        mc = ModelConfig(provider=Provider.OPENAI, model="gpt-4o")
        result = ModelRunResult(
            model_config_obj=mc,
            error="API timeout",
        )
        assert result.passed is False

    def test_model_name_property(self) -> None:
        mc = ModelConfig(provider=Provider.OPENAI, model="gpt-4o-mini")
        result = ModelRunResult(model_config_obj=mc, output="test")
        assert result.model_name == "gpt-4o-mini"


class TestTestResult:
    def test_all_passed(self) -> None:
        tc = TestCase(name="t", prompt="p")
        mc = ModelConfig(provider=Provider.OPENAI, model="gpt-4o")
        tr = TestResult(
            test_case=tc,
            model_results=[
                ModelRunResult(model_config_obj=mc, output="ok"),
            ],
        )
        assert tr.all_passed is True


class TestLitmuxConfig:
    def test_minimal_config(self) -> None:
        config = LitmuxConfig(
            models=[ModelConfig(provider=Provider.OPENAI, model="gpt-4o")]
        )
        assert len(config.models) == 1
        assert config.tests == []
        assert config.evals == []
        assert config.default_test is None


class TestDefaultTest:
    def test_defaults(self) -> None:
        dt = DefaultTest()
        assert dt.assertions == []
        assert dt.inputs == {}

    def test_with_assertions(self) -> None:
        dt = DefaultTest(**{"assert": [{"type": "json-valid"}]})
        assert len(dt.assertions) == 1
        assert dt.assertions[0].type == AssertionType.JSON_VALID


class TestEvalModels:
    def test_eval_case_defaults(self) -> None:
        ec = EvalCase(name="e", prompt="p")
        assert ec.dataset == ""
        assert ec.input_mapping == {}
        assert ec.assertions == []
        assert ec.judge is None

    def test_judge_config(self) -> None:
        jc = JudgeConfig(criteria="accuracy", threshold=8.0)
        assert jc.sample is None

    def test_eval_row_result(self) -> None:
        err = EvalRowResult(row_index=0, passed=True, judge_score=9.5)
        assert err.passed is True
        assert err.cost_usd == 0.0

    def test_eval_result(self) -> None:
        ec = EvalCase(name="e", prompt="p")
        er = EvalResult(
            eval_case=ec,
            model_name="gpt-4o",
            pass_rate=0.85,
            total_cost_usd=0.05,
        )
        assert er.avg_score is None
        assert er.row_results == []
