"""Pydantic data models for Litmux."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ─── Enums ───────────────────────────────────────────────────────


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"


class AssertionType(str, Enum):
    CONTAINS = "contains"
    NOT_CONTAINS = "not-contains"
    REGEX = "regex"
    JSON_VALID = "json-valid"
    JSON_SCHEMA = "json-schema"
    COST_LESS_THAN = "cost-less-than"
    LATENCY_LESS_THAN = "latency-less-than"
    LLM_JUDGE = "llm-judge"


# ─── Config Models ──────────────────────────────────────────────


class ModelConfig(BaseModel):
    provider: Provider
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024


class Assertion(BaseModel):
    type: AssertionType
    value: Any = None
    criteria: str | None = None
    threshold: float = 7.0


class JudgeConfig(BaseModel):
    criteria: str
    threshold: float = 7.0
    sample: int | None = None
    model: str | None = None


class TestCase(BaseModel):
    name: str
    prompt: str
    prompt_source: str = ""
    inputs: dict[str, str] = Field(default_factory=dict)
    assertions: list[Assertion] = Field(default_factory=list, alias="assert")

    model_config = {"populate_by_name": True}


class EvalCase(BaseModel):
    name: str
    prompt: str
    prompt_source: str = ""
    dataset: str = ""
    input_mapping: dict[str, str] = Field(default_factory=dict)
    expected: str | None = None
    assertions: list[Assertion] = Field(default_factory=list, alias="assert")
    judge: JudgeConfig | None = None

    model_config = {"populate_by_name": True}


class DefaultTest(BaseModel):
    assertions: list[Assertion] = Field(default_factory=list, alias="assert")
    inputs: dict[str, str] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class LitmuxConfig(BaseModel):
    models: list[ModelConfig]
    tests: list[TestCase] = Field(default_factory=list)
    evals: list[EvalCase] = Field(default_factory=list)
    default_test: DefaultTest | None = None


# ─── Result Models ──────────────────────────────────────────────


class AssertionResult(BaseModel):
    assertion: Assertion
    passed: bool
    message: str = ""
    actual_value: Any = None


class ModelRunResult(BaseModel):
    model_config_obj: ModelConfig
    output: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    assertion_results: list[AssertionResult] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.assertion_results) and self.error is None

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.assertion_results if r.passed)

    @property
    def total_assertions(self) -> int:
        return len(self.assertion_results)

    @property
    def model_name(self) -> str:
        return self.model_config_obj.model


class TestResult(BaseModel):
    test_case: TestCase
    model_results: list[ModelRunResult] = Field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.model_results)


class EvalRowResult(BaseModel):
    row_index: int
    inputs: dict[str, Any] = Field(default_factory=dict)
    expected: str | None = None
    actual_output: str = ""
    passed: bool = False
    judge_score: float | None = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    assertion_results: list[AssertionResult] = Field(default_factory=list)


class EvalResult(BaseModel):
    eval_case: EvalCase
    model_name: str
    row_results: list[EvalRowResult] = Field(default_factory=list)
    pass_rate: float = 0.0
    avg_score: float | None = None
    avg_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
