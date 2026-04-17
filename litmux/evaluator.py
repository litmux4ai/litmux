"""Assertion evaluators for Litmux."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from litmux.models import (
    Assertion,
    AssertionResult,
    AssertionType,
    ModelRunResult,
)


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ```) from model output."""
    stripped = text.strip()
    match = re.match(r"^```(?:\w+)?\s*\n(.*?)\n?```\s*$", stripped, re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


def evaluate_assertions(
    result: ModelRunResult, assertions: list[Assertion]
) -> list[AssertionResult]:
    """Evaluate all assertions against a model run result."""
    return [_evaluate_single(result, a) for a in assertions]


def _evaluate_single(result: ModelRunResult, assertion: Assertion) -> AssertionResult:
    """Evaluate a single assertion."""
    handlers = {
        AssertionType.CONTAINS: _eval_contains,
        AssertionType.NOT_CONTAINS: _eval_not_contains,
        AssertionType.REGEX: _eval_regex,
        AssertionType.JSON_VALID: _eval_json_valid,
        AssertionType.JSON_SCHEMA: _eval_json_schema,
        AssertionType.COST_LESS_THAN: _eval_cost,
        AssertionType.LATENCY_LESS_THAN: _eval_latency,
        AssertionType.LLM_JUDGE: _eval_llm_judge,
    }

    handler = handlers.get(assertion.type)
    if not handler:
        return AssertionResult(
            assertion=assertion,
            passed=False,
            message=f"Unknown assertion type: {assertion.type}",
        )

    return handler(result, assertion)


def _eval_contains(result: ModelRunResult, assertion: Assertion) -> AssertionResult:
    value = str(assertion.value).lower()
    output = result.output.lower()
    passed = value in output
    return AssertionResult(
        assertion=assertion,
        passed=passed,
        message="" if passed else f'Output does not contain "{assertion.value}"',
        actual_value=result.output[:200],
    )


def _eval_not_contains(result: ModelRunResult, assertion: Assertion) -> AssertionResult:
    value = str(assertion.value).lower()
    output = result.output.lower()
    passed = value not in output
    return AssertionResult(
        assertion=assertion,
        passed=passed,
        message="" if passed else f'Output unexpectedly contains "{assertion.value}"',
        actual_value=result.output[:200],
    )


def _eval_regex(result: ModelRunResult, assertion: Assertion) -> AssertionResult:
    pattern = str(assertion.value)
    passed = bool(re.search(pattern, result.output))
    return AssertionResult(
        assertion=assertion,
        passed=passed,
        message="" if passed else f"Output does not match pattern: {pattern}",
        actual_value=result.output[:200],
    )


def _eval_json_valid(result: ModelRunResult, assertion: Assertion) -> AssertionResult:
    try:
        json.loads(_strip_markdown_fences(result.output))
        return AssertionResult(assertion=assertion, passed=True)
    except json.JSONDecodeError as e:
        return AssertionResult(
            assertion=assertion,
            passed=False,
            message=f"Invalid JSON: {e}",
            actual_value=result.output[:200],
        )


def _eval_json_schema(result: ModelRunResult, assertion: Assertion) -> AssertionResult:
    try:
        data = json.loads(_strip_markdown_fences(result.output))
        required = assertion.value.get("required", []) if isinstance(assertion.value, dict) else []
        missing = [k for k in required if k not in data]
        if missing:
            return AssertionResult(
                assertion=assertion,
                passed=False,
                message=f"Missing required keys: {missing}",
                actual_value=list(data.keys()) if isinstance(data, dict) else str(data)[:200],
            )
        return AssertionResult(assertion=assertion, passed=True)
    except json.JSONDecodeError as e:
        return AssertionResult(
            assertion=assertion, passed=False, message=f"Invalid JSON: {e}"
        )


def _eval_cost(result: ModelRunResult, assertion: Assertion) -> AssertionResult:
    threshold = float(assertion.value)
    passed = result.cost_usd < threshold
    return AssertionResult(
        assertion=assertion,
        passed=passed,
        message="" if passed else f"Cost ${result.cost_usd:.6f} exceeds ${threshold}",
        actual_value=result.cost_usd,
    )


def _eval_latency(result: ModelRunResult, assertion: Assertion) -> AssertionResult:
    threshold = float(assertion.value)
    passed = result.latency_ms < threshold
    return AssertionResult(
        assertion=assertion,
        passed=passed,
        message="" if passed else f"Latency {result.latency_ms:.0f}ms exceeds {threshold}ms",
        actual_value=result.latency_ms,
    )


def _eval_llm_judge(result: ModelRunResult, assertion: Assertion) -> AssertionResult:
    """Use an LLM to score quality 1-10. Defaults to gpt-4o-mini; override with LITMUX_JUDGE_MODEL."""
    criteria = assertion.criteria or "Is the output high quality and accurate?"
    threshold = assertion.threshold
    judge_model = os.environ.get("LITMUX_JUDGE_MODEL", "gpt-4o-mini")

    if not os.environ.get("OPENAI_API_KEY"):
        return AssertionResult(
            assertion=assertion,
            passed=False,
            message="llm-judge requires OPENAI_API_KEY (or override the judge model with LITMUX_JUDGE_MODEL)",
        )

    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=judge_model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Score the following AI output from 1 to 10.\n\n"
                        f"Criteria: {criteria}\n\n"
                        f"Output to evaluate:\n{result.output[:2000]}\n\n"
                        f"Return ONLY a single number between 1 and 10."
                    ),
                }
            ],
            temperature=0,
            max_tokens=5,
        )

        score_text = response.choices[0].message.content or "0"
        score = float(score_text.strip())
        passed = score >= threshold

        return AssertionResult(
            assertion=assertion,
            passed=passed,
            message=f"Score: {score}/10 (threshold: {threshold})",
            actual_value=score,
        )
    except Exception as e:
        return AssertionResult(
            assertion=assertion,
            passed=False,
            message=f"LLM judge failed: {e}",
        )
