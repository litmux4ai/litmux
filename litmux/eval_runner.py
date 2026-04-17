"""Eval engine — runs prompt × dataset × models."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from litmux.dataset import apply_input_mapping, load_dataset
from litmux.evaluator import evaluate_assertions
from litmux.models import (
    EvalCase,
    EvalResult,
    EvalRowResult,
    LitmuxConfig,
    ModelConfig,
    ModelRunResult,
)
from litmux.runner import run_model


async def run_eval(
    config: LitmuxConfig,
    eval_case: EvalCase,
    models: list[ModelConfig] | None = None,
    limit: int | None = None,
) -> list[EvalResult]:
    """Run an eval case against all models. Returns one EvalResult per model."""
    selected_models = models or config.models
    rows = load_dataset(eval_case.dataset)
    if limit:
        rows = rows[:limit]

    results = []
    for model in selected_models:
        row_results = await _eval_model(model, eval_case, rows)

        # Aggregate stats
        total = len(row_results)
        passed = sum(1 for r in row_results if r.passed)
        pass_rate = (passed / total * 100) if total > 0 else 0

        scores = [r.judge_score for r in row_results if r.judge_score is not None]
        avg_score = sum(scores) / len(scores) if scores else None

        avg_latency = sum(r.latency_ms for r in row_results) / total if total > 0 else 0
        total_cost = sum(r.cost_usd for r in row_results)

        results.append(
            EvalResult(
                eval_case=eval_case,
                model_name=model.model,
                row_results=row_results,
                pass_rate=pass_rate,
                avg_score=avg_score,
                avg_latency_ms=avg_latency,
                total_cost_usd=total_cost,
            )
        )

    return results


async def _eval_model(
    model: ModelConfig,
    eval_case: EvalCase,
    rows: list[dict[str, Any]],
    batch_size: int = 10,
) -> list[EvalRowResult]:
    """Evaluate a single model against all dataset rows."""
    row_results: list[EvalRowResult] = []

    # Process in batches to avoid rate limits
    for batch_start in range(0, len(rows), batch_size):
        batch = rows[batch_start : batch_start + batch_size]
        tasks = []

        for i, row in enumerate(batch):
            row_index = batch_start + i
            prompt = apply_input_mapping(
                eval_case.prompt, eval_case.input_mapping, row
            )
            tasks.append(_eval_single_row(model, eval_case, row, row_index, prompt))

        batch_results = await asyncio.gather(*tasks)
        row_results.extend(batch_results)

    return row_results


async def _eval_single_row(
    model: ModelConfig,
    eval_case: EvalCase,
    row: dict[str, Any],
    row_index: int,
    prompt: str,
) -> EvalRowResult:
    """Evaluate a single dataset row against a model."""
    mr = await run_model(model, prompt)

    # Evaluate assertions
    assertion_results = []
    if not mr.error:
        assertion_results = evaluate_assertions(mr, eval_case.assertions)

    # Get expected value
    expected = None
    if eval_case.expected and eval_case.expected in row:
        expected = str(row[eval_case.expected])

    # LLM judge scoring
    judge_score = None
    if eval_case.judge and not mr.error:
        judge_score = await _run_judge(
            mr.output, expected, eval_case.judge.criteria,
            judge_model=eval_case.judge.model,
        )

    # Determine pass/fail
    assertions_pass = all(r.passed for r in assertion_results)
    judge_pass = (
        judge_score >= eval_case.judge.threshold
        if eval_case.judge and judge_score is not None
        else True
    )
    passed = assertions_pass and judge_pass and mr.error is None

    return EvalRowResult(
        row_index=row_index,
        inputs={col: str(row.get(col, "")) for col in eval_case.input_mapping.values()},
        expected=expected,
        actual_output=mr.output,
        passed=passed,
        judge_score=judge_score,
        latency_ms=mr.latency_ms,
        cost_usd=mr.cost_usd,
        assertion_results=assertion_results,
    )


async def _run_judge(
    output: str,
    expected: str | None,
    criteria: str,
    judge_model: str | None = None,
) -> float | None:
    """Run LLM judge to score output 1-10."""
    model = judge_model or os.environ.get("LITMUX_JUDGE_MODEL", "gpt-4o-mini")
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        judge_prompt = (
            f"Score this AI output from 1 to 10.\n\n"
            f"Criteria: {criteria}\n\n"
        )
        if expected:
            judge_prompt += f"Expected output: {expected}\n\n"
        judge_prompt += f"Actual output:\n{output[:2000]}\n\nReturn ONLY a number 1-10."

        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0,
            max_tokens=5,
        )

        score_text = response.choices[0].message.content or "0"
        return float(score_text.strip())
    except Exception:
        return None
