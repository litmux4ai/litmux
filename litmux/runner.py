"""Async model runners for OpenAI, Anthropic, Google, and HuggingFace."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from litmux.cost import calculate_cost
from litmux.models import ModelConfig, ModelRunResult, Provider


async def run_model(config: ModelConfig, prompt: str) -> ModelRunResult:
    """Run a prompt against a single model and return the result."""
    from litmux.cache import get_cached, set_cached

    # Check cache first
    cached = get_cached(config.model, prompt, config.temperature, config.max_tokens)
    if cached is not None:
        cost = calculate_cost(
            config.model, cached["input_tokens"], cached["output_tokens"]
        )
        return ModelRunResult(
            model_config_obj=config,
            output=cached["output"],
            latency_ms=0.0,
            input_tokens=cached["input_tokens"],
            output_tokens=cached["output_tokens"],
            cost_usd=cost,
        )

    start = time.perf_counter()
    try:
        if config.provider == Provider.OPENAI:
            result = await _run_openai(config, prompt)
        elif config.provider == Provider.ANTHROPIC:
            result = await _run_anthropic(config, prompt)
        elif config.provider == Provider.GOOGLE:
            result = await _run_google(config, prompt)
        elif config.provider == Provider.HUGGINGFACE:
            result = await _run_huggingface(config, prompt)
        else:
            raise ValueError(f"Unknown provider: {config.provider}")

        elapsed_ms = (time.perf_counter() - start) * 1000
        cost = calculate_cost(
            config.model, result["input_tokens"], result["output_tokens"]
        )

        # Cache successful result
        set_cached(config.model, prompt, config.temperature, config.max_tokens, result)

        return ModelRunResult(
            model_config_obj=config,
            output=result["output"],
            latency_ms=elapsed_ms,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            cost_usd=cost,
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return ModelRunResult(
            model_config_obj=config,
            latency_ms=elapsed_ms,
            error=str(e),
        )


async def run_models_parallel(
    models: list[ModelConfig], prompt: str
) -> list[ModelRunResult]:
    """Run a prompt against multiple models in parallel."""
    tasks = [run_model(m, prompt) for m in models]
    return await asyncio.gather(*tasks)


async def _run_openai(config: ModelConfig, prompt: str) -> dict[str, Any]:
    """Run a prompt against OpenAI."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model=config.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    choice = response.choices[0]
    usage = response.usage

    return {
        "output": choice.message.content or "",
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
    }


async def _run_anthropic(config: ModelConfig, prompt: str) -> dict[str, Any]:
    """Run a prompt against Anthropic."""
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic()
    response = await client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.temperature,
    )

    output = ""
    for block in response.content:
        if hasattr(block, "text"):
            output += block.text

    return {
        "output": output,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


async def _run_google(config: ModelConfig, prompt: str) -> dict[str, Any]:
    """Run a prompt against Google Gemini."""
    from google import genai

    client = genai.Client()
    response = await asyncio.to_thread(
        client.models.generate_content,
        model=config.model,
        contents=prompt,
    )

    output = response.text or ""
    usage_meta = response.usage_metadata
    input_tokens = usage_meta.prompt_token_count if usage_meta else 0
    output_tokens = usage_meta.candidates_token_count if usage_meta else 0

    return {
        "output": output,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


async def _run_huggingface(config: ModelConfig, prompt: str) -> dict[str, Any]:
    """Run a prompt against HuggingFace Inference API."""
    from huggingface_hub import AsyncInferenceClient

    client = AsyncInferenceClient(token=os.environ.get("HF_TOKEN"))
    response = await client.chat_completion(
        model=config.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.temperature or 0.01,
        max_tokens=config.max_tokens,
    )

    choice = response.choices[0]
    usage = response.usage

    return {
        "output": choice.message.content or "",
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
    }
