"""Pricing tables and cost calculation for Litmux."""

from __future__ import annotations

# Prices per 1M tokens (USD): (input_price, output_price)
PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o3-mini": (1.10, 4.40),
    # Anthropic
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-opus-4-6": (5.00, 25.00),
    # Legacy (kept for fuzzy matching)
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-opus-20240229": (15.00, 75.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    # Google
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    # HuggingFace (free-tier / serverless inference)
    "meta-llama/Llama-3.1-8B-Instruct": (0.0, 0.0),
    "meta-llama/Llama-3.1-70B-Instruct": (0.0, 0.0),
    "mistralai/Mistral-7B-Instruct-v0.3": (0.0, 0.0),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": (0.0, 0.0),
    "Qwen/Qwen2.5-72B-Instruct": (0.0, 0.0),
    "deepseek-ai/DeepSeek-R1": (0.0, 0.0),
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate the cost for a single API call in USD."""
    pricing = _get_pricing(model)
    input_cost = (input_tokens / 1_000_000) * pricing[0]
    output_cost = (output_tokens / 1_000_000) * pricing[1]
    return input_cost + output_cost


def _get_pricing(model: str) -> tuple[float, float]:
    """Get pricing for a model, with fuzzy matching."""
    if model in PRICING:
        return PRICING[model]

    model_lower = model.lower()
    for key, price in PRICING.items():
        if key in model_lower or model_lower in key:
            return price

    return (0.0, 0.0)


def project_cost(
    model: str,
    avg_input_tokens: int,
    avg_output_tokens: int,
    daily_volume: int,
) -> dict[str, float]:
    """Project costs for a given daily volume."""
    per_call = calculate_cost(model, avg_input_tokens, avg_output_tokens)
    daily = per_call * daily_volume
    monthly = daily * 30
    yearly = daily * 365
    return {
        "per_call": per_call,
        "daily": daily,
        "monthly": monthly,
        "yearly": yearly,
    }
