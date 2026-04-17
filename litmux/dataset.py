"""Dataset loader and AI generation for Litmux."""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any


def load_dataset(path: str) -> list[dict[str, Any]]:
    """Load dataset from local file (CSV/JSON)."""
    if os.path.exists(path):
        return _load_local(path)

    raise FileNotFoundError(
        f"Dataset not found: {path}"
    )


def _load_local(path: str) -> list[dict[str, Any]]:
    """Load from a local CSV or JSON file."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return _load_csv(path)
    elif p.suffix.lower() == ".json":
        return _load_json(path)
    else:
        raise ValueError(f"Unsupported dataset format: {p.suffix}. Use .csv or .json.")


def _load_csv(path: str) -> list[dict[str, Any]]:
    """Load CSV into list of dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_json(path: str) -> list[dict[str, Any]]:
    """Load JSON file (expects array of objects)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError("JSON dataset must be an array of objects.")


def apply_input_mapping(
    prompt: str, mapping: dict[str, str], row: dict[str, Any]
) -> str:
    """Replace {{variables}} in prompt using column mapping + row data."""
    for prompt_var, column_name in mapping.items():
        value = str(row.get(column_name, ""))
        prompt = prompt.replace(f"{{{{{prompt_var}}}}}", value)
    return prompt


def extract_variables(template: str) -> list[str]:
    """Find all {{variable}} patterns in a template."""
    return re.findall(r"\{\{(\w+)\}\}", template)


def _infer_generation_provider(model: str) -> str:
    """Detect provider for dataset generation (OpenAI or Anthropic only)."""
    from litmux.config import _infer_provider

    try:
        provider = _infer_provider(model)
    except ValueError:
        provider = None

    if provider in ("openai", "anthropic"):
        return provider

    raise ValueError(
        f"Model '{model}' is not supported for dataset generation. "
        f"Use an OpenAI model (e.g. gpt-4o-mini) or Anthropic model (e.g. claude-3-5-haiku-20241022)."
    )


def _build_meta_prompt(
    template: str, variables: list[str], description: str, n: int,
    seed_rows: list[dict[str, Any]] | None = None,
) -> str:
    """Build the meta-prompt for dataset generation."""
    prompt = (
        "You are a QA engineer creating test scenarios for an AI system.\n\n"
        f"The system uses this prompt template:\n---\n{template}\n---\n\n"
        f"Input variables: {variables}\n"
        f"Task description: {description}\n\n"
    )

    if seed_rows:
        prompt += "Here are example rows showing the EXACT format to follow:\n"
        for i, row in enumerate(seed_rows[:5]):
            prompt += f"  Example {i + 1}: {json.dumps(row, ensure_ascii=False)}\n"
        prompt += (
            "\nGenerate new scenarios using the SAME columns and format as the examples above.\n"
            "Do NOT add extra columns. Do NOT change column names.\n"
            "Add a 'scenario_type' column if not already present.\n\n"
        )
    else:
        prompt += (
            "Each scenario should have:\n"
            f"- Input values for each variable: {', '.join(variables)}\n"
            "- expected_output: what a correct response should contain or be\n"
        )

    prompt += (
        f"Generate exactly {n} diverse test scenarios as a JSON array.\n"
        'Each scenario must include "difficulty": "easy" | "medium" | "hard"\n'
        'and "scenario_type": "happy_path" | "edge_case" | "adversarial" | '
        '"multilingual" | "empty_input" | "long_input" | "ambiguous"\n\n'
        "Distribution:\n"
        "- 40% happy_path\n"
        "- 20% edge_case\n"
        "- 15% adversarial\n"
        "- 10% multilingual\n"
        "- 10% empty_input\n"
        "- 5% long_input\n\n"
        "Return ONLY a valid JSON array, no explanation."
    )

    return prompt


def _parse_json_response(content: str) -> list[dict[str, Any]]:
    """Parse JSON array from model response with multi-step fallback."""

    def _extract_list(data: Any) -> list[dict[str, Any]]:
        """Extract list from raw parsed data (handles wrapped or bare arrays)."""
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "scenarios" in data:
            return data["scenarios"]
        raise ValueError(f"Expected a JSON array or {{scenarios: [...]}}, got {type(data).__name__}")

    # Step 1: Try direct parse
    try:
        return _extract_list(json.loads(content.strip()))
    except (json.JSONDecodeError, ValueError):
        pass

    # Step 2: Strip markdown fences (```json ... ```)
    stripped = content.strip()
    fence_match = re.match(r"^```(?:\w+)?\s*\n(.*?)\n?```\s*$", stripped, re.DOTALL)
    if fence_match:
        try:
            return _extract_list(json.loads(fence_match.group(1).strip()))
        except (json.JSONDecodeError, ValueError):
            pass

    # Step 3: Regex extract [...] as last resort
    json_match = re.search(r"\[.*\]", content, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Failed to parse JSON array from model response. "
        f"First 200 chars: {content[:200]}"
    )


def _build_scenario_schema(
    variables: list[str], n: int, provider: str,
    seed_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Build a JSON schema for structured output with exact item count."""
    scenario_properties: dict[str, Any] = {}

    if seed_columns:
        # Use seed columns — generate data matching the seed format
        for col in seed_columns:
            scenario_properties[col] = {"type": "string"}
        # Ensure difficulty and scenario_type are present
        if "difficulty" not in scenario_properties:
            scenario_properties["difficulty"] = {
                "type": "string",
                "enum": ["easy", "medium", "hard"],
            }
        if "scenario_type" not in scenario_properties:
            scenario_properties["scenario_type"] = {
                "type": "string",
                "enum": [
                    "happy_path", "edge_case", "adversarial",
                    "multilingual", "empty_input", "long_input", "ambiguous",
                ],
            }
        required = list(scenario_properties.keys())
    else:
        # Default: use prompt variables + expected_output
        for var in variables:
            scenario_properties[var] = {"type": "string"}
        scenario_properties["expected_output"] = {"type": "string"}
        scenario_properties["difficulty"] = {
            "type": "string",
            "enum": ["easy", "medium", "hard"],
        }
        scenario_properties["scenario_type"] = {
            "type": "string",
            "enum": [
                "happy_path", "edge_case", "adversarial",
                "multilingual", "empty_input", "long_input", "ambiguous",
            ],
        }
        required = [*variables, "expected_output", "difficulty", "scenario_type"]

    array_constraints: dict[str, Any] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": scenario_properties,
            "required": required,
            "additionalProperties": False,
        },
    }

    # OpenAI supports minItems/maxItems for exact count; Anthropic only supports 0 or 1
    if provider == "openai":
        array_constraints["minItems"] = n
        array_constraints["maxItems"] = n

    return {
        "type": "object",
        "properties": {
            "scenarios": array_constraints,
        },
        "required": ["scenarios"],
        "additionalProperties": False,
    }


async def _generate_with_openai(
    model: str, prompt: str, max_tokens: int, schema: dict[str, Any]
) -> str:
    """Call OpenAI with structured output and return raw text response."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=max_tokens,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "test_scenarios",
                "strict": True,
                "schema": schema,
            },
        },
    )
    return response.choices[0].message.content or '{"scenarios": []}'


async def _generate_with_anthropic(
    model: str, prompt: str, max_tokens: int, schema: dict[str, Any]
) -> str:
    """Call Anthropic and ask for a JSON response matching the schema."""
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic()
    system = (
        "You are a test-data generator. Respond with a single JSON object that "
        "conforms to this JSON Schema. Output only the JSON, no prose, no markdown.\n\n"
        f"Schema: {json.dumps(schema)}"
    )
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )

    output = ""
    for block in response.content:
        if hasattr(block, "text"):
            output += block.text
    return output or '{"scenarios": []}'


async def generate_dataset(
    prompt_template: str,
    description: str,
    n: int = 30,
    model: str = "claude-haiku-4-5-20251001",
    seed_path: str | None = None,
) -> list[dict[str, Any]]:
    """Generate test scenarios using AI."""
    variables = extract_variables(prompt_template)

    # Load seed data if provided
    seed_rows: list[dict[str, Any]] | None = None
    seed_columns: list[str] | None = None
    if seed_path:
        seed_rows = _load_local(seed_path)
        if seed_rows:
            seed_columns = list(seed_rows[0].keys())

    meta_prompt = _build_meta_prompt(
        prompt_template, variables, description, n, seed_rows=seed_rows
    )
    max_tokens = min(8192, max(4096, n * 200))
    provider = _infer_generation_provider(model)
    schema = _build_scenario_schema(variables, n, provider, seed_columns=seed_columns)

    if provider == "anthropic":
        content = await _generate_with_anthropic(model, meta_prompt, max_tokens, schema)
    else:
        content = await _generate_with_openai(model, meta_prompt, max_tokens, schema)

    rows = _parse_json_response(content)

    # Trim to exact count (structured output guarantees format, not always exact count)
    if len(rows) > n:
        rows = rows[:n]

    return rows


def save_dataset_csv(rows: list[dict[str, Any]], path: str) -> None:
    """Save dataset rows to a CSV file."""
    if not rows:
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
