"""Tests for cost calculation."""

from __future__ import annotations

import pytest

from litmux.cost import calculate_cost, project_cost, _get_pricing, PRICING


class TestGetPricing:
    def test_exact_match(self) -> None:
        assert _get_pricing("gpt-4o") == PRICING["gpt-4o"]

    def test_fuzzy_match(self) -> None:
        price = _get_pricing("gpt-4o-2024-08-06")
        assert price == PRICING["gpt-4o"]

    def test_unknown_model_returns_zero(self) -> None:
        assert _get_pricing("totally-unknown-model") == (0.0, 0.0)


class TestCalculateCost:
    def test_known_model(self) -> None:
        cost = calculate_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500)
        expected_input = (1000 / 1_000_000) * 0.15
        expected_output = (500 / 1_000_000) * 0.60
        assert cost == pytest.approx(expected_input + expected_output)

    def test_zero_tokens(self) -> None:
        cost = calculate_cost("gpt-4o", input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_unknown_model_zero_cost(self) -> None:
        cost = calculate_cost("mystery-model", input_tokens=1000, output_tokens=1000)
        assert cost == 0.0


class TestProjectCost:
    def test_projection(self) -> None:
        result = project_cost(
            model="gpt-4o-mini",
            avg_input_tokens=500,
            avg_output_tokens=200,
            daily_volume=100,
        )
        assert "per_call" in result
        assert "daily" in result
        assert "monthly" in result
        assert "yearly" in result
        assert result["daily"] == pytest.approx(result["per_call"] * 100)
        assert result["monthly"] == pytest.approx(result["daily"] * 30)
        assert result["yearly"] == pytest.approx(result["daily"] * 365)

    def test_zero_volume(self) -> None:
        result = project_cost("gpt-4o", 100, 100, daily_volume=0)
        assert result["daily"] == 0.0
        assert result["monthly"] == 0.0
