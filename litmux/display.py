"""Rich terminal display for Litmux results."""

from __future__ import annotations

import json

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from litmux import __version__
from litmux.cost import project_cost
from litmux.models import EvalResult, ModelRunResult, TestResult

console = Console()


def print_header() -> None:
    """Print the Litmux header."""
    console.print()
    console.print(
        Panel(
            f"[bold cyan]⚡ Litmux[/bold cyan] v{__version__} — CI/CD for AI",
            border_style="cyan",
        )
    )
    console.print()


def print_test_result(test_result: TestResult) -> None:
    """Print results for a single test case."""
    tc = test_result.test_case
    source = tc.prompt_source or "inline"

    console.print(f"  [bold]Test:[/bold] {tc.name}")
    console.print(f"  [dim]Prompt:[/dim] {source}")
    console.print()

    table = Table(show_header=True, header_style="bold", padding=(0, 1))
    table.add_column("Model", style="cyan", min_width=22)
    table.add_column("Result", justify="center", min_width=8)
    table.add_column("Latency", justify="right", min_width=8)
    table.add_column("Cost", justify="right", min_width=10)
    table.add_column("In Tok", justify="right", min_width=7)
    table.add_column("Out Tok", justify="right", min_width=7)
    table.add_column("Assertions", justify="center", min_width=10)

    for mr in test_result.model_results:
        _add_result_row(table, mr)

    console.print(table)
    console.print()

    for mr in test_result.model_results:
        if mr.error:
            console.print(f"  [red]✗ {mr.model_name}:[/red] Error — {mr.error}")
        else:
            failures = [r for r in mr.assertion_results if not r.passed]
            if failures:
                console.print(f"  [red]Failed assertions for {mr.model_name}:[/red]")
                for f in failures:
                    console.print(f"    [red]✗[/red] {f.assertion.type.value}: {f.message}")
                console.print()


def _add_result_row(table: Table, mr: ModelRunResult) -> None:
    """Add a row for a model result."""
    if mr.error:
        table.add_row(mr.model_name, "[red]ERROR[/red]", f"{mr.latency_ms:.0f}ms", "—", "—", "—", "—")
        return

    result_text = "[green]✅ PASS[/green]" if mr.passed else "[red]❌ FAIL[/red]"
    latency = f"{mr.latency_ms:.0f}ms" if mr.latency_ms < 10_000 else f"{mr.latency_ms / 1000:.1f}s"
    cost = f"${mr.cost_usd:.6f}" if mr.cost_usd < 0.01 else f"${mr.cost_usd:.4f}"
    in_tok = str(mr.input_tokens)
    out_tok = str(mr.output_tokens)
    assertions = f"{mr.pass_count}/{mr.total_assertions}"

    table.add_row(mr.model_name, result_text, latency, cost, in_tok, out_tok, assertions)


def print_summary(results: list[TestResult]) -> None:
    """Print a summary line."""
    total = len(results)
    passed = sum(1 for r in results if r.all_passed)
    failed = total - passed

    console.print("─" * 60)
    if failed == 0:
        console.print(f"  [bold green]✅ All {total} tests passed![/bold green]")
    else:
        console.print(f"  [bold red]❌ {failed}/{total} tests failed[/bold red]")
    console.print()


def print_savings_summary(
    results: list[TestResult], daily_volume: int = 10_000
) -> None:
    """Show cheapest passing model and potential savings."""
    # Collect per-model stats across all tests
    model_stats: dict[str, dict] = {}
    for tr in results:
        for mr in tr.model_results:
            name = mr.model_name
            if name not in model_stats:
                model_stats[name] = {
                    "passed_all": True,
                    "total_cost": 0.0,
                    "input_tokens": [],
                    "output_tokens": [],
                    "model": mr.model_config_obj.model,
                }
            if mr.error or not mr.passed:
                model_stats[name]["passed_all"] = False
            model_stats[name]["total_cost"] += mr.cost_usd
            model_stats[name]["input_tokens"].append(mr.input_tokens)
            model_stats[name]["output_tokens"].append(mr.output_tokens)

    # Only show if 2+ models and at least one passes
    passing = {k: v for k, v in model_stats.items() if v["passed_all"]}
    if len(passing) < 1 or len(model_stats) < 2:
        return

    # Project monthly costs for passing models
    for name, stats in passing.items():
        avg_in = sum(stats["input_tokens"]) / max(len(stats["input_tokens"]), 1)
        avg_out = sum(stats["output_tokens"]) / max(len(stats["output_tokens"]), 1)
        proj = project_cost(stats["model"], int(avg_in), int(avg_out), daily_volume)
        stats["monthly"] = proj["monthly"]

    cheapest = min(passing.items(), key=lambda x: x[1]["monthly"])

    # Calculate savings vs most expensive model (passing or not)
    expensive_monthly = 0.0
    most_expensive_name: str | None = None
    for name, stats in model_stats.items():
        avg_in = sum(stats["input_tokens"]) / max(len(stats["input_tokens"]), 1)
        avg_out = sum(stats["output_tokens"]) / max(len(stats["output_tokens"]), 1)
        proj = project_cost(stats["model"], int(avg_in), int(avg_out), daily_volume)
        if proj["monthly"] > expensive_monthly:
            expensive_monthly = proj["monthly"]
            most_expensive_name = name

    cheapest_monthly = cheapest[1]["monthly"]
    savings = expensive_monthly - cheapest_monthly

    if savings <= 0 or most_expensive_name is None:
        return

    console.print(f"  [bold]💡 Cost Insight[/bold] (at {daily_volume:,} calls/day)")
    if cheapest_monthly == 0:
        console.print(f"  Cheapest passing model: [bold cyan]{cheapest[0]}[/bold cyan] ([green]FREE[/green])")
    else:
        console.print(f"  Cheapest passing model: [bold cyan]{cheapest[0]}[/bold cyan] (${cheapest_monthly:,.0f}/mo)")
    console.print(f"  vs most expensive: {most_expensive_name} (${expensive_monthly:,.0f}/mo)")
    console.print(f"  [green]→ Save ${savings:,.0f}/month (${savings * 12:,.0f}/year)[/green]")
    console.print()


def print_recommendation(
    results: list[TestResult], daily_volume: int = 10_000
) -> None:
    """Print a bold recommendation box: cheapest passing model + savings."""
    # Collect per-model stats
    model_stats: dict[str, dict] = {}
    for tr in results:
        for mr in tr.model_results:
            name = mr.model_name
            if name not in model_stats:
                model_stats[name] = {
                    "passed_all": True,
                    "total_cost": 0.0,
                    "latency": [],
                    "input_tokens": [],
                    "output_tokens": [],
                    "model": mr.model_config_obj.model,
                    "total_tests": 0,
                    "passed_tests": 0,
                }
            if mr.error or not mr.passed:
                model_stats[name]["passed_all"] = False
            else:
                model_stats[name]["passed_tests"] += 1
            model_stats[name]["total_tests"] += 1
            model_stats[name]["total_cost"] += mr.cost_usd
            model_stats[name]["latency"].append(mr.latency_ms)
            model_stats[name]["input_tokens"].append(mr.input_tokens)
            model_stats[name]["output_tokens"].append(mr.output_tokens)

    # Need at least 2 models to make a recommendation
    if len(model_stats) < 2:
        return

    passing = {k: v for k, v in model_stats.items() if v["passed_all"]}
    if not passing:
        return

    # Project costs
    for name, stats in model_stats.items():
        avg_in = sum(stats["input_tokens"]) / max(len(stats["input_tokens"]), 1)
        avg_out = sum(stats["output_tokens"]) / max(len(stats["output_tokens"]), 1)
        proj = project_cost(stats["model"], int(avg_in), int(avg_out), daily_volume)
        stats["monthly"] = proj["monthly"]
        stats["per_call"] = proj["per_call"]
        stats["avg_latency"] = sum(stats["latency"]) / max(len(stats["latency"]), 1)

    cheapest_name, cheapest = min(passing.items(), key=lambda x: x[1]["monthly"])
    most_exp_name, most_exp = max(model_stats.items(), key=lambda x: x[1]["monthly"])

    if cheapest_name == most_exp_name:
        return

    savings_monthly = most_exp["monthly"] - cheapest["monthly"]
    if savings_monthly <= 0:
        return

    savings_pct = (savings_monthly / most_exp["monthly"] * 100) if most_exp["monthly"] > 0 else 0

    # Speed comparison
    cheapest_lat = cheapest["avg_latency"]
    expensive_lat = most_exp["avg_latency"]
    speed_ratio = expensive_lat / cheapest_lat if cheapest_lat > 0 else 0

    # Build the recommendation box
    total_tests = cheapest["total_tests"]
    lines: list[str] = []
    lines.append("")
    lines.append(f"  [bold green]✅ Passes all {total_tests} tests[/bold green]", )

    badges: list[str] = []
    if speed_ratio > 1.1:
        badges.append(f"⚡ {speed_ratio:.1f}x faster")
    badges.append(f"💰 {savings_pct:.0f}% cheaper")
    if badges:
        lines.append(f"  {' · '.join(badges)}")

    lines.append("")
    if cheapest["monthly"] == 0:
        lines.append(f"  Saves [bold green]${savings_monthly:,.0f}/month[/bold green] ([bold green]${savings_monthly * 12:,.0f}/year[/bold green]) at {daily_volume:,} calls/day")
    else:
        lines.append(f"  Saves [bold green]${savings_monthly:,.0f}/month[/bold green] ([bold green]${savings_monthly * 12:,.0f}/year[/bold green]) at {daily_volume:,} calls/day")
    lines.append(f"  vs [dim]{most_exp_name}[/dim]")

    body = "\n".join(lines)
    title = f"[bold]💡 RECOMMENDATION: Switch to [cyan]{cheapest_name}[/cyan][/bold]"

    console.print(
        Panel(
            body,
            title=title,
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print()


def get_recommendation_data(
    results: list[TestResult], daily_volume: int = 10_000
) -> dict | None:
    """Extract recommendation data for use in reports. Returns None if no recommendation."""
    model_stats: dict[str, dict] = {}
    for tr in results:
        for mr in tr.model_results:
            name = mr.model_name
            if name not in model_stats:
                model_stats[name] = {
                    "passed_all": True,
                    "total_cost": 0.0,
                    "latency": [],
                    "input_tokens": [],
                    "output_tokens": [],
                    "model": mr.model_config_obj.model,
                    "total_tests": 0,
                    "passed_tests": 0,
                }
            if mr.error or not mr.passed:
                model_stats[name]["passed_all"] = False
            else:
                model_stats[name]["passed_tests"] += 1
            model_stats[name]["total_tests"] += 1
            model_stats[name]["total_cost"] += mr.cost_usd
            model_stats[name]["latency"].append(mr.latency_ms)
            model_stats[name]["input_tokens"].append(mr.input_tokens)
            model_stats[name]["output_tokens"].append(mr.output_tokens)

    if len(model_stats) < 2:
        return None

    passing = {k: v for k, v in model_stats.items() if v["passed_all"]}
    if not passing:
        return None

    for name, stats in model_stats.items():
        avg_in = sum(stats["input_tokens"]) / max(len(stats["input_tokens"]), 1)
        avg_out = sum(stats["output_tokens"]) / max(len(stats["output_tokens"]), 1)
        proj = project_cost(stats["model"], int(avg_in), int(avg_out), daily_volume)
        stats["monthly"] = proj["monthly"]
        stats["per_call"] = proj["per_call"]
        stats["avg_latency"] = sum(stats["latency"]) / max(len(stats["latency"]), 1)

    cheapest_name, cheapest = min(passing.items(), key=lambda x: x[1]["monthly"])
    most_exp_name, most_exp = max(model_stats.items(), key=lambda x: x[1]["monthly"])

    if cheapest_name == most_exp_name:
        return None

    savings_monthly = most_exp["monthly"] - cheapest["monthly"]
    if savings_monthly <= 0:
        return None

    savings_pct = (savings_monthly / most_exp["monthly"] * 100) if most_exp["monthly"] > 0 else 0
    cheapest_lat = cheapest["avg_latency"]
    expensive_lat = most_exp["avg_latency"]
    speed_ratio = expensive_lat / cheapest_lat if cheapest_lat > 0 else 0

    # Build model rows for report
    model_rows = []
    for name, stats in sorted(model_stats.items(), key=lambda x: x[1]["monthly"]):
        model_rows.append({
            "name": name,
            "passed_all": stats["passed_all"],
            "passed_tests": stats["passed_tests"],
            "total_tests": stats["total_tests"],
            "monthly": stats["monthly"],
            "per_call": stats["per_call"],
            "avg_latency": stats["avg_latency"],
            "avg_input_tokens": sum(stats["input_tokens"]) / max(len(stats["input_tokens"]), 1),
            "avg_output_tokens": sum(stats["output_tokens"]) / max(len(stats["output_tokens"]), 1),
        })

    return {
        "cheapest_name": cheapest_name,
        "most_expensive_name": most_exp_name,
        "savings_monthly": savings_monthly,
        "savings_yearly": savings_monthly * 12,
        "savings_pct": savings_pct,
        "speed_ratio": speed_ratio,
        "cheapest_monthly": cheapest["monthly"],
        "expensive_monthly": most_exp["monthly"],
        "total_tests": cheapest["total_tests"],
        "daily_volume": daily_volume,
        "model_rows": model_rows,
    }


def print_eval_results(eval_results: list[EvalResult]) -> None:
    """Print eval results summary table."""
    if not eval_results:
        return

    eval_name = eval_results[0].eval_case.name
    total_rows = len(eval_results[0].row_results) if eval_results else 0
    num_models = len(eval_results)

    console.print(
        f"  [bold]⚡ Litmux Eval:[/bold] {eval_name} "
        f"({total_rows} rows × {num_models} models)"
    )
    console.print()

    table = Table(show_header=True, header_style="bold", padding=(0, 1))
    table.add_column("Model", style="cyan", min_width=22)
    table.add_column("Pass Rate", justify="center", min_width=10)
    table.add_column("Avg Score", justify="center", min_width=10)
    table.add_column("Total $", justify="right", min_width=10)
    table.add_column("Avg Lat.", justify="right", min_width=9)
    table.add_column("Failures", justify="center", min_width=8)

    for er in eval_results:
        rate_color = "green" if er.pass_rate >= 90 else "yellow" if er.pass_rate >= 70 else "red"
        score_str = f"{er.avg_score:.1f}/10" if er.avg_score is not None else "—"
        failures = sum(1 for r in er.row_results if not r.passed)
        latency = f"{er.avg_latency_ms:.0f}ms"
        cost = f"${er.total_cost_usd:.4f}"

        table.add_row(
            er.model_name,
            f"[{rate_color}]{er.pass_rate:.1f}%[/{rate_color}]",
            score_str,
            cost,
            latency,
            str(failures),
        )

    console.print(table)
    console.print()

    # Show sample failures
    for er in eval_results:
        failures = [r for r in er.row_results if not r.passed][:5]
        if failures:
            console.print(f"  [red]Failures ({er.model_name}):[/red]")
            for f in failures:
                input_preview = str(list(f.inputs.values())[0])[:50] if f.inputs else "—"
                console.print(
                    f"    Row {f.row_index}: "
                    f'input="{input_preview}" '
                    f"expected={f.expected} "
                    f"got={f.actual_output[:50]}"
                )
            console.print()


def print_cost_projection(results: list[TestResult], daily_volume: int) -> None:
    """Print cost projections for each model."""
    console.print()
    console.print(f"  [bold]💰 Cost Projection[/bold] ({daily_volume:,} calls/day)")
    console.print()

    table = Table(show_header=True, header_style="bold", padding=(0, 1))
    table.add_column("Model", style="cyan", min_width=22)
    table.add_column("Per Call", justify="right", min_width=10)
    table.add_column("Monthly", justify="right", min_width=12)
    table.add_column("vs. Most Expensive", min_width=30)

    model_stats: dict[str, dict] = {}
    for tr in results:
        for mr in tr.model_results:
            name = mr.model_name
            if name not in model_stats:
                model_stats[name] = {"input_tokens": [], "output_tokens": [], "model": mr.model_config_obj.model}
            model_stats[name]["input_tokens"].append(mr.input_tokens)
            model_stats[name]["output_tokens"].append(mr.output_tokens)

    projections = {}
    for name, stats in model_stats.items():
        avg_in = sum(stats["input_tokens"]) / max(len(stats["input_tokens"]), 1)
        avg_out = sum(stats["output_tokens"]) / max(len(stats["output_tokens"]), 1)
        projections[name] = project_cost(stats["model"], int(avg_in), int(avg_out), daily_volume)

    max_monthly = max((p["monthly"] for p in projections.values()), default=0)

    for name, proj in sorted(projections.items(), key=lambda x: -x[1]["monthly"]):
        monthly = proj["monthly"]
        if monthly >= max_monthly * 0.99:
            comparison = "—"
        else:
            savings = max_monthly - monthly
            pct = (savings / max_monthly * 100) if max_monthly > 0 else 0
            comparison = f"[green]✅ Save ${savings:,.0f}/mo ({pct:.0f}% less)[/green]"

        table.add_row(name, f"${proj['per_call']:.6f}", f"${monthly:,.0f}", comparison)

    console.print(table)
    console.print()

    # ROI Summary
    if len(projections) > 1:
        cheapest = min(projections.items(), key=lambda x: x[1]["monthly"])
        most_exp = max(projections.items(), key=lambda x: x[1]["monthly"])
        savings = most_exp[1]["monthly"] - cheapest[1]["monthly"]
        if savings > 0:
            console.print(f"  [bold]💡 ROI Summary[/bold]")
            console.print(f"  Switch from [red]{most_exp[0]}[/red] → [green]{cheapest[0]}[/green]")
            console.print(f"  Monthly savings: [bold green]${savings:,.0f}[/bold green]")
            console.print(f"  Annual savings:  [bold green]${savings * 12:,.0f}[/bold green]")
            console.print()


def format_ci_output(results: list[TestResult]) -> str:
    """Format results as CI-friendly markdown."""
    lines = ["## ⚡ Litmux Results", ""]
    total = len(results)
    passed = sum(1 for r in results if r.all_passed)

    lines.append(f"{'✅' if passed == total else '❌'} **{passed}/{total} tests passed**")
    lines.append("")

    for tr in results:
        tc = tr.test_case
        lines.append(f"### Test: {tc.name}")
        lines.append("")
        lines.append("| Model | Result | Latency | Cost | Assertions |")
        lines.append("|-------|--------|---------|------|------------|")

        for mr in tr.model_results:
            if mr.error:
                lines.append(f"| {mr.model_name} | ❌ ERROR | {mr.latency_ms:.0f}ms | — | — |")
            else:
                result = "✅ PASS" if mr.passed else "❌ FAIL"
                lines.append(
                    f"| {mr.model_name} | {result} | {mr.latency_ms:.0f}ms | "
                    f"${mr.cost_usd:.6f} | {mr.pass_count}/{mr.total_assertions} |"
                )
        lines.append("")

    return "\n".join(lines)


def format_json_output(results: list[TestResult]) -> str:
    """Format results as JSON."""
    data = []
    for tr in results:
        test_data = {
            "test": tr.test_case.name,
            "prompt_source": tr.test_case.prompt_source,
            "all_passed": tr.all_passed,
            "models": [
                {
                    "model": mr.model_name,
                    "passed": mr.passed,
                    "latency_ms": round(mr.latency_ms, 1),
                    "cost_usd": mr.cost_usd,
                    "input_tokens": mr.input_tokens,
                    "output_tokens": mr.output_tokens,
                    "assertions": {"passed": mr.pass_count, "total": mr.total_assertions},
                    **({"error": mr.error} if mr.error else {}),
                }
                for mr in tr.model_results
            ],
        }
        data.append(test_data)

    return json.dumps(data, indent=2)


def print_compare(results: list[TestResult]) -> None:
    """Print side-by-side model output comparison."""
    for tr in results:
        tc = tr.test_case
        source = tc.prompt_source or "inline"

        console.print(f"  [bold]Test:[/bold] {tc.name}")
        console.print(f"  [dim]Prompt:[/dim] {source}")
        console.print()

        # Build panels for each model's output
        panels = []
        for mr in tr.model_results:
            # Header with stats
            if mr.error:
                title = f"[red]{mr.model_name}[/red]"
                body = f"[red]ERROR:[/red] {mr.error[:300]}"
            else:
                status = "[green]PASS[/green]" if mr.passed else "[red]FAIL[/red]"
                latency = f"{mr.latency_ms:.0f}ms" if mr.latency_ms < 10_000 else f"{mr.latency_ms / 1000:.1f}s"
                cost = f"${mr.cost_usd:.6f}" if mr.cost_usd < 0.01 else f"${mr.cost_usd:.4f}"
                assertions = f"{mr.pass_count}/{mr.total_assertions}" if mr.total_assertions > 0 else "—"

                title = f"[cyan]{mr.model_name}[/cyan]"
                tokens = f"in: {mr.input_tokens}  out: {mr.output_tokens}"
                stats = f"{status}  {latency}  {cost}  {tokens}  assertions: {assertions}"
                output_text = mr.output.strip()[:1500]

                body = f"{stats}\n{'─' * 40}\n{output_text}"

                # Show failed assertions
                failures = [r for r in mr.assertion_results if not r.passed]
                if failures:
                    body += f"\n{'─' * 40}"
                    for f in failures:
                        body += f"\n[red]✗[/red] {f.assertion.type.value}: {f.message}"

            panels.append(
                Panel(
                    body,
                    title=title,
                    border_style="dim",
                    expand=True,
                    padding=(1, 2),
                )
            )

        # Render side-by-side (2 columns max for readability)
        if len(panels) <= 3:
            console.print(Columns(panels, equal=True, expand=True))
        else:
            # More than 3 — render in rows of 2
            for i in range(0, len(panels), 2):
                batch = panels[i : i + 2]
                console.print(Columns(batch, equal=True, expand=True))

        console.print()

    # Winner summary
    console.print("─" * 60)
    console.print("  [bold]Summary[/bold]")
    console.print()

    # Aggregate stats per model across all tests
    model_stats: dict[str, dict] = {}
    for tr in results:
        for mr in tr.model_results:
            name = mr.model_name
            if name not in model_stats:
                model_stats[name] = {
                    "tests": 0, "passed": 0, "total_latency": 0.0,
                    "total_cost": 0.0, "errors": 0,
                }
            model_stats[name]["tests"] += 1
            if mr.error:
                model_stats[name]["errors"] += 1
            else:
                if mr.passed:
                    model_stats[name]["passed"] += 1
                model_stats[name]["total_latency"] += mr.latency_ms
                model_stats[name]["total_cost"] += mr.cost_usd

    table = Table(show_header=True, header_style="bold", padding=(0, 1))
    table.add_column("Model", style="cyan", min_width=30)
    table.add_column("Pass Rate", justify="center", min_width=10)
    table.add_column("Avg Latency", justify="right", min_width=10)
    table.add_column("Total Cost", justify="right", min_width=10)
    table.add_column("", min_width=8)

    # Sort by pass rate desc, then latency asc
    ranked = sorted(
        model_stats.items(),
        key=lambda x: (-x[1]["passed"], x[1]["total_latency"]),
    )

    for i, (name, s) in enumerate(ranked):
        run_count = s["tests"] - s["errors"]
        rate = f"{s['passed']}/{s['tests']}"
        avg_lat = f"{s['total_latency'] / max(run_count, 1):.0f}ms"
        cost = f"${s['total_cost']:.6f}"
        badge = "[bold green]👑 BEST[/bold green]" if i == 0 and s["passed"] > 0 else ""

        table.add_row(name, rate, avg_lat, cost, badge)

    console.print(table)

    # Cost savings insight
    if len(ranked) >= 2 and ranked[0][1]["passed"] > 0:
        best = ranked[0]
        worst_cost = max(ranked, key=lambda x: x[1]["total_cost"])
        if worst_cost[1]["total_cost"] > best[1]["total_cost"]:
            per_call_diff = (worst_cost[1]["total_cost"] - best[1]["total_cost"]) / max(best[1]["tests"], 1)
            monthly_savings = per_call_diff * 10_000 * 30
            if monthly_savings > 0:
                console.print(
                    f"  [green]💰 {best[0]} saves ${monthly_savings:,.0f}/mo "
                    f"vs {worst_cost[0]} at 10k calls/day[/green]"
                )

    console.print()
