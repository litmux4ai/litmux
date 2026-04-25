"""Litmux CLI — the testing and evaluation platform for AI-powered software."""

from __future__ import annotations

import asyncio
import os
import time as time_module
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
import typer

# Load .env from current directory (user's project root)
load_dotenv()
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from litmux import __version__
from litmux.config import load_config
from litmux.display import (
    console,
    format_ci_output,
    format_json_output,
    print_compare,
    print_cost_projection,
    print_eval_results,
    print_header,
    print_recommendation,
    print_savings_summary,
    print_summary,
    print_test_result,
)
from litmux.evaluator import evaluate_assertions
from litmux.models import TestResult
from litmux.runner import run_models_parallel
from litmux.cloud import is_logged_in, sync_run, sync_eval

app = typer.Typer(
    name="litmux",
    help="⚡ The testing and evaluation platform for AI-powered software.",
    add_completion=False,
)


# Cloud sync is in private beta. Set LITMUX_CLOUD_ENABLED=1 to opt in.
CLOUD_ENABLED = os.environ.get("LITMUX_CLOUD_ENABLED") == "1"


def _cloud_beta_notice() -> None:
    console.print()
    console.print("  [bold yellow]Litmux Cloud is in private beta.[/bold yellow]")
    console.print("  Join the waitlist: [link=https://litmux.dev]https://litmux.dev[/link]")
    console.print("  Already have access? Set [bold]LITMUX_CLOUD_ENABLED=1[/bold] in your environment.")
    console.print()


class OutputFormat(str, Enum):
    table = "table"
    json = "json"
    ci = "ci"


# ─── litmux run ──────────────────────────────────────────────────


@app.command()
def run(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file"),
    test_name: Optional[str] = typer.Option(None, "--test", "-t", help="Run a specific test"),
    output: OutputFormat = typer.Option(OutputFormat.table, "--output", "-o", help="Output format"),
    ci: bool = typer.Option(False, "--ci", help="Shorthand for --output ci"),
    volume: Optional[int] = typer.Option(None, "--volume", "-v", help="Daily volume for cost projection"),
    verbose: bool = typer.Option(False, "--verbose", help="Show model outputs"),
    no_sync: bool = typer.Option(False, "--no-sync", help="Do not upload results to Litmux Cloud, even if logged in"),
    report: Optional[str] = typer.Option(None, "--report", help="Generate HTML cost report (default: litmux-report.html)"),
) -> None:
    """Run prompt tests across models."""
    if ci:
        output = OutputFormat.ci

    trigger = "ci" if ci else "cli"
    run_start = time_module.perf_counter()

    try:
        cfg = load_config(config)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    tests = cfg.tests
    if test_name:
        tests = [t for t in tests if t.name == test_name]
        if not tests:
            console.print(f"[red]Error:[/red] No test found with name '{test_name}'")
            raise typer.Exit(1)

    if output == OutputFormat.table:
        print_header()
        console.print(
            f"  Running [bold]{len(tests)}[/bold] test(s) across "
            f"[bold]{len(cfg.models)}[/bold] model(s)...\n"
        )

    results: list[TestResult] = []

    for tc in tests:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            if output == OutputFormat.table:
                progress.add_task(description=f"  Running test: {tc.name}...", total=None)
            model_results = asyncio.run(run_models_parallel(cfg.models, tc.prompt))

        for mr in model_results:
            if not mr.error:
                mr.assertion_results = evaluate_assertions(mr, tc.assertions)

        test_result = TestResult(test_case=tc, model_results=model_results)
        results.append(test_result)

        if output == OutputFormat.table:
            print_test_result(test_result)
            if verbose:
                for mr in test_result.model_results:
                    console.print(f"  [dim]── {mr.model_name} ──[/dim]")
                    console.print(f"  {mr.output[:500]}\n")

    # Output
    if output == OutputFormat.table:
        print_summary(results)
        if len(cfg.models) > 1:
            print_savings_summary(results, daily_volume=volume or 10_000)
            print_recommendation(results, daily_volume=volume or 10_000)
        if volume:
            print_cost_projection(results, volume)
    elif output == OutputFormat.ci:
        print(format_ci_output(results))
    elif output == OutputFormat.json:
        print(format_json_output(results))

    # Generate HTML report
    if report is not None:
        from litmux.report import generate_report

        report_path = report if report else "litmux-report.html"
        html = generate_report(results, daily_volume=volume or 10_000)
        with open(report_path, "w") as f:
            f.write(html)
        console.print(f"  [green]📄 Report saved → {report_path}[/green]")
        try:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(report_path)}")
        except Exception:
            pass

    # Sync to Litmux Cloud
    duration_ms = (time_module.perf_counter() - run_start) * 1000
    if CLOUD_ENABLED and is_logged_in() and not no_sync:
        run_id = sync_run(results, duration_ms=duration_ms, trigger=trigger)
        if run_id and output == OutputFormat.table:
            console.print(f"  [dim]Run synced → {_dashboard_url()}/runs/{run_id[:8]}[/dim]\n")
    elif not is_logged_in() and output == OutputFormat.table:
        # Quietly suppress the cloud-sync tip while the cloud is in private beta.
        pass

    if not all(r.all_passed for r in results):
        raise typer.Exit(1)


# ─── litmux eval ─────────────────────────────────────────────────


@app.command()
def eval(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file"),
    eval_name: Optional[str] = typer.Option(None, "--eval", "-e", help="Run specific eval"),
    output: OutputFormat = typer.Option(OutputFormat.table, "--output", "-o", help="Output format"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max rows to evaluate"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Specific model only"),
    verbose: bool = typer.Option(False, "--verbose", help="Show per-row details"),
    no_sync: bool = typer.Option(False, "--no-sync", help="Do not upload results to Litmux Cloud, even if logged in"),
) -> None:
    """Run evaluations against datasets."""
    from litmux.eval_runner import run_eval

    try:
        cfg = load_config(config)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    evals = cfg.evals
    if not evals:
        console.print("[red]Error:[/red] No evals defined in config.")
        raise typer.Exit(1)

    if eval_name:
        evals = [e for e in evals if e.name == eval_name]
        if not evals:
            console.print(f"[red]Error:[/red] No eval found with name '{eval_name}'")
            raise typer.Exit(1)

    # Filter models if specified
    models = cfg.models
    if model:
        models = [m for m in models if m.model == model]
        if not models:
            console.print(f"[red]Error:[/red] Model '{model}' not found in config")
            raise typer.Exit(1)

    run_start = time_module.perf_counter()

    if output == OutputFormat.table:
        print_header()

    for eval_case in evals:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            if output == OutputFormat.table:
                progress.add_task(description=f"  Running eval: {eval_case.name}...", total=None)
            eval_results = asyncio.run(run_eval(cfg, eval_case, models=models, limit=limit))

        if output == OutputFormat.table:
            print_eval_results(eval_results)

    # Sync to Litmux Cloud
    duration_ms = (time_module.perf_counter() - run_start) * 1000
    if CLOUD_ENABLED and is_logged_in() and not no_sync:
        run_id = sync_run([], duration_ms=duration_ms, trigger="cli")
        if run_id:
            sync_eval(eval_results, run_id)
            if output == OutputFormat.table:
                console.print(f"  [dim]Eval synced → {_dashboard_url()}/runs/{run_id[:8]}[/dim]\n")


# ─── litmux generate ─────────────────────────────────────────────


@app.command()
def generate(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Path to prompt template"),
    seed: str = typer.Option(..., "--seed", "-s", help="Seed CSV with example rows"),
    n: int = typer.Option(30, "--n", help="Number of scenarios"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Also save as local CSV"),
    description: Optional[str] = typer.Option(None, "--description", "-d", help="What the prompt does (auto-detected from seed)"),
    gen_model: str = typer.Option("claude-haiku-4-5-20251001", "--model", help="Model for generation"),
) -> None:
    """Generate test datasets with AI."""
    from litmux.dataset import generate_dataset, save_dataset_csv, extract_variables
    from pathlib import Path

    print_header()

    # Read prompt template
    if not os.path.exists(prompt):
        console.print(f"[red]Error:[/red] Prompt file not found: {prompt}")
        raise typer.Exit(1)

    if not os.path.exists(seed):
        console.print(f"[red]Error:[/red] Seed file not found: {seed}")
        raise typer.Exit(1)

    # Auto-derive description from seed filename if not provided
    desc = description or Path(seed).stem.replace("_", " ").replace("-", " ")

    with open(prompt) as f:
        template = f.read()

    variables = extract_variables(template)
    console.print(f"  Task: {desc}")
    console.print(f"  Variables: {', '.join(f'{{{{{v}}}}}' for v in variables)}")
    console.print(f"  Seed: {seed}")
    console.print(f"  Generating {n} scenarios...\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(description=f"  Generating {n} scenarios with {gen_model}...", total=None)
            rows = asyncio.run(generate_dataset(
                template, desc, n=n, model=gen_model, seed_path=seed,
            ))
    except (ValueError, TypeError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] Generation failed: {e}")
        raise typer.Exit(1)

    # Save locally if -o provided
    if output:
        save_dataset_csv(rows, output)
        console.print(f"  [green]✅ Saved locally → {output}[/green]")
    else:
        console.print(f"  [green]✅ Generated {len(rows)} scenarios[/green]")

    if not is_logged_in() and not output:
        console.print("  [dim]Tip: Use -o to save locally, or run [bold]litmux login[/bold] to sync to dashboard[/dim]")

    # Breakdown
    types: dict[str, int] = {}
    for row in rows:
        st = row.get("scenario_type", "unknown")
        types[st] = types.get(st, 0) + 1

    console.print("\n  Breakdown:")
    for st, count in sorted(types.items(), key=lambda x: -x[1]):
        pct = count / len(rows) * 100 if rows else 0
        console.print(f"    {st}: {count} ({pct:.0f}%)")
    console.print()


# ─── litmux cost ─────────────────────────────────────────────────


@app.command()
def cost(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file"),
    volume: int = typer.Option(1000, "--volume", "-v", help="Daily call volume"),
    report: Optional[str] = typer.Option(None, "--report", help="Generate HTML cost report (default: litmux-report.html)"),
) -> None:
    """Project costs across models without running tests."""
    try:
        cfg = load_config(config)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    print_header()
    console.print("  Running sample calls to estimate token usage...\n")

    sample_test = cfg.tests[0] if cfg.tests else None
    if not sample_test:
        console.print("[red]Error:[/red] Need at least one test case for cost estimation.")
        raise typer.Exit(1)

    model_results = asyncio.run(run_models_parallel(cfg.models, sample_test.prompt))
    test_result = TestResult(test_case=sample_test, model_results=model_results)
    print_cost_projection([test_result], volume)

    if len(cfg.models) > 1:
        print_recommendation([test_result], daily_volume=volume)

    # Generate HTML report
    if report is not None:
        from litmux.report import generate_report

        report_path = report if report else "litmux-report.html"
        html = generate_report([test_result], daily_volume=volume)
        with open(report_path, "w") as f:
            f.write(html)
        console.print(f"  [green]📄 Report saved → {report_path}[/green]")
        try:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(report_path)}")
        except Exception:
            pass


# ─── litmux history ──────────────────────────────────────────────


@app.command()
def history(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of runs"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Filter by project"),
) -> None:
    """Show recent run history."""
    if not CLOUD_ENABLED:
        _cloud_beta_notice()
        raise typer.Exit(0)
    if not is_logged_in():
        console.print("[red]Error:[/red] Run [bold]litmux login[/bold] first to view history.")
        raise typer.Exit(1)

    from litmux.cloud import get_history

    print_header()
    runs = get_history(limit=limit, project=project)

    if not runs:
        console.print("  No runs found.")
        return

    table = Table(show_header=True, header_style="bold", padding=(0, 1))
    table.add_column("Run ID", style="dim", min_width=10)
    table.add_column("Time", min_width=18)
    table.add_column("Branch", style="cyan", min_width=12)
    table.add_column("Result", justify="center", min_width=8)
    table.add_column("Tests", justify="center", min_width=8)
    table.add_column("Duration", justify="right", min_width=8)
    table.add_column("Trigger", justify="center", min_width=6)

    for r in runs:
        all_pass = r["passed_tests"] == r["total_tests"]
        result = "[green]✅ PASS[/green]" if all_pass else "[red]❌ FAIL[/red]"
        duration = f"{r['duration_ms']:.0f}ms" if r["duration_ms"] < 10000 else f"{r['duration_ms']/1000:.1f}s"

        table.add_row(
            r["id"][:8],
            r["created_at"][:19].replace("T", " "),
            r.get("git_branch") or "—",
            result,
            f"{r['passed_tests']}/{r['total_tests']}",
            duration,
            r.get("trigger", "cli"),
        )

    console.print(table)
    console.print()


# ─── litmux init ─────────────────────────────────────────────────


@app.command()
def init() -> None:
    """Create a sample litmux.yaml config file."""
    config_path = "litmux.yaml"

    if os.path.exists(config_path):
        console.print(f"[yellow]Warning:[/yellow] {config_path} already exists.")
        raise typer.Exit(0)

    os.makedirs("prompts", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)

    def _write_if_missing(path: str, content: str) -> None:
        if os.path.exists(path):
            console.print(f"  [dim]skip {path} (already exists)[/dim]")
            return
        with open(path, "w") as f:
            f.write(content)

    _write_if_missing(
        "prompts/summarize.txt",
        "Summarize the following text in 2-3 sentences. Be concise and capture the key facts:\n\n{{text}}",
    )

    _write_if_missing(
        "prompts/classify.txt",
        "Classify the following customer support ticket into exactly one category.\n"
        'Return ONLY a JSON object with a "category" field.\n\n'
        "Categories: auth, billing, shipping, returns, product, technical, other\n\n"
        "Ticket: {{ticket}}",
    )

    _write_if_missing(
        "datasets/sample_tickets.csv",
        "text,expected_category,difficulty\n"
        '"I can\'t login to my account",auth,easy\n'
        '"My credit card was charged twice",billing,medium\n'
        '"Where is my package?",shipping,easy\n'
        '"I want to return this item",returns,easy\n'
        '"The app crashes when I open it",technical,medium\n'
        '"How do I change my password?",auth,easy\n'
        '"\u00bfD\u00f3nde est\u00e1 mi pedido?",shipping,edge_case\n'
        '"",null,edge_case\n',
    )

    sample_config = """\
# Litmux Configuration
# Docs: https://github.com/litmux/litmux

models:
  - model: gpt-4o-mini                        # OpenAI
  - model: claude-haiku-4-5-20251001           # Anthropic
  # - model: gemini-2.0-flash                  # Google (needs GOOGLE_API_KEY)
  # - provider: huggingface                    # HuggingFace (needs HF_TOKEN)
  #   model: meta-llama/Llama-3.1-8B-Instruct

tests:
  - name: summarize_earnings
    prompt: prompts/summarize.txt
    inputs:
      text: >
        The quarterly earnings report shows revenue increased by 15% year over year,
        reaching $4.2 billion. Operating margins improved to 28%, up from 24% in the
        prior year. The company announced a new $10 billion share buyback program.
    assert:
      - type: contains
        value: "revenue"
      - type: contains
        value: "15%"
      - type: latency-less-than
        value: 5000
      - type: cost-less-than
        value: 0.01

  - name: classify_basic
    prompt: prompts/classify.txt
    inputs:
      ticket: "I can't login to my account"
    assert:
      - type: json-valid
      - type: contains
        value: "auth"

evals:
  - name: ticket_classifier
    prompt: prompts/classify.txt
    dataset: datasets/sample_tickets.csv
    input_mapping:
      ticket: text
    expected: expected_category
    assert:
      - type: json-valid
    judge:
      criteria: "Did the model correctly classify the support ticket?"
      threshold: 7.0
"""
    _write_if_missing(config_path, sample_config)

    # Create .env.example if not present
    if not os.path.exists(".env.example"):
        with open(".env.example", "w") as f:
            f.write(
                "# Litmux — API Keys\n"
                "# Copy this file: cp .env.example .env\n"
                "# Then fill in the keys for the providers you use.\n\n"
                "# HuggingFace — https://huggingface.co/settings/tokens\n"
                "HF_TOKEN=\n\n"
                "# OpenAI — https://platform.openai.com/api-keys\n"
                "OPENAI_API_KEY=\n\n"
                "# Anthropic — https://console.anthropic.com/settings/keys\n"
                "ANTHROPIC_API_KEY=\n\n"
                "# Google Gemini — https://aistudio.google.com/apikey\n"
                "GOOGLE_API_KEY=\n"
            )

    console.print("[green]✅ Created:[/green]")
    console.print(f"  • {config_path}")
    console.print("  • prompts/summarize.txt")
    console.print("  • prompts/classify.txt")
    console.print("  • datasets/sample_tickets.csv")
    console.print("  • .env.example")
    console.print()
    console.print("Next steps:")
    console.print("  1. [bold]cp .env.example .env[/bold] and add your API keys")
    console.print("  2. Run tests:      [bold]litmux run[/bold]")
    console.print("  3. Generate data:  [bold]litmux generate -p prompts/classify.txt -s datasets/sample_tickets.csv --n 20[/bold]")
    console.print("  4. Run evals:      [bold]litmux eval[/bold]")
    console.print("  5. Compare costs:  [bold]litmux cost --volume 10000[/bold]")
    console.print()

# ─── litmux login ────────────────────────────────────────────────

DEFAULT_DASHBOARD_URL = "https://app.litmux.dev"


def _dashboard_url() -> str:
    return os.environ.get("LITMUX_DASHBOARD_URL", DEFAULT_DASHBOARD_URL).rstrip("/")


@app.command()
def login() -> None:
    """Authenticate with Litmux Cloud."""
    if not CLOUD_ENABLED:
        _cloud_beta_notice()
        raise typer.Exit(0)

    import webbrowser

    from litmux.cloud import save_token

    dashboard = _dashboard_url()
    auth_url = f"{dashboard}/auth/cli"

    console.print()
    console.print("  [bold cyan]⚡ Litmux Cloud[/bold cyan]")
    console.print(f"  Open in your browser: [link={auth_url}]{auth_url}[/link]")
    console.print()

    try:
        webbrowser.open(auth_url)
    except Exception:
        # Headless / no display — user follows the printed URL.
        pass

    token = typer.prompt("  Paste your token")
    if not token.strip():
        console.print("[red]Error:[/red] No token provided.")
        raise typer.Exit(1)

    save_token(token.strip())
    console.print("  [green]✅ Logged in![/green] Results will now sync to Litmux Cloud.")
    console.print()


@app.command()
def logout() -> None:
    """Log out of Litmux Cloud."""
    from litmux.cloud import remove_token, is_logged_in

    if not is_logged_in():
        console.print("  Not logged in.")
        return

    remove_token()
    console.print("  [green]✅ Logged out.[/green] Results will no longer sync.")


@app.command()
def dashboard() -> None:
    """Open the Litmux dashboard in your browser."""
    if not CLOUD_ENABLED:
        _cloud_beta_notice()
        raise typer.Exit(0)

    import webbrowser

    url = _dashboard_url()

    if not is_logged_in():
        console.print("  [yellow]Note:[/yellow] Not logged in. Run [bold]litmux login[/bold] to sync results.")
        console.print()

    console.print()
    console.print(f"  [bold cyan]⚡ Litmux Dashboard[/bold cyan]")
    console.print(f"  Opening [link={url}]{url}[/link]")
    console.print()
    try:
        webbrowser.open(url)
    except Exception:
        pass


# ─── litmux compare ───────────────────────────────────────────────


@app.command()
def compare(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file"),
    test_name: Optional[str] = typer.Option(None, "--test", "-t", help="Compare a specific test"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Compare a specific model across tests"),
) -> None:
    """Compare model outputs side-by-side."""
    try:
        cfg = load_config(config)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    tests = cfg.tests
    if not tests:
        console.print("[red]Error:[/red] No tests defined in config.")
        raise typer.Exit(1)

    if test_name:
        tests = [t for t in tests if t.name == test_name]
        if not tests:
            console.print(f"[red]Error:[/red] No test found with name '{test_name}'")
            raise typer.Exit(1)

    models = cfg.models
    if model:
        models = [m for m in models if m.model == model]
        if not models:
            console.print(f"[red]Error:[/red] Model '{model}' not found in config")
            raise typer.Exit(1)

    print_header()
    console.print(
        f"  Comparing [bold]{len(tests)}[/bold] test(s) across "
        f"[bold]{len(models)}[/bold] model(s)...\n"
    )

    results: list[TestResult] = []

    for tc in tests:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(description=f"  Running: {tc.name}...", total=None)
            model_results = asyncio.run(run_models_parallel(models, tc.prompt))

        for mr in model_results:
            if not mr.error:
                mr.assertion_results = evaluate_assertions(mr, tc.assertions)

        results.append(TestResult(test_case=tc, model_results=model_results))

    print_compare(results)

    if not all(r.all_passed for r in results):
        raise typer.Exit(1)


# ─── litmux cache ────────────────────────────────────────────────


@app.command()
def cache(
    clear: bool = typer.Option(False, "--clear", help="Clear all cached responses"),
) -> None:
    """Manage the response cache."""
    from litmux.cache import clear_cache, CACHE_DIR

    if clear:
        count = clear_cache()
        console.print(f"  [green]✅ Cleared {count} cached response(s)[/green]")
    else:
        if CACHE_DIR.exists():
            entries = list(CACHE_DIR.glob("*.json"))
            console.print(f"  Cache: {len(entries)} response(s) in {CACHE_DIR}/")
            console.print(f"  [dim]Use --clear to remove, or set LITMUX_NO_CACHE=1 to disable[/dim]")
        else:
            console.print("  Cache is empty.")


# ─── litmux version ──────────────────────────────────────────────


@app.command()
def version() -> None:
    """Show the Litmux version."""
    console.print(f"⚡ Litmux v{__version__}")


if __name__ == "__main__":
    app()
