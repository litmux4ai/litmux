"""Shareable HTML report generation for Litmux."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone

from litmux import __version__
from litmux.display import get_recommendation_data
from litmux.models import TestResult


def _get_git_info() -> dict[str, str]:
    """Get current git branch and commit, silently returning empty on failure."""
    info: dict[str, str] = {}
    try:
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass
    return info


def generate_report(
    results: list[TestResult],
    daily_volume: int = 10_000,
) -> str:
    """Generate a self-contained HTML cost report."""
    rec = get_recommendation_data(results, daily_volume)
    git = _get_git_info()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Collect all model data for the table
    model_data: list[dict] = []
    model_agg: dict[str, dict] = {}
    for tr in results:
        for mr in tr.model_results:
            name = mr.model_name
            if name not in model_agg:
                model_agg[name] = {
                    "name": name,
                    "passed_all": True,
                    "total_tests": 0,
                    "passed_tests": 0,
                    "total_cost": 0.0,
                    "latencies": [],
                    "input_tokens": [],
                    "output_tokens": [],
                }
            if mr.error or not mr.passed:
                model_agg[name]["passed_all"] = False
            else:
                model_agg[name]["passed_tests"] += 1
            model_agg[name]["total_tests"] += 1
            model_agg[name]["total_cost"] += mr.cost_usd
            model_agg[name]["latencies"].append(mr.latency_ms)
            model_agg[name]["input_tokens"].append(mr.input_tokens)
            model_agg[name]["output_tokens"].append(mr.output_tokens)

    for name, agg in model_agg.items():
        avg_lat = sum(agg["latencies"]) / max(len(agg["latencies"]), 1)
        avg_in = sum(agg["input_tokens"]) / max(len(agg["input_tokens"]), 1)
        avg_out = sum(agg["output_tokens"]) / max(len(agg["output_tokens"]), 1)
        model_data.append({
            "name": name,
            "passed_all": agg["passed_all"],
            "passed_tests": agg["passed_tests"],
            "total_tests": agg["total_tests"],
            "total_cost": agg["total_cost"],
            "avg_latency": avg_lat,
            "avg_input_tokens": avg_in,
            "avg_output_tokens": avg_out,
        })

    model_data.sort(key=lambda x: x["total_cost"])

    # Build HTML
    hero_html = _build_hero(rec)
    chart_html = _build_chart(rec)
    table_html = _build_table(model_data)
    roi_html = _build_roi(rec)
    test_details_html = _build_test_details(results)
    git_html = ""
    if git:
        parts = []
        if git.get("branch"):
            parts.append(f"Branch: <strong>{_esc(git['branch'])}</strong>")
        if git.get("commit"):
            parts.append(f"Commit: <code>{_esc(git['commit'])}</code>")
        git_html = " · ".join(parts)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Litmux Cost Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0a0a; color: #e0e0e0; padding: 40px 20px; max-width: 900px; margin: 0 auto; }}
  .header {{ text-align: center; margin-bottom: 40px; }}
  .header h1 {{ font-size: 14px; color: #666; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px; }}
  .header .meta {{ font-size: 13px; color: #555; }}
  .hero {{ background: linear-gradient(135deg, #0d1f0d, #0a1a0a); border: 1px solid #1a3a1a; border-radius: 12px; padding: 32px; text-align: center; margin-bottom: 32px; }}
  .hero h2 {{ font-size: 28px; color: #4ade80; margin-bottom: 12px; }}
  .hero .subtitle {{ font-size: 16px; color: #a0a0a0; }}
  .hero .badges {{ margin-top: 16px; display: flex; justify-content: center; gap: 16px; flex-wrap: wrap; }}
  .hero .badge {{ background: #1a2a1a; border: 1px solid #2a4a2a; padding: 6px 16px; border-radius: 20px; font-size: 14px; color: #4ade80; }}
  .section {{ margin-bottom: 32px; }}
  .section h3 {{ font-size: 14px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #222; }}
  .chart {{ display: flex; flex-direction: column; gap: 12px; }}
  .chart-row {{ display: flex; align-items: center; gap: 12px; }}
  .chart-label {{ width: 200px; font-size: 14px; text-align: right; color: #ccc; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .chart-bar-container {{ flex: 1; background: #1a1a1a; border-radius: 6px; height: 32px; position: relative; overflow: hidden; }}
  .chart-bar {{ height: 100%; border-radius: 6px; display: flex; align-items: center; padding-left: 12px; font-size: 13px; font-weight: 600; color: #fff; min-width: 60px; transition: width 0.3s ease; }}
  .chart-bar.cheapest {{ background: linear-gradient(90deg, #166534, #22c55e); }}
  .chart-bar.passing {{ background: linear-gradient(90deg, #1e3a5f, #3b82f6); }}
  .chart-bar.failing {{ background: linear-gradient(90deg, #3f1515, #ef4444); }}
  .chart-value {{ width: 100px; font-size: 14px; text-align: right; color: #888; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th {{ text-align: left; padding: 10px 12px; color: #888; font-weight: 600; border-bottom: 2px solid #222; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #1a1a1a; }}
  tr:hover td {{ background: #111; }}
  .pass {{ color: #4ade80; }}
  .fail {{ color: #ef4444; }}
  .roi {{ background: linear-gradient(135deg, #0d1f0d, #0a1a0a); border: 1px solid #1a3a1a; border-radius: 12px; padding: 24px; display: flex; justify-content: space-around; text-align: center; flex-wrap: wrap; gap: 16px; }}
  .roi-item {{ flex: 1; min-width: 150px; }}
  .roi-item .label {{ font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }}
  .roi-item .value {{ font-size: 28px; font-weight: 700; color: #4ade80; }}
  .roi-item .value.neutral {{ color: #888; }}
  .test-card {{ background: #111; border: 1px solid #222; border-radius: 8px; padding: 16px; margin-bottom: 12px; }}
  .test-card h4 {{ font-size: 15px; margin-bottom: 8px; color: #e0e0e0; }}
  .test-card .prompt {{ font-size: 12px; color: #555; margin-bottom: 8px; }}
  .model-row {{ display: flex; justify-content: space-between; padding: 4px 0; font-size: 13px; border-bottom: 1px solid #1a1a1a; }}
  .model-row:last-child {{ border-bottom: none; }}
  .footer {{ text-align: center; padding: 32px 0; font-size: 13px; color: #444; border-top: 1px solid #1a1a1a; margin-top: 40px; }}
  .footer a {{ color: #3b82f6; text-decoration: none; }}
  .no-rec {{ background: #111; border: 1px solid #222; border-radius: 12px; padding: 32px; text-align: center; }}
  .no-rec h2 {{ font-size: 22px; color: #e0e0e0; margin-bottom: 8px; }}
  .no-rec p {{ color: #888; }}
</style>
</head>
<body>
  <div class="header">
    <h1>⚡ Litmux Cost Report</h1>
    <div class="meta">{now}{(' · ' + git_html) if git_html else ''}</div>
  </div>

  {hero_html}
  {chart_html}
  {table_html}
  {roi_html}
  {test_details_html}

  <div class="footer">
    Generated by <a href="https://github.com/litmux/litmux">Litmux</a> v{__version__} · <a href="https://litmux.dev">litmux.dev</a>
  </div>
</body>
</html>"""
    return html


def _esc(s: str) -> str:
    """Escape HTML special characters."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _build_hero(rec: dict | None) -> str:
    if not rec:
        return """<div class="no-rec">
  <h2>Model Comparison Complete</h2>
  <p>Run with 2+ models to see cost optimization recommendations.</p>
</div>"""

    badges = []
    badges.append(f'<span class="badge">✅ Passes all {rec["total_tests"]} tests</span>')
    if rec["speed_ratio"] > 1.1:
        badges.append(f'<span class="badge">⚡ {rec["speed_ratio"]:.1f}x faster</span>')
    badges.append(f'<span class="badge">💰 {rec["savings_pct"]:.0f}% cheaper</span>')

    return f"""<div class="hero">
  <h2>Switch to {_esc(rec["cheapest_name"])}</h2>
  <div class="subtitle">Cheapest model that passes all tests — saves ${rec["savings_monthly"]:,.0f}/month vs {_esc(rec["most_expensive_name"])}</div>
  <div class="badges">{''.join(badges)}</div>
</div>"""


def _build_chart(rec: dict | None) -> str:
    if not rec or not rec.get("model_rows"):
        return ""

    rows = rec["model_rows"]
    max_monthly = max((r["monthly"] for r in rows), default=1)
    if max_monthly == 0:
        max_monthly = 1

    bar_rows = []
    for r in sorted(rows, key=lambda x: x["monthly"], reverse=True):
        pct = (r["monthly"] / max_monthly * 100) if max_monthly > 0 else 0
        pct = max(pct, 3)  # minimum visible width

        if r["name"] == rec["cheapest_name"]:
            bar_class = "cheapest"
        elif r["passed_all"]:
            bar_class = "passing"
        else:
            bar_class = "failing"

        status = "✅" if r["passed_all"] else "❌"
        cost_label = f"${r['monthly']:,.0f}/mo" if r["monthly"] > 0 else "FREE"

        bar_rows.append(f"""    <div class="chart-row">
      <div class="chart-label">{status} {_esc(r["name"])}</div>
      <div class="chart-bar-container">
        <div class="chart-bar {bar_class}" style="width: {pct:.1f}%">{cost_label}</div>
      </div>
      <div class="chart-value">{cost_label}</div>
    </div>""")

    return f"""<div class="section">
  <h3>Monthly Cost Comparison ({rec["daily_volume"]:,} calls/day)</h3>
  <div class="chart">
{chr(10).join(bar_rows)}
  </div>
</div>"""


def _build_table(model_data: list[dict]) -> str:
    rows = []
    for m in model_data:
        status = '<span class="pass">✅ PASS</span>' if m["passed_all"] else '<span class="fail">❌ FAIL</span>'
        cost = f"${m['total_cost']:.6f}" if m["total_cost"] < 0.01 else f"${m['total_cost']:.4f}"
        latency = f"{m['avg_latency']:.0f}ms" if m["avg_latency"] < 10_000 else f"{m['avg_latency'] / 1000:.1f}s"

        rows.append(f"""    <tr>
      <td><strong>{_esc(m["name"])}</strong></td>
      <td>{status}</td>
      <td>{m["passed_tests"]}/{m["total_tests"]}</td>
      <td>{cost}</td>
      <td>{latency}</td>
      <td>{m["avg_input_tokens"]:.0f}</td>
      <td>{m["avg_output_tokens"]:.0f}</td>
    </tr>""")

    return f"""<div class="section">
  <h3>Model Results</h3>
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Result</th>
        <th>Tests</th>
        <th>Cost</th>
        <th>Latency</th>
        <th>In Tokens</th>
        <th>Out Tokens</th>
      </tr>
    </thead>
    <tbody>
{chr(10).join(rows)}
    </tbody>
  </table>
</div>"""


def _build_roi(rec: dict | None) -> str:
    if not rec:
        return ""

    return f"""<div class="section">
  <h3>ROI Summary</h3>
  <div class="roi">
    <div class="roi-item">
      <div class="label">Monthly Savings</div>
      <div class="value">${rec["savings_monthly"]:,.0f}</div>
    </div>
    <div class="roi-item">
      <div class="label">Annual Savings</div>
      <div class="value">${rec["savings_yearly"]:,.0f}</div>
    </div>
    <div class="roi-item">
      <div class="label">Cost Reduction</div>
      <div class="value">{rec["savings_pct"]:.0f}%</div>
    </div>
    <div class="roi-item">
      <div class="label">Daily Volume</div>
      <div class="value neutral">{rec["daily_volume"]:,}</div>
    </div>
  </div>
</div>"""


def _build_test_details(results: list[TestResult]) -> str:
    if not results:
        return ""

    cards = []
    for tr in results:
        tc = tr.test_case
        source = tc.prompt_source or "inline"

        model_rows = []
        for mr in tr.model_results:
            if mr.error:
                status = '<span class="fail">ERROR</span>'
                detail = _esc(mr.error[:80])
            else:
                status = '<span class="pass">PASS</span>' if mr.passed else '<span class="fail">FAIL</span>'
                cost = f"${mr.cost_usd:.6f}" if mr.cost_usd < 0.01 else f"${mr.cost_usd:.4f}"
                latency = f"{mr.latency_ms:.0f}ms"
                detail = f"{cost} · {latency} · {mr.pass_count}/{mr.total_assertions} assertions"

            model_rows.append(f"""        <div class="model-row">
          <span>{_esc(mr.model_name)} {status}</span>
          <span style="color: #666">{detail}</span>
        </div>""")

        cards.append(f"""    <div class="test-card">
      <h4>{_esc(tc.name)}</h4>
      <div class="prompt">Prompt: {_esc(source)}</div>
{chr(10).join(model_rows)}
    </div>""")

    return f"""<div class="section">
  <h3>Test Details</h3>
{chr(10).join(cards)}
</div>"""
