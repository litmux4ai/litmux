# Litmux

Unit tests for AI. Test prompts, compare models, catch regressions.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
  <img src="https://img.shields.io/badge/tests-107%20passing-brightgreen" />
</p>

```bash
pip install litmux && litmux init && litmux run
```

---

## Why

Every team shipping AI features hits the same three problems:

1. **No testing standard.** REST has Postman, frontends have Cypress. LLM calls have manual spot-checking.
2. **Prompt regression is invisible.** A one-word change can silently break 15% of edge cases.
3. **Model selection is vibes.** "We use GPT-4o because it's good" — but is it $15k/month better than Gemini Flash?

Litmux gives you a YAML config, pass/fail assertions, and a cost report. That's it.

---

## Quick Start

```bash
pip install litmux

cp .env.example .env
# Add at least one: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, HF_TOKEN

litmux init    # scaffold a project
litmux run     # run tests against all configured models
```

No database, no cloud account, no Docker.

---

## Core Commands

### `litmux run` — unit tests for prompts

```yaml
# litmux.yaml
models:
  - model: gpt-4o-mini
  - model: claude-haiku-4-5-20251001

tests:
  - name: summarize_earnings
    prompt: prompts/summarize.txt
    inputs:
      text: "Revenue grew 15% to $4.2 billion..."
    assert:
      - type: contains
        value: "revenue"
      - type: cost-less-than
        value: 0.01
```

### `litmux eval` — bulk evaluation against datasets

```yaml
evals:
  - name: ticket_classifier
    prompt: prompts/classify.txt
    dataset: datasets/support_tickets.csv
    input_mapping:
      ticket: text
    expected: expected_category
    assert:
      - type: json-valid
    judge:
      criteria: "Did the model correctly classify the ticket?"
      threshold: 7.0
```

### `litmux generate` — AI-generated test datasets

```bash
litmux generate \
  --prompt prompts/classify.txt \
  --seed datasets/sample_tickets.csv \
  --n 50 \
  --output datasets/support_tickets.csv
```

### `litmux cost` — cost projection across models

```bash
litmux cost --volume 50000
```

Finds the cheapest model that passes your tests.

### `litmux compare` — side-by-side model outputs

```bash
litmux compare
```

---

## Cloud (Optional, Free)

Sync results to a hosted dashboard for history, trends, and team visibility.

```bash
litmux login       # one-time browser auth
litmux run         # results auto-sync
litmux dashboard   # open app.litmux.dev
```

The CLI works fully offline. Cloud is opt-in.

---

## Assertion Types

| Type | Description |
|------|-------------|
| `contains` | Output contains substring |
| `not-contains` | Output does not contain substring |
| `regex` | Output matches regex pattern |
| `json-valid` | Output is valid JSON |
| `json-schema` | Output has required JSON keys |
| `cost-less-than` | Cost below threshold (USD) |
| `latency-less-than` | Latency below threshold (ms) |
| `llm-judge` | LLM scores output 1–10 against criteria |

---

## CI/CD

```yaml
# .github/workflows/litmux.yml
- run: litmux run --ci
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

---

## Configuration

```yaml
models:
  - provider: openai | anthropic | google | huggingface
    model: string
    temperature: 0.0
    max_tokens: 1024

defaultTest:
  assert:
    - type: cost-less-than
      value: 0.01

tests:
  - name: string
    prompt: path/to/prompt.txt
    inputs: { variable: "value" }
    assert:
      - type: contains
        value: "expected"

evals:
  - name: string
    prompt: path/to/prompt.txt
    dataset: path/to/data.csv
    input_mapping: { prompt_var: csv_column }
    expected: csv_column
    assert: [...]
    judge:
      criteria: "..."
      threshold: 7.0
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI models, LLM judge, dataset generation |
| `ANTHROPIC_API_KEY` | Anthropic models |
| `GOOGLE_API_KEY` | Google models |
| `HF_TOKEN` | HuggingFace models |
| `LITMUX_NO_CACHE` | Set to `1` to skip the response cache |
| `LITMUX_API_URL` | Override cloud API endpoint (default: `https://api.litmux.dev`) |
| `LITMUX_API_URL_ALLOW_INSECURE` | Set to `1` to allow non-HTTPS `LITMUX_API_URL` (local dev only) |
| `LITMUX_DASHBOARD_URL` | Override dashboard URL (default: `https://app.litmux.dev`) |
| `LITMUX_JUDGE_MODEL` | LLM model used for `llm-judge` assertions (default: `gpt-4o-mini`) |
| `LITMUX_CLOUD_ENABLED` | Set to `1` to opt in to Litmux Cloud (private beta) |

---

## All Commands

```
litmux run                    Run all tests
litmux run -t <name>          Run a specific test
litmux run --ci               CI output (markdown)
litmux eval                   Run all evals
litmux eval --limit 10        Evaluate first N rows
litmux generate ...           Generate a test dataset
litmux compare                Side-by-side model outputs
litmux cost -v 50000          Project monthly cost
litmux cache                  View / clear response cache
litmux init                   Scaffold a new project
litmux version                Show version

# Cloud (private beta — join the waitlist at https://litmux.dev)
litmux login                  Authenticate with Litmux Cloud
litmux logout                 Remove saved credentials
litmux history                Recent runs from cloud
litmux dashboard              Open the dashboard
```

---

## Examples

See [`examples/`](examples/) for three ready-to-run projects:

- `01-quickstart` — minimal single-model test
- `02-multi-model` — compare across providers
- `03-generate-and-eval` — AI-generated dataset + LLM judge

---

## License

MIT
