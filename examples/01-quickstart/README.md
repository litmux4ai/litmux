# Example 1: Quickstart

Test a single prompt against one model in 30 seconds.

## What you'll learn

- How to set up API keys
- How to run a simple test
- How to interpret pass/fail results

## Setup

```bash
pip install litmux
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

## Run

```bash
litmux run -c litmux.yaml
```

This will:
1. Load the config from `litmux.yaml`
2. Send "Say hello" to `gpt-4o-mini`
3. Check if the response contains "hello"
4. Show pass/fail with latency and cost

## What you'll see

```
⚡ Litmux v0.1.0 — CI/CD for AI

  Test: hello_world
  ┌──────────────────────┬──────────┬─────────┬──────────┐
  │ Model                │  Result  │ Latency │     Cost │
  ├──────────────────────┼──────────┼─────────┼──────────┤
  │ gpt-4o-mini          │ ✅ PASS  │ 520ms   │ $0.0000  │
  └──────────────────────┴──────────┴─────────┴──────────┘

  ✅ All 1 tests passed!
```

## Next steps

- Try adding more models in `litmux.yaml`
- Add more assertion types (`json-valid`, `cost-less-than`)
- → Continue to [Example 2: Multi-Model](../02-multi-model/README.md)
