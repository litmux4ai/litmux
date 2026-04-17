# Example 2: Multi-Model Comparison

Compare GPT-4o-mini, Claude Haiku, and Llama side-by-side. See which passes, which is fastest, and which is cheapest.

## What you'll learn

- How to test the same prompt across multiple providers
- How cost savings are calculated automatically
- How the 👑 BEST badge works

## Setup

```bash
cp .env.example .env
# Add keys for the providers you want to test
# At minimum: OPENAI_API_KEY
# Optional: ANTHROPIC_API_KEY, HF_TOKEN
```

## Run

```bash
# Test all models
litmux run -c litmux.yaml

# Side-by-side output comparison
litmux compare -c litmux.yaml

# Project costs at 50k calls/day
litmux cost -c litmux.yaml --volume 50000
```

## What you'll see

```
✅ All 2 tests passed!

💡 Cost Insight (at 10,000 calls/day)
  Cheapest passing model: llama-3.1-8b (FREE)
  vs most expensive: claude-haiku-4.5 ($79/mo)
  → Save $79/month ($943/year)
```

## Next steps

- → Continue to [Example 3: Generate & Eval](../03-generate-and-eval/README.md)
