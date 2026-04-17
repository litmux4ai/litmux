# Example 3: Generate Test Data + Eval

The full Litmux workflow: AI generates edge-case test scenarios from seed data, then you evaluate prompt quality across models.

## What you'll learn

- How `--seed` teaches AI your data format
- How AI generates adversarial, multilingual, and edge case scenarios
- How eval compares pass rates across models

## Setup

```bash
cp .env.example .env
# Add OPENAI_API_KEY and/or ANTHROPIC_API_KEY
```

## The seed file

`datasets/seed_tickets.csv` has 5 example rows:
```csv
text,expected_category,difficulty
"I can't login to my account",auth,easy
"My credit card was charged twice",billing,medium
"Where is my package?",shipping,easy
```

The `--seed` flag teaches AI to generate rows in this exact format — no JSON blobs, just clean individual columns.

## Run the full flow

```bash
# Step 1: Generate 20 test scenarios from 5 seed examples
litmux generate \
  -p prompts/classify.txt \
  -s datasets/seed_tickets.csv \
  --n 20 \
  -o datasets/generated.csv

# Step 2: Eval against the generated dataset
litmux eval -c litmux.yaml --limit 10

# Step 3: See cost savings
litmux cost -c litmux.yaml --volume 50000
```

## What you'll see

**Step 1** — AI generates 20 diverse scenarios:
```
✅ Generated 20 scenarios → datasets/generated.csv

  Breakdown:
    happy_path:   10 (50%)
    edge_case:     4 (20%)
    adversarial:   2 (10%)
    multilingual:  2 (10%)
    empty_input:   1 (5%)
    long_input:    1 (5%)
```

**Step 2** — Eval shows pass rate per model:
```
  ┌──────────────────────┬───────────┬───────────┬──────────┐
  │ Model                │ Pass Rate │ Avg Score │ Failures │
  ├──────────────────────┼───────────┼───────────┼──────────┤
  │ gpt-4o-mini          │ 94.0%     │ 8.2/10    │ 1        │
  │ claude-haiku-4.5     │ 90.0%     │ 7.8/10    │ 2        │
  └──────────────────────┴───────────┴───────────┴──────────┘
```

**Step 3** — Cost shows annual savings from picking the right model.
