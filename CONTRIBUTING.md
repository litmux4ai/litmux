# Contributing to Litmux

Thanks for your interest in contributing.

## Setup

```bash
git clone https://github.com/litmux/litmux.git
cd litmux
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Verify the install:

```bash
litmux --help
pytest
```

## Environment

Copy `.env.example` to `.env` and add the API keys you need. Only set the providers you plan to test — others are skipped automatically.

Set `LITMUX_NO_CACHE=1` to disable the response cache during development.

## Project Layout

```
litmux/          Python package
  cli.py         Typer commands
  runner.py      Async model runners
  evaluator.py   Assertion handlers + LLM judge
  eval_runner.py Bulk evaluation engine
  dataset.py     CSV/JSON loader + AI generation
  cost.py        Pricing tables and projections
  config.py      YAML config parser
  models.py      Pydantic data models
  display.py     Rich terminal output
  cache.py       Response cache
  cloud.py       Cloud sync client

tests/           Pytest suite
examples/        Sample projects
```

## Code Style

- Python 3.11+, `from __future__ import annotations` at the top of every module
- Type hints on public functions
- Use `rich` for all terminal output, never bare `print`
- Async for all network I/O

## Adding a CLI Command

1. Add Pydantic models to `litmux/models.py` if needed.
2. Add the command to `litmux/cli.py` with `@app.command()`.
3. Add output formatting in `litmux/display.py`.
4. Add tests in `tests/`.

## Adding an Assertion Type

1. Add the enum value in `litmux/models.py` (`AssertionType`).
2. Add the handler in `litmux/evaluator.py` and register it in the `handlers` dict.
3. Document it in the README assertion table.
4. Add tests in `tests/test_evaluator.py`.

## Pull Requests

- Keep PRs focused — one feature or fix per PR.
- Run `pytest` before pushing.
- Use conventional commit prefixes: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
