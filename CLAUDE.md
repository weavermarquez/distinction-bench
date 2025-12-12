# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Note**: This project uses [bd (beads)](https://github.com/steveyegge/beads) for issue tracking. Use `bd` commands instead of markdown TODOs. See AGENTS.md for workflow details.

## Project Overview

**distinction-bench** is a benchmark for evaluating LLM reasoning on Laws of Form arithmetic. Spencer-Brown's Laws of Form defines a minimal calculus with two axioms:

- **I1 (Calling)**: `()() → ()` — Adjacent marks condense to one
- **I2 (Crossing)**: `(()) → void` — Nested mark cancels to void

Every expression reduces to either `()` (marked) or void (unmarked).

## Development Commands

```bash
# Environment setup
uv sync                    # Install dependencies
uv run pytest              # Run tests
uv run pytest -x -v        # Run tests, stop on first failure, verbose
uv run pytest tests/test_form.py::test_name  # Run single test

# Linting
uv run ruff check .        # Lint
uv run ruff format .       # Format

# CLI (after package is built)
uv run bench generate --n 1000 --seed 2025 --out data/cases.jsonl
uv run bench validate --in data/cases.jsonl

# Pre-commit
pre-commit install         # Install hooks
pre-commit run --all-files # Run manually
```

## Architecture

### Core Library (`src/`)

- `form.py` — Canonical Form representation (nested lists), parsing (`string_to_form`, `form_to_string`), JSON conversion, validation, `depth()`, `size()`
- `simplify.py` — Ground-truth simplifier applying I1/I2 until normal form; `simplify()`, `evaluate()`, `simplify_parens()`
- `generate.py` — Deterministic random form generator with depth/width/marks controls
- `dataset.py` — Dataclasses for test cases (single + composite), JSONL I/O
- `cli.py` — CLI entry points for generate/validate commands

### Representation

Internal: Nested Python lists. `[[]]` = `(())`, `[[], []]` = `()()`

Normal forms:
- **void**: Empty root list `[]`
- **mark**: Single mark `[[]]` renders as `()`

### Test Structure (`tests/`)

- Parsing roundtrips (string ↔ form ↔ JSON)
- Simplification correctness on known examples
- Generator well-formedness and constraint adherence
- Evaluation matches expected outputs

## Constraints

- Python 3.11+
- Use `uv` for environment/dependency management
- Prefer stdlib; minimize external dependencies
- Type hints throughout
- No model API calls in core library (provider-agnostic)
- Keep notebooks under `notebooks/` as research examples only
