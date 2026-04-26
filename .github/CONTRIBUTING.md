# Contributing to LightRAG

Thank you for your interest in contributing! This guide covers everything you need to get started.

## Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Running Tests](#running-tests)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Bugs](#reporting-bugs)
- [Requesting Features](#requesting-features)

---

## Ways to Contribute

- **Bug reports** — open an [issue](https://github.com/HKUDS/LightRAG/issues) using the Bug Report template
- **Feature requests** — open an [issue](https://github.com/HKUDS/LightRAG/issues) using the Feature Request template
- **Documentation** — fix typos, clarify explanations, or add examples
- **Code** — fix bugs, implement features, or add storage/LLM backends
- **Testing** — add test coverage for untested code paths

---

## Development Setup

```bash
# Clone the repository
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG

# Install in development mode (requires uv)
uv sync
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install with optional extras as needed
uv sync --extra api              # FastAPI server
uv sync --extra test             # Test dependencies
uv sync --extra offline-storage  # Storage backends
uv sync --extra offline-llm      # Additional LLM providers

# Set up pre-commit hooks (run once)
pip install pre-commit
pre-commit install
```

---

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for formatting and linting, enforced via [pre-commit](https://pre-commit.com/).

### Automatic fixing

Running `pre-commit run --all-files` will automatically fix most style issues:

```bash
# Fix all files
pre-commit run --all-files

# Fix only staged files (faster during development)
pre-commit run
```

### What is checked

| Hook | What it does |
|------|-------------|
| `trailing-whitespace` | Removes trailing whitespace |
| `end-of-file-fixer` | Ensures files end with a newline |
| `requirements-txt-fixer` | Keeps `requirements.txt` entries sorted |
| `ruff-format` | Formats Python code (Black-compatible) |
| `ruff` | Fixes Python lint errors |

### CI check

The same checks run automatically on every pull request. If the CI check fails, run `pre-commit run --all-files` locally, commit the fixes, and push again.

### Language conventions

- **Python code and comments**: English
- **Frontend (WebUI)**: uses i18next for internationalization — add translation keys rather than hardcoding strings

---

## Running Tests

```bash
# Run offline tests (no external services required)
python -m pytest tests

# Run integration tests (requires configured external services)
python -m pytest tests --run-integration

# Run a specific test file
python -m pytest tests/test_lightrag.py

# Keep test artifacts for debugging
python -m pytest tests --keep-artifacts
```

Set `LIGHTRAG_RUN_INTEGRATION=true` as an environment variable as an alternative to `--run-integration`.

---

## Submitting a Pull Request

1. **Fork** the repository and create a branch from `main`:
   ```bash
   git checkout -b fix/your-descriptive-branch-name
   ```

2. **Make your changes** and ensure:
   - Pre-commit checks pass: `pre-commit run --all-files`
   - Relevant tests pass: `python -m pytest tests`
   - New behavior is covered by tests where applicable

3. **Commit** with a clear message describing *why* the change was made:
   ```bash
   git commit -m "fix: handle permission-only encrypted PDFs without password"
   ```

4. **Push** and open a pull request against `main`. Fill out the pull request template completely.

5. **Respond to review feedback** — a maintainer will review your PR and may request changes.

### Pull request checklist

- [ ] Changes tested locally
- [ ] Pre-commit checks pass (`pre-commit run --all-files`)
- [ ] Unit/integration tests added or updated where applicable
- [ ] Documentation updated if behavior changes
- [ ] PR description explains the *why*, not just the *what*

---

## Reporting Bugs

Please use the [Bug Report issue template](https://github.com/HKUDS/LightRAG/issues/new?template=bug_report.yml). Include:

- LightRAG version and Python version
- Storage backend and LLM provider being used
- Minimal reproducible example
- Full error traceback

---

## Requesting Features

Please use the [Feature Request issue template](https://github.com/HKUDS/LightRAG/issues/new?template=feature_request.yml). Describe:

- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

---

## Questions

For usage questions, check the [Discussions](https://github.com/HKUDS/LightRAG/discussions) tab or open a [Question issue](https://github.com/HKUDS/LightRAG/issues/new?template=question.yml).
