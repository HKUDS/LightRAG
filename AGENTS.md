# Repository Guidelines

LightRAG is an advanced Retrieval-Augmented Generation (RAG) framework designed to enhance information retrieval and generation through graph-based knowledge representation.

## Project Structure & Module Organization
- `lightrag/`: Core Python package with orchestrators (`lightrag/lightrag.py`), storage adapters in `kg/`, LLM bindings in `llm/`, and helpers such as `operate.py` and `utils_*.py`.
- `lightrag-api/`: FastAPI service (`lightrag_server.py`) with routers under `routers/` and Gunicorn launcher `run_with_gunicorn.py`.
- `lightrag_webui/`: React 19 + TypeScript client driven by Bun + Vite; UI components live in `src/`.
- Tests live in `tests/` and root-level `test_*.py`. Working datasets stay in `inputs/`, `rag_storage/`, `temp/`; deployment collateral lives in `docs/`, `k8s-deploy/`, and `docker-compose.yml`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: set up the Python runtime.
- `pip install -e .` / `pip install -e .[api]`: install the package and API extras in editable mode.
- `lightrag-server` or `uvicorn lightrag.api.lightrag_server:app --reload`: start the API locally; ensure `.env` is present.
- `python -m pytest tests` or `python test_graph_storage.py`: run the full suite or a targeted script.
- `ruff check .`: lint Python sources before committing.
- `bun install`, `bun run dev`, `bun run build`, `bun test`: manage the web UI workflow (Bun is mandatory).

## Coding Style & Naming Conventions
- Backend code follow PEP 8 with four-space indentation, annotate functions, and reach for dataclasses when modelling state.
- Use `lightrag.utils.logger` instead of `print`; respect logger configuration flags.
- Extend storage or pipeline abstractions via `lightrag.base` and keep reusable helpers in the existing `utils_*.py`.
- Python modules remain lowercase with underscores; React components use `PascalCase.tsx` and hooks-first patterns.
- Front-end code should remain in TypeScript with two-space indentation, rely on functional React components with hooks, and follow Tailwind utility style.

## Testing Guidelines
- Add pytest cases beside the affected module or the relevant `test_*.py`; functions should start with `test_`.
- Export required `LIGHTRAG_*` environment variables before running integration or storage tests.
- For UI updates, pair code with Vitest specs and run `bun test`.

## Commit & Pull Request Guidelines
- Use concise, imperative commit subjects (e.g., `Fix lock key normalization`) and add body context only when necessary.
- PRs should include a summary, operational impact, linked issues, and screenshots or API samples for user-facing work.
- Verify `ruff check .`, `python -m pytest`, and affected Bun commands succeed before requesting review; note the runs in the PR text.

## Security & Configuration Tips
- Copy `.env.example` and `config.ini.example`; never commit secrets or real connection strings.
- Configure storage backends through `LIGHTRAG_*` variables and validate them with `docker-compose` services when needed.
- Treat `lightrag.log*` as local artefacts; purge sensitive information before sharing logs or outputs.
