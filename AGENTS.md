# Project Guide for AI Agents

This AGENTS.md file provides operational guidance for AI assistants collaborating on the LightRAG codebase. Use it to understand the repository layout, preferred tooling, and expectations for adding or modifying functionality.

## Core Purpose

LightRAG is an advanced Retrieval-Augmented Generation (RAG) framework designed to enhance information retrieval and generation through graph-based knowledge representation. The project aims to provide a more intelligent and efficient way to process and retrieve information from documents by leveraging both graph structures and vector embeddings.

## Project Structure for Navigation

- `/lightrag`: Core Python package (ingestion, querying, storage abstractions, utilities). Key modules include `lightrag/lightrag.py` orchestration, `operate.py` pipeline helpers, `kg/` storage backends, `llm/` bindings, and `utils*.py`.
- `/lightrag/api`: FastAPI with Gunicorn for production. FastAPI service for LightRAG , auth, WebUI assets live in `lightrag_server.py`. Routers live in `routers/`, shared helpers in `utils_api.py`. Gunicorn startup logic lives in `run_with_gunicorn.py`.
- `/lightrag_webui`: React 19 + TypeScript + Tailwind front-end built with Vite/Bun. Uses component folders under `src/` and configuration via `env.*.sample`.
- `/inputs`, `/rag_storage`, `/dickens`, `/temp`: data directories. Treat contents as mutable working data; avoid committing generated artefacts.
- `/tests` and root-level `test_*.py`: Integration and smoke-test scripts (graph storage, API endpoints, behaviour regressions). Many expect specific environment variables or services.
- `/docs`, `/k8s-deploy`, `docker-compose.yml`: Deployment notes, Kubernetes manifests, and container orchestration helpers.
- Configuration templates: `.env.example`, `config.ini.example`, `lightrag.service.example`. Copy and adapt for local runs without committing secrets.

## Environment Setup and Tooling

- Python 3.10 is required. Recommended bootstrap:

  ```bash
  # Development installation
  python -m venv .venv
  source .venv/bin/activate
  pip install -e .
  pip install -e .[api]

  # Start API server
  lightrag-server

  # Production deployment
  lightrag-gunicorn --workers 3
  ```

- Duplicate `.env.example` to `.env` and adjust storage, LLM, and reranker bindings. Mirror `config.ini.example` when customising pipeline defaults.
- Storage backends (PostgreSQL, Redis, Neo4j, Milvus, etc.) are selected via `LIGHTRAG_*` environment variables. Ensure connection URLs and credentials are in place before running ingestion or tests.
- CLI entry points: `python -m lightrag` for package usage, `lightrag-server` (or `uvicorn lightrag.api.lightrag_server:app --reload`) for the API, `lightrag-gunicorn` for production gunicorn runs.
- Front-end work: install dependencies with `bun install` (preferred) or `npm install`, then use `bunx --bun vite` commands defined in `package.json`.

## Frontend Development

- **Package Manager**: **ALWAYS USE BUN** - Never use npm or yarn unless Bun is unavailable
  **Commands**:
  - `bun install` - Install dependencies

  - `bun run dev` - Start development server

  - `bun run build` - Build for production

  - `bun run lint` - Run linting

  - `bun test` - Run tests

  - `bun run preview` - Preview production build

- **Pattern**: All frontend operations must use Bun commands
- **Testing**: Use `bun test` for all frontend testing

## Coding Conventions

- Embrace type hints, dataclasses, and asynchronous patterns already present in `lightrag/lightrag.py` and storage implementations. Keep long-running jobs within `asyncio` flows and reuse helpers from `lightrag.operate`.
- Honour abstraction boundaries: new storage providers should inherit from the relevant base classes in `lightrag.base`; reusable logic belongs in `utils.py`/`utils_graph.py`.
- Use `lightrag.utils.logger` (not bare `print`) and let environment toggles (`VERBOSE`, `LOG_LEVEL`) control verbosity.
- Respect configuration defaults in `lightrag/constants.py`, extending with care and synchronising related documentation when behaviour changes.
- API additions should live under `lightrag/api/routers`, leverage dependency injections from `utils_api.py`, and return structured responses consistent with existing handlers.
- Front-end code should remain in TypeScript, rely on functional React components with hooks, and follow Tailwind utility style. Co-locate component-specific styles; reserve custom CSS for cases Tailwind cannot cover.
- Storage Backends
  - **Default**: In-memory with file persistence
  - **Production Options**: PostgreSQL, MongoDB, Redis, Neo4j
  - **Pattern**: Abstract storage interface with multiple implementations

* Lock Key Generation Consistency
  - **Critical Pattern**: Always sort parameters for lock key generation to prevent deadlocks
  - **Example**: `sorted_key_parts = sorted([src, tgt])` before creating lock key
  - **Why**: Prevents different lock keys for same relationship pair processed in different orders
  - **Apply to**: Any function that uses locks with multiple parameters
* Priority Queue Implementation
  - **Pattern**: Use priority-based task queuing for LLM requests
  - **Benefits**: Critical operations get higher priority
  - **Implementation**: Lower priority values = higher priority

## Testing and Quality Gates

- Run Python tests with `python -m pytest tests` for the FastAPI suite, and execute targeted scripts (for example `python tests/test_graph_storage.py`, `python test_lock_fix.py`) when touching related functionality. Many scripts require running backing services; check `.env` for prerequisites.
- Perform linting via `ruff check .` (configured in `pyproject.toml`) and address warnings. For formatting, match the existing style rather than introducing new tools.
- Front-end validation: `bun test`, `bunx --bun vite build`, and `bunx --bun vite lint`. The `*-no-bun` scripts exist if Bun is unavailable.
- When touching deployment assets, ensure `docker-compose config` or relevant `kubectl` dry-runs succeed before submitting changes.

## Runtime and Operational Notes

- Knowledge ingestion expects documents inside `inputs/` and writes intermediate state to `rag_storage/`. Keep these directories gitignored; never check in private data or large artefacts.
- Use `operate.py` helpers (e.g., `chunking_by_token_size`, `extract_entities`) to keep ingestion behaviour consistent. If extending the pipeline, document new steps in `docs/` and update any affected CLI usage.
- The API and core package rely on `.env`/`config.ini` being co-located with the current working directory. Scripts such as `tests/test_graph_storage.py` dynamically read these files; ensure they are in sync.

## Contribution Checklist

1. Run `pre-commit run --all-files` before sumitting PR.
2. Describe the change, affected modules, and operational impact in your PR. Mention any new environment knobs or storage dependencies.
3. Link related issues or discussions when available.
4. Confirm all applicable checks pass (`ruff`, pytest suite, targeted integration scripts, front-end build/tests when touched).
5. Capture screenshots or GIFs for front-end or API changes that affect user-visible behaviour.
6. Keep each PR focused on a single concern and update documentation (`README.md`, `docs/`, `.env.example`) when behaviour or configuration changes.

Follow this playbook to keep LightRAG contributions predictable, testable, and production-ready.
