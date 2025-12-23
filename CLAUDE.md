# LightRAG-MT Development Guidelines

Auto-generated from all feature plans. Last updated: 2025-12-01

## Active Technologies
- Python 3.10+ (existing codebase) + FastAPI, Pydantic, asyncio, uvicorn, OpenAI SDK (003-api-usage-metering)
- PostgreSQL (via existing `postgres_impl.py` patterns), workspace-namespaced (003-api-usage-metering)

- Python 3.10+ + FastAPI, Pydantic, asyncio, uvicorn (001-multi-workspace-server)

## Project Structure

```text
src/
tests/
```

## Commands

cd src; pytest; ruff check .

## Code Style

Python 3.10+: Follow standard conventions

## Recent Changes
- 003-api-usage-metering: Added Python 3.10+ (existing codebase) + FastAPI, Pydantic, asyncio, uvicorn, OpenAI SDK

- 001-multi-workspace-server: Added Python 3.10+ + FastAPI, Pydantic, asyncio, uvicorn

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
