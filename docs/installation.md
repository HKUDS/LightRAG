# Installation

## Local Setup

Use the repository setup flow and keep secrets in local environment files that are not committed.

## Commands

- `python -m venv .venv && source .venv/bin/activate`
- `pip install -e .`
- `pip install -e .[api]`
- `cd lightrag_webui && bun install`

## Acceptance Criteria

- Dependencies install without committing generated environments.
- `.env` values remain local and are never printed in shared logs.
