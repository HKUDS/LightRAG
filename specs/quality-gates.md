# TRAG Quality Gates

## Required Commands

- `python -m lightrag_enterprise.system.migrate`
- `uv run ruff check lightrag_enterprise tests_enterprise lightrag/api/lightrag_server.py`
- `./scripts/test.sh tests_enterprise -q`
- `node /Users/joao_tourinho/Documents/specops-tooling-os/packages/cli/dist/index.js validate`

## Product Gates

- Postgres health.
- Neo4j health and authenticated local profile outside diagnostic runs.
- Qdrant health with compatible client/server versions.
- Workspace isolation and audit trails.

## Acceptance Criteria

- READY is not declared until all phase gates and release gates pass.
- Failed checks are reported with residual risk.
- Secrets are never printed in validation output.
