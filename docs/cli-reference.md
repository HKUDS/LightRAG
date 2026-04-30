# CLI Reference

## Project Commands

- `python -m lightrag_enterprise.system.migrate`
- `uv run ruff check lightrag_enterprise tests_enterprise lightrag/api/lightrag_server.py`
- `./scripts/test.sh tests_enterprise -q`
- `node /Users/joao_tourinho/Documents/specops-tooling-os/packages/cli/dist/index.js validate`

## Pilot Commands

- `scripts/little_bull_phase3_pilot.py` runs opt-in data-plane pilots.
- `scripts/little_bull_phase3_inventory.py` lists pilot artifacts without deletion.

## Acceptance Criteria

- CLI commands do not echo secrets.
- Destructive commands are not run without explicit confirmation.
