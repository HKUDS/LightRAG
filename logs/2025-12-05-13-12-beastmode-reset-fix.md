Task logs

Actions:
- Inspected top-level Makefile and `starter/Makefile` to find cause of `make reset-demo-tenants` failure.
- Implemented robust readiness checks and auto-start logic for Postgres in `starter/Makefile`:
  - Added a pg_isready wait loop in `init-db`.
  - Added a service-start-and-wait step before `DROP/terminate` in `db-reset`.
- Ran dry-run checks (`make -n`) to validate the updated Makefile commands.

Decisions:
- Making Makefile resilient to containers that are still starting avoids the `service "postgres" is not running` error.
- Prefer to attempt `docker compose up -d postgres` if not found and wait up to a sensible timeout.

Next steps:
- Start the compose stack and re-run the reset workflow:
  - make -C starter up  # start services
  - make reset-demo-tenants  # now should succeed
- If issues remain, check postgres logs and compose status:
  - docker compose -f starter/docker-compose.yml -p lightrag-multitenant ps
  - docker compose -f starter/docker-compose.yml -p lightrag-multitenant logs -f postgres

Lessons/insights:
- Race conditions are common when running compose-based init scripts immediately after a `docker compose up`.
- Defensively checking container presence and readiness improves robustness for local dev workflows.
