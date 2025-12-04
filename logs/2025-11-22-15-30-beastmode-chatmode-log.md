Actions: Updated docker-compose to avoid exposing Postgres host port and updated starter docs.

Files changed:
- starter/docker-compose.yml — removed host port mapping for postgres, added `expose: "5432"` (internal-only).
- starter/README.md — clarified PostgreSQL is internal-only (not exposed to host).
- starter/QUICK_START.md — clarified Postgres is internal-only and updated troubleshooting notes.

Decisions: Keep Postgres accessible only on the compose network for security; provide guidance to access via `docker compose exec` or temporary port mapping if needed.

Next steps: If you want me to update other docs referencing `-p 5432:5432` or to add an override file that maps ports for local debugging, I can do that next.

Lessons: Using `expose` keeps services communicating on the network while preventing accidental host exposure. If host access is required, explicitly opt-in via override or port-forwarding.
