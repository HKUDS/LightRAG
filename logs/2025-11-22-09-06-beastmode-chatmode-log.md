Actions: Added explicit demo database credentials (user/password) and dev-only warning across starter help and docs.

Files changed:
- starter/Makefile — help output now prints demo username/password and dev-only warning; removed backticks that caused shell execution.
- starter/README.md — added a clear 'Demo credentials' block under Quick Start with the username, password, and a production-warning.
- starter/QUICK_START.md — corrected Web UI port to 3000 and clarified the database credentials block as "Demo database credentials (local/dev only)".

Decisions: Keep demo credentials visible for easier local onboarding, and clearly mark them as development-only. Avoid advertising Postgres on localhost by default (internal-only by default). Use `docker compose exec` to access DB or provide an opt-in override later if users need host access.

Next steps: (Optional) Add a small `docker-compose.override.yml` to publish Postgres to the host for debugging (opt-in), and add a short 'developer debugging' section to Quick Start showing how to temporarily expose ports.

Lessons/insights: Small, explicit credential guidance reduces friction for new users. Avoid printing backticks in Makefile echo statements to prevent shell execution.
