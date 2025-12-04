Actions: Updated `starter/Makefile` help output to be shorter and more actionable; added 'Get started' summary and clarified PostgreSQL is internal-only.

Decisions: Keep help output verbose enough to list helpful commands but add a short actionable summary at the end directing users to `make up` and the main URLs; avoid incorrectly suggesting Postgres is bound to localhost by default.

Next steps: (Optional) Update any README/QUICK_START references elsewhere to match 'internal-only' messaging, or add an override compose file for dev port mapping.

Lessons/insights: Users prefer a single 'Get started' step and the exact URLs/command nearby; avoid showing incorrect port bindings when services are intentionally internal-only.
