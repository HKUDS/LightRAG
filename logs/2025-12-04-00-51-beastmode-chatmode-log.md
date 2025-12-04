Actions:
- Pulled upstream (HKUDS/LightRAG) and diffed HEAD vs upstream/main
- Inspected and documented major changes (multi-tenant support, security hardening, RLS, RBAC, config defaults)
- Created concise docs under docs/diff_hku: index.md, summary.md, technical_diffs.md, security_audit.md, migration_guide.md, tests_needed.md

Decisions:
- Focused on security, DB migrations, and runtime wiring as top priorities
- Kept documents concise but dense for engineering and DevOps audiences

Next steps:
- Add DB migrations and instrument DB session setter for RLS
- Implement CI tests for RLS + tenant isolation and permission matrix
- Run e2e tests under staging Postgres before production rollout

Lessons / insights:
- Multi-tenant changes are substantive — require DB migrations + end-to-end tests to avoid silent data leakage
- Default secrets and env defaults are currently unsafe for production; rotate and require via env validation
