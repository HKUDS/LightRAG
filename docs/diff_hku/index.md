# HKU diff audit — this version vs upstream HKUDS/main

This folder contains a focused, concise multi-document audit comparing the current repo state (local branch) against the original upstream HKUDS/LightRAG `upstream/main`.

Files included:

- summary.md — one-page top-level summary of the most important behavioral, security, and compatibility changes
- technical_diffs.md — file-by-file highlights and rationale (engineer-friendly)
- security_audit.md — prioritized security observations and fixes
- migration_guide.md — precise steps to deploy safely (DB migrations, env flags, testing)
- tests_needed.md — minimal E2E/unit tests required before release

Use these to quickly evaluate risk and plan next work.
