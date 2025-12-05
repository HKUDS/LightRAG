# Task Logs - Add Makefile Dry-run

**Date**: 2025-12-05 13:12
**Mode**: beastmode

## Actions
- Added `reset-demo-tenants-dry-run` Makefile target to preview `reset-demo-tenants` workflow without executing destructive steps.
- Updated help text in root `Makefile`.
- Tested `make -n reset-demo-tenants-dry-run` to ensure output is as expected.
- Committed changes (commit b69214e2) and pushed branch `premerge/integration-upstream`.

## Decisions
- Implemented dry-run by using `make -n -C starter db-reset` to safely preview the destructive step and echo the subsequent script invocation.

## Next steps
- Optionally implement a `seed-only` non-destructive target that will only run the API seeding script.

## Lessons
- Dry-run commands provide a safe way for developers to confirm destructive workflows before executing them.
