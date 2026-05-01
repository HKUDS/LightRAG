# Release Process

## Steps

1. Confirm scope and feature flags.
2. Run backend, enterprise, frontend, SpecOps, and integration gates.
3. Verify rollback.
4. Review secrets, audit, workspace isolation, costs, legal review, and exports.

## Acceptance Criteria

- READY is declared only after all release gates pass.
- Residual risks are documented with owners or blockers.
- Phase-specific release notes and cleanup inventories should be linked when diagnostic smokes create additive artifacts.
