Actions:
- Generated a per-commit mapping of upstream/main commits (docs/diff_hku/unmerged_upstream_mapping.csv) and a summary (docs/diff_hku/unmerged_upstream_mapping_summary.md).
- Mapped 747 upstream commits into categories to support ordered cherry-pick planning and integration.

Decisions/assumptions:
- Matching uses heuristic regex patterns to categorize commits; this is sufficient for planning but may need manual review for ambiguous commits.

Next steps:
- Offer to produce an ordered cherry-pick plan CSV (category-prioritized) or to try an automated cherry-pick for the first N commits and run tests in a premerge/integration branch.
