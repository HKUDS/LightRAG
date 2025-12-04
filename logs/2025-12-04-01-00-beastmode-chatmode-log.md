Actions:
- Expanded cherry-pick plan from top-10 to top-25 upstream commits (docs/diff_hku/cherry_pick_top10.md updated).
- The expanded list focuses on chunking correctness, tests, workspace isolation, DOCX/XLSX parser improvements, JSON sanitizers, Postgres reliability and dependency hygiene.

Decisions:
- Keep splitting into small PRs; top-25 picks prioritized for early integration.

Next steps:
- Await instruction to apply these cherry-picks into a premerge/integration branch and run tests, or provide a runnable shell script for local execution.
