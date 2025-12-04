## Ordered cherry-pick CSV

Summary of generated file: `docs/diff_hku/cherry_pick_ordered.csv`

- Rows (commits): 747
- Ordering rule: primary = category priority (safety-first), secondary = chronological (oldest first by auth_date)
- Priority ordering (applied): security → postgres → storage → ci → tests → workspace → chunking → ingestion → embedding → llm_cloud → rerank → json → pdf → docx → katex → dependabot → webui → misc → docs → other

Top categories (by count):

- misc: 282
- postgres: 85
- webui: 60
- tests: 50
- embedding: 45
- llm_cloud: 45

How to use

1. Create a new integration branch to apply cherry-picks (do not work on main):

   ```bash
   git checkout -b premerge/integration-upstream
   ```

2. Apply cherry-picks in order (from top to bottom) using the generated `git cherry-pick <commit>` commands in the CSV.
   If a cherry-pick conflicts, stop and handle the conflict (or skip and note the commit).

3. Run the repository test suites and any integration tests after each wave (recommended). Use smaller batches (Wave 0 → Wave N) rather than all commits at once.

Quick-run (non-interactive example — use with caution):

```bash
# iterate each commit and attempt cherry-pick; stop on conflicts
while IFS=, read -r commit _; do
  if [ "$commit" = "commit" ]; then continue; fi
  echo "Applying $commit";
  git cherry-pick -x $commit || { echo "Conflict on $commit — stopping"; exit 1; }
done < docs/diff_hku/cherry_pick_ordered.csv
```

WARNING: applying a large number of cherry-picks to a long-lived branch may create complex conflicts and history issues. Prefer staged waves, resolve conflicts carefully, and add tests for each wave.
