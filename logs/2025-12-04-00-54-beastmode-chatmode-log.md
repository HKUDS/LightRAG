Actions:
- Enumerated commits present only in upstream/main and created a grouped audit of features/fixes not merged locally.
- Wrote docs/diff_hku/unmerged_upstream.md (summary) and docs/diff_hku/unmerged_upstream_commits.txt (raw commit list) with 746 commits.

Decisions/assumptions:
- Focused the summary on substantive features, storage, LLM/embedding changes, chunking, docs, tooling, and tests. Dependency bumps were not enumerated individually in the summary but are included in raw commits.

Next steps:
- Optionally cherry-pick or rebase high-priority upstream fixes into the local branch (recommend chunking, embedding, doc parsing, Postgres/vchordrq and RLS compatibility fixes first).
- Add CI jobs in this repo to run the newly added upstream tests and candidate merge validation.

File outputs:
- docs/diff_hku/unmerged_upstream.md
- docs/diff_hku/unmerged_upstream_commits.txt
