Actions:
- Parsed docs/diff_hku/unmerged_upstream_mapping.csv and created scripts/generate_cherrypick_order.py.
- Generated docs/diff_hku/cherry_pick_ordered.csv (747 commits ordered by category priority then date).
- Added docs/diff_hku/cherry_pick_ordered_summary.md with summary, counts, and quick-run commands.

Decisions:
- Category priority list chosen: security → postgres → storage → ci → tests → workspace → chunking → ingestion → embedding → llm_cloud → rerank → json → pdf → docx → katex → dependabot → webui → misc → docs → other.
- Sorting uses auth_date ascending for chronological ordering within each category.

Next steps:
- (Optional) Split the ordered list into waves (Wave 0..N) and produce per-wave scripts.
- (Optional) Attempt to apply top-N cherry-picks into a new integration branch and run tests to catch conflicts early.

Lessons/insights:
- The upstream delta is large but structured (many non-risky "misc" and dependabot updates); focusing on safety categories first reduces merge risk.
