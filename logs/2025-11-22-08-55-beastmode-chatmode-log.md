Actions: Updated .gitignore to include lightrag/api/webui/assets/; untracked existing files in current branch index.
Decisions: Added directory ignore at root and API-level .gitignore; stopped short of rewriting history until user confirmation.
Next steps: On user confirmation, will create a backup ref and purge the directory from repo history (prefer git-filter-repo; fallback to git filter-branch/BFG if unavailable), then advise force-push steps.
Lessons/insights: The assets directory contains many large built artifact files and vendor bundles that should not be stored in git history; rewriting history is required to truly remove them.
