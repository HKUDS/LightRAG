#!/usr/bin/env python3
from pathlib import Path
import re

infile = Path("docs/diff_hku/unmerged_upstream_commits.txt")
out_csv = Path("docs/diff_hku/unmerged_upstream_mapping.csv")
summary_md = Path("docs/diff_hku/unmerged_upstream_mapping_summary.md")
if not infile.exists():
    raise SystemExit("ERROR: input file not found")
text = infile.read_text()
lines = [line.strip() for line in text.splitlines() if line.strip()]
patterns = [
    (
        "chunking",
        re.compile(
            r"chunk|chunking|chunk_token|chunk_size|chunk_top_k|ChunkTokenLimit|recursive split|top_n",
            re.I,
        ),
    ),
    (
        "ingestion",
        re.compile(
            r"duplicate|dedup|deduplication|insert|insertion|track_id|content dedup|ingest|ingestion",
            re.I,
        ),
    ),
    (
        "embedding",
        re.compile(
            r"embedding|embed|jina_embed|embedding provider|embedding_model|embedding_token",
            re.I,
        ),
    ),
    (
        "llm_cloud",
        re.compile(
            r"openai|ollama|azure|bedrock|gemini|llm|structured output|parsed", re.I
        ),
    ),
    (
        "postgres",
        re.compile(
            r"postgres|vchordrq|AGE|age|postgres_impl|set_config|row level security|rls",
            re.I,
        ),
    ),
    ("pdf", re.compile(r"pdf|pypdf|PyPDF2|pdf-decrypt|decryption", re.I)),
    ("docx", re.compile(r"docx|DOCX", re.I)),
    ("xlsx", re.compile(r"xlsx|excel|sheet\.max_column", re.I)),
    (
        "webui",
        re.compile(r"webui|lightrag_webui|vite|react|i18next|plugin-react-swc", re.I),
    ),
    (
        "tests",
        re.compile(
            r"\btest\b|e2e|pytest|workspace_isolation|integration|coverage|CI|github actions|actions|ruff",
            re.I,
        ),
    ),
    (
        "ci",
        re.compile(
            r"workflow|tests\.yml|docker-build|docker-compose|docker-build-push", re.I
        ),
    ),
    ("dependabot", re.compile(r"dependabot|bump", re.I)),
    ("json", re.compile(r"json|sanitiz|sanitize|sanitizer|UTF-8", re.I)),
    (
        "tools",
        re.compile(
            r"clean_llm_query_cache|migrate_llm|download_cache|clean_llm|lightrag-clean",
            re.I,
        ),
    ),
    (
        "workspace",
        re.compile(
            r"workspace|namespace|NamespaceLock|pipeline status|pipeline|isolation|auto-initialize",
            re.I,
        ),
    ),
    ("katex", re.compile(r"katex|KaTeX|mhchem|copy-tex", re.I)),
    (
        "security",
        re.compile(
            r"auth|token|secret|passlib|bcrypt|security|super_admin|has_tenant_access|membership",
            re.I,
        ),
    ),
    (
        "storage",
        re.compile(
            r"faiss|milvus|qdrant|redis|neo4j|memgraph|nano_vector|mongo|vector_db|vchordrq",
            re.I,
        ),
    ),
    ("rerank", re.compile(r"rerank|cohere|reranker|rerank_chunking", re.I)),
    (
        "docs",
        re.compile(
            r"Readme|README|Documentation|docs/|FrontendBuildGuide|OfflineDeployment|UV_LOCK_GUIDE|evaluation",
            re.I,
        ),
    ),
    ("misc", re.compile(r".*", re.I)),
]
rows = []
counts = {}
for line in lines:
    parts = line.split(" ", 3)
    if len(parts) >= 4:
        h, date, author, subject = parts
    elif len(parts) == 3:
        h, date, author = parts
        subject = ""
    else:
        h = parts[0]
        date = ""
        author = ""
        subject = ""
    assigned = None
    for name, regex in patterns[:-1]:
        if regex.search(line):
            assigned = name
            break
    if not assigned:
        assigned = "misc"
    rows.append((h, date, author, subject, assigned))
    counts[assigned] = counts.get(assigned, 0) + 1
# write csv
with out_csv.open("w") as fh:
    fh.write("commit,auth_date,author,subject,category\n")
    for h, date, author, subject, cat in rows:
        subject = subject.replace('"', '""')
        fh.write(f'{h},{date},{author},"{subject}",{cat}\n')
# write summary
with summary_md.open("w") as fh:
    fh.write("# Upstream commit mapping summary\n\n")
    fh.write(f"- total upstream commits mapped: {len(lines)}\n\n")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        fh.write(f"- {k}: {v} commits\n")
print("Done: wrote mapping CSV and summary")
