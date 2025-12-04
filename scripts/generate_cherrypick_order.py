#!/usr/bin/env python3
"""
Generate an ordered cherry-pick CSV from docs/diff_hku/unmerged_upstream_mapping.csv

Ordering rule: primary = category priority (safety-first), secondary = chronological by auth_date (oldest first).

Output: docs/diff_hku/cherry_pick_ordered.csv with columns:
  commit,auth_date,author,subject,category,priority_idx,git_cherry_pick_cmd

Usage:
  python scripts/generate_cherrypick_order.py

"""
import csv
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "diff_hku" / "unmerged_upstream_mapping.csv"
OUT = ROOT / "docs" / "diff_hku" / "cherry_pick_ordered.csv"


DEFAULT_PRIORITY_ORDER = [
    # Wave 0 - security and DB safety
    "security",
    "postgres",
    "storage",
    "ci",
    "tests",
    # workspace and data safety
    "workspace",
    "chunking",
    "ingestion",
    # embeddings / llm providers
    "embedding",
    "llm_cloud",
    "rerank",
    # docs and misc
    "json",
    "pdf",
    "docx",
    "katex",
    "dependabot",
    "webui",
    "misc",
    "docs",
    "other",
]


def build_priority_map(order_list):
    mapping = {}
    for idx, name in enumerate(order_list):
        if name not in mapping:
            mapping[name] = idx
    # unknown categories will be placed at end using high index
    return mapping


def parse_date(s: str):
    try:
        return datetime.fromisoformat(s)
    except Exception:
        # fallback: try parsing date-only
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            return datetime.min


def main():
    if not SRC.exists():
        print("Source mapping CSV not found at", SRC)
        return 1

    priority_map = build_priority_map(DEFAULT_PRIORITY_ORDER)

    rows = []
    with SRC.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            cat = (r.get("category") or "").strip() or "other"
            priority_idx = priority_map.get(cat, max(priority_map.values()) + 1)
            date_val = parse_date((r.get("auth_date") or "").strip())
            rows.append({
                "commit": r.get("commit"),
                "auth_date": r.get("auth_date"),
                "author": r.get("author"),
                "subject": r.get("subject"),
                "category": cat,
                "priority_idx": priority_idx,
                "date_val": date_val,
            })

    # Sort by priority_idx then date_val then commit
    rows.sort(key=lambda x: (x["priority_idx"], x["date_val"], x["commit"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "commit",
            "auth_date",
            "author",
            "subject",
            "category",
            "priority_idx",
            "git_cherry_pick_cmd",
        ])
        for r in rows:
            cmd = f"git cherry-pick {r['commit']}"
            writer.writerow([
                r["commit"],
                r["auth_date"],
                r["author"],
                r["subject"],
                r["category"],
                r["priority_idx"],
                cmd,
            ])

    print("Wrote ordered cherry-pick CSV to:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
