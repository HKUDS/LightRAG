#!/usr/bin/env python3
"""
Split docs/diff_hku/cherry_pick_ordered.csv into per-wave CSVs under docs/diff_hku/waves/

Wave definitions (category groups):
  Wave 0: security, postgres, storage, ci          (DB safety & infra)
  Wave 1: tests, workspace, chunking, ingestion    (correctness & pipeline)
  Wave 2: embedding, llm_cloud, rerank             (providers)
  Wave 3: json, pdf, docx, katex, xlsx             (data formats)
  Wave 4: dependabot, webui, misc, docs, other     (low-risk churn)

Usage:
  python scripts/split_cherrypick_waves.py
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "diff_hku" / "cherry_pick_ordered.csv"
OUT_DIR = ROOT / "docs" / "diff_hku" / "waves"

WAVE_CATEGORIES = {
    0: {"security", "postgres", "storage", "ci"},
    1: {"tests", "workspace", "chunking", "ingestion"},
    2: {"embedding", "llm_cloud", "rerank"},
    3: {"json", "pdf", "docx", "katex", "xlsx"},
    4: {"dependabot", "webui", "misc", "docs", "other"},
}


def main():
    if not SRC.exists():
        print("Source CSV not found:", SRC)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build category → wave mapping
    cat_to_wave = {}
    for wave_id, cats in WAVE_CATEGORIES.items():
        for c in cats:
            cat_to_wave[c] = wave_id

    # Read all rows
    with SRC.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Bucket rows by wave
    wave_rows = {w: [] for w in WAVE_CATEGORIES}
    for r in rows:
        cat = r.get("category", "other").strip() or "other"
        w = cat_to_wave.get(cat, 4)  # default to Wave 4
        wave_rows[w].append(r)

    # Write per-wave CSVs
    for wave_id in sorted(WAVE_CATEGORIES.keys()):
        out_path = OUT_DIR / f"wave_{wave_id}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(wave_rows[wave_id])
        print(f"Wave {wave_id}: {len(wave_rows[wave_id])} commits → {out_path}")

    # Also write a shell script per wave for convenience
    for wave_id in sorted(WAVE_CATEGORIES.keys()):
        script_path = OUT_DIR / f"apply_wave_{wave_id}.sh"
        with script_path.open("w", encoding="utf-8") as fh:
            fh.write("#!/usr/bin/env bash\n")
            fh.write(f"# Auto-generated script to apply Wave {wave_id} commits\n")
            fh.write("set -e\n\n")
            for r in wave_rows[wave_id]:
                commit = r.get("commit", "")
                subject = r.get("subject", "").replace('"', '\\"')
                fh.write(f'echo "Cherry-picking {commit}: {subject}"\n')
                fh.write(f"git cherry-pick -x {commit}\n\n")
        script_path.chmod(0o755)
        print(f"  → shell script: {script_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
