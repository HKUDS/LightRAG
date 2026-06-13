#!/usr/bin/env python3
"""
Create benchmark evaluation dataset for LightRAG.

Sources:
  - HotpotQA (bridge, hard) → multi-hop questions   (50)
  - SQuAD v1.1              → single-hop questions   (50)

Output:
  lightrag/evaluation/benchmark_documents/  ← Wikipedia passages as .txt
  lightrag/evaluation/benchmark_dataset.json
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import itertools
from pathlib import Path

SEED = 42
N_MULTI  = 50
N_SINGLE = 50

OUT_DOCS    = Path("lightrag/evaluation/benchmark_documents")
OUT_DATASET = Path("lightrag/evaluation/benchmark_dataset.json")


# ── helpers ───────────────────────────────────────────────────────────────────

def slug(text: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^\w]+", "_", text).strip("_")
    return s[:maxlen].lower()


def token_f1(pred: str, ref: str) -> float:
    """Token-level F1 (case-insensitive, punctuation stripped)."""
    def tokens(s):
        return re.sub(r"[^\w\s]", "", s.lower()).split()
    pt, rt = set(tokens(pred)), set(tokens(ref))
    if not pt or not rt:
        return 1.0 if not pt and not rt else 0.0
    common = pt & rt
    p = len(common) / len(pt)
    r = len(common) / len(rt)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


# ── HotpotQA bridge (multi-hop) ───────────────────────────────────────────────

def load_hotpotqa_bridge(n: int) -> tuple[list[dict], dict[str, str]]:
    """Return (test_cases, {filename: text}) for n bridge questions."""
    from datasets import load_dataset

    print(f"  Loading HotpotQA distractor (bridge, hard) ...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation", streaming=True)

    collected: list[dict] = []
    docs: dict[str, str] = {}

    rng = random.Random(SEED)
    # stream until we have enough bridge/hard
    candidates: list = []
    for item in itertools.islice(ds, 5000):
        if item["type"] == "bridge" and item["level"] == "hard":
            candidates.append(item)
        if len(candidates) >= n * 4:
            break

    rng.shuffle(candidates)
    selected = candidates[:n]

    for item in selected:
        titles   = item["context"]["title"]
        sentences = item["context"]["sentences"]

        sup_titles = set(item["supporting_facts"]["title"])

        for title, sents in zip(titles, sentences):
            fname = f"hotpot_{slug(title)}.txt"
            if fname not in docs:
                docs[fname] = f"# {title}\n\n" + " ".join(sents)

        collected.append({
            "question":          item["question"],
            "ground_truth":      item["answer"],
            "type":              "multi_hop",
            "source":            "hotpotqa_bridge",
            "supporting_titles": sorted(sup_titles),
        })

    print(f"    → {len(collected)} multi-hop questions, {len(docs)} unique passages")
    return collected, docs


# ── SQuAD (single-hop) ────────────────────────────────────────────────────────

def load_squad_single(n: int) -> tuple[list[dict], dict[str, str]]:
    """Return (test_cases, {filename: text}) for n SQuAD questions."""
    from datasets import load_dataset

    print(f"  Loading SQuAD validation (single-hop) ...")
    ds = load_dataset("squad", split="validation", streaming=True)

    rng = random.Random(SEED + 1)
    candidates: list = []
    for item in itertools.islice(ds, 3000):
        # skip very short answers (likely confusing for RAG)
        answer_text = item["answers"]["text"][0] if item["answers"]["text"] else ""
        if len(answer_text.split()) >= 2:
            candidates.append(item)
        if len(candidates) >= n * 4:
            break

    rng.shuffle(candidates)
    selected = candidates[:n]

    docs: dict[str, str] = {}
    collected: list[dict] = []

    for item in selected:
        ctx_hash = hashlib.md5(item["context"].encode()).hexdigest()[:8]
        fname = f"squad_{slug(item['title'])}_{ctx_hash}.txt"
        if fname not in docs:
            docs[fname] = f"# {item['title']}\n\n{item['context']}"

        collected.append({
            "question":     item["question"],
            "ground_truth": item["answers"]["text"][0],
            "type":         "single_hop",
            "source":       "squad",
            "context_file": fname,
        })

    print(f"    → {len(collected)} single-hop questions, {len(docs)} unique passages")
    return collected, docs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DOCS.mkdir(parents=True, exist_ok=True)

    print("\n=== Benchmark Dataset Creation ===\n")

    multi_cases, multi_docs = load_hotpotqa_bridge(N_MULTI)
    single_cases, single_docs = load_squad_single(N_SINGLE)

    # merge docs
    all_docs = {**multi_docs, **single_docs}
    for fname, text in all_docs.items():
        (OUT_DOCS / fname).write_text(text, encoding="utf-8")

    all_cases = multi_cases + single_cases
    random.Random(SEED + 2).shuffle(all_cases)
    for i, tc in enumerate(all_cases, 1):
        tc["id"] = i

    dataset = {
        "test_cases": all_cases,
        "stats": {
            "total":      len(all_cases),
            "multi_hop":  sum(1 for t in all_cases if t["type"] == "multi_hop"),
            "single_hop": sum(1 for t in all_cases if t["type"] == "single_hop"),
            "documents":  len(all_docs),
            "sources":    {
                "hotpotqa_bridge": N_MULTI,
                "squad":           N_SINGLE,
            },
        },
    }
    OUT_DATASET.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\n  Documents written → {OUT_DOCS}/  ({len(all_docs)} files)")
    print(f"  Dataset written  → {OUT_DATASET}")
    print(f"\n  Stats: {dataset['stats']}")
    print("\nDone. Run eval_benchmark.py next.")


if __name__ == "__main__":
    main()
