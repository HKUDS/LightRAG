"""Aggregate per-story ChronoQA FrameRAG results into an overall report."""
import glob
import json

E2RAG_REF = {
    "E2RAG (comb. extraction)": 7.1257,
    "LightRAG hybrid": 6.8804,
    "vanilla RAG": 6.6022,
}
TITLES = {"1": "A Study in Scarlet", "2": "The Hound of the Baskervilles",
          "5": "Les Miserables", "6": "The Phantom of the Opera",
          "7": "The Sign of the Four", "8": "The Wonderful Wizard of Oz",
          "9": "The Adventures of Sherlock Holmes"}

all_samples = []
per_story = {}
for f in sorted(glob.glob("story_results/story_*.json")):
    d = json.load(open(f, encoding="utf-8"))
    sid = str(d["story_id"])
    per_story[sid] = d["summary"]
    all_samples.extend(d["samples"])

if all_samples:
    overall = sum(s["avg_score"] for s in all_samples) / len(all_samples)
else:
    overall = 0.0

print("=" * 60)
print("FrameRAG ChronoQA Evaluation (LLM-judge Likert 1-10)")
print("=" * 60)
print(f"{'Story':<36}{'Avg':>7}{'N':>6}")
print("-" * 60)
for sid in sorted(per_story, key=lambda x: int(x)):
    s = per_story[sid]
    print(f"{sid}:{TITLES.get(sid,sid)[:30]:<33}{s['avg_score']:>7.3f}{s['n']:>6}")
print("-" * 60)
print(f"{'OVERALL (micro-avg over all Q)':<36}{overall:>7.3f}{len(all_samples):>6}")
if per_story:
    macro = sum(s["avg_score"] for s in per_story.values()) / len(per_story)
    print(f"{'OVERALL (macro-avg over stories)':<36}{macro:>7.3f}{len(per_story):>6}")
print("-" * 60)
print("Reference (E2RAG paper, 9 stories):")
for k, v in E2RAG_REF.items():
    print(f"  {k:<34}{v:>7.4f}")

# by facet
from collections import defaultdict
facet = defaultdict(list)
for s in all_samples:
    facet[s.get("category") or s.get("type") or "general"].append(s["avg_score"])
if facet:
    print("-" * 60)
    print("By reasoning facet:")
    for k in sorted(facet):
        v = facet[k]
        print(f"  {k:<34}{sum(v)/len(v):>7.3f}  (n={len(v)})")
