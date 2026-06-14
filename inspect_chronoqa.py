"""Quick script to download ChronoQA and inspect its structure.

Run:
    pip install datasets
    python inspect_chronoqa.py
"""
from datasets import load_dataset

print("Downloading zy113/ChronoQA ...")
ds = load_dataset("zy113/ChronoQA")

print("\n=== Splits ===")
print(ds)

split = list(ds.keys())[0]
data = ds[split]

print(f"\n=== Features (split='{split}') ===")
for name, feat in data.features.items():
    print(f"  {name:20s} : {feat}")

print(f"\n=== Total rows: {len(data)} ===")

print("\n=== Sample [0] — all fields ===")
s = data[0]
for k, v in s.items():
    val = v
    if isinstance(val, str) and len(val) > 300:
        val = val[:300] + "…"
    elif isinstance(val, list) and len(val) > 3:
        val = val[:3] + [f"... ({len(v)} total)"]
    print(f"  [{k}]")
    print(f"    {val}")

print("\n=== Story distribution ===")
from collections import Counter
story_col = "story_id" if "story_id" in data.features else None
if story_col:
    counts = Counter(data[story_col])
    for sid, n in sorted(counts.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
        title_col = "story_title" if "story_title" in data.features else None
        title = ""
        if title_col:
            titles = [r["story_title"] for r in data if r["story_id"] == sid]
            title = titles[0] if titles else ""
        print(f"  story {sid:>2} | {n:>3} questions | {title}")

print("\n=== Category distribution ===")
cat_col = "category" if "category" in data.features else "facet" if "facet" in data.features else None
if cat_col:
    counts = Counter(data[cat_col])
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<45} {n:>4} samples")

print("\n=== Passages field detail (sample[0]) ===")
passages = s.get("passages", [])
print(f"  Number of passages: {len(passages)}")
if passages:
    p = passages[0]
    print(f"  Passage[0] keys: {list(p.keys()) if isinstance(p, dict) else type(p)}")
    if isinstance(p, dict):
        for pk, pv in p.items():
            pv_str = str(pv)
            print(f"    {pk}: {pv_str[:200]}")
