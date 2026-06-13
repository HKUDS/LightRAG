from datasets import load_dataset

print("Loading dataset (non-streaming)...")
ds = load_dataset("TommyChien/UltraDomain", split="train")
print(f"Total rows: {len(ds)}")
print("Features:", ds.features)

from collections import Counter
labels = Counter(ds["label"])
print("Label distribution:", dict(labels))

cs_rows = [r for r in ds if r.get("label") == "cs"]
print(f"\nCS rows: {len(cs_rows)}")

unique_ctx = {}
for r in cs_rows:
    cid = r.get("context_id", "")
    if cid and cid not in unique_ctx:
        unique_ctx[cid] = r.get("context", "")

print(f"Unique CS contexts: {len(unique_ctx)}")
total_chars = sum(len(c) for c in unique_ctx.values())
print(f"Total chars: {total_chars:,} (~{total_chars//4:,} tokens)")
if unique_ctx:
    sample = list(unique_ctx.values())[0]
    print(f"\nFirst context preview:\n{sample[:400]}")
