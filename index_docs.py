"""Index sample documents và in chi tiết từng bước."""
import httpx, time, json, sys

API = "http://localhost:9621"
DOCS_DIR = "lightrag/evaluation/sample_documents"
MODE = sys.argv[1] if len(sys.argv) > 1 else "full"

from pathlib import Path
files = sorted(Path(DOCS_DIR).glob("*.md"))
files = [f for f in files if f.name.lower() != "readme.md"]

print(f"\n{'='*60}")
print(f"  INDEX MODE={MODE.upper()} — {len(files)} files")
print(f"{'='*60}")

with httpx.Client(timeout=30) as c:
    for f in files:
        text = f.read_text(encoding="utf-8")
        r = c.post(f"{API}/documents/text", json={"text": text, "file_path": f.name})
        print(f"  [{'OK' if r.status_code==200 else 'ERR'}] {f.name} ({len(text)} chars)")

print("\nPolling indexing status...")
with httpx.Client(timeout=15) as c:
    for attempt in range(120):
        time.sleep(5)
        data = c.get(f"{API}/documents").json()
        raw = data.get("statuses", data)
        statuses = {k: len(v) for k, v in raw.items() if isinstance(v, list)} if any(isinstance(v,list) for v in raw.values()) else raw
        print(f"  [{attempt*5:3d}s] {statuses}")
        pending = statuses.get("pending", 0) + statuses.get("processing", 0)
        failed  = statuses.get("failed", 0)
        done    = statuses.get("processed", statuses.get("success", 0))
        if pending == 0:
            if failed > 0 and done == 0:
                print("  ERROR: All documents FAILED")
                sys.exit(1)
            print(f"\n  Done: {done} processed, {failed} failed")
            break

# In entities và relations từ graph
import networkx as nx
from pathlib import Path
gpath = Path("rag_storage/graph_chunk_entity_relation.graphml")
if gpath.exists():
    G = nx.read_graphml(gpath)
    print(f"\n{'='*60}")
    print(f"  GRAPH STATS (mode={MODE})")
    print(f"{'='*60}")
    print(f"  Nodes (entities): {G.number_of_nodes()}")
    print(f"  Edges (relations): {G.number_of_edges()}")
    print(f"\n  Top 10 entities (by degree):")
    by_degree = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]
    for node, deg in by_degree:
        data = G.nodes[node]
        etype = data.get("entity_type", "?")
        desc  = data.get("description", "")[:60]
        print(f"    [{deg:2d}] {node[:35]:<35} type={etype}")
        if desc:
            print(f"         desc: {desc}...")
    print(f"\n  Sample edges (relations):")
    for u, v, d in list(G.edges(data=True))[:10]:
        kw   = d.get("keywords", "")[:30]
        desc = d.get("description", "")[:50]
        print(f"    {u[:20]:<20} --[{kw}]--> {v[:20]}")
        if desc: print(f"      {desc}...")
