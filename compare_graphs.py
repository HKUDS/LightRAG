"""So sánh 3 graphs: mode=none (LLM), mode=full (frame-only), mode=full+paraphrase."""
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap, os

GRAPHS = {
    "mode=none\n(LLM baseline)":                "graphs/graph_none.graphml",
    "mode=full\n(frame-only)":                  "graphs/graph_full.graphml",
    "mode=full+paraphrase\n(LLM→frame→LLM)":    "graphs/graph_full_paraphrase.graphml",
}

TYPE_COLORS = {
    "project":       "#e74c3c",
    "method":        "#3498db",
    "concept":       "#2ecc71",
    "organization":  "#f39c12",
    "agent":         "#9b59b6",
    "technology":    "#1abc9c",
    "tool":          "#e67e22",
    "database":      "#16a085",
    "data":          "#2980b9",
    "metric":        "#27ae60",
}
DEFAULT_COLOR = "#95a5a6"

def type_color(t):
    return TYPE_COLORS.get(str(t).lower(), DEFAULT_COLOR)

def draw_graph(G, title, ax, max_label=16):
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, f"Graph trống\n0 nodes", ha="center", va="center",
                fontsize=14, transform=ax.transAxes, color="#e74c3c")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
        return

    if G.number_of_nodes() <= 5:
        pos = nx.spring_layout(G, k=3, seed=42)
    elif G.number_of_nodes() <= 15:
        pos = nx.spring_layout(G, k=2.5, seed=42, iterations=80)
    else:
        pos = nx.spring_layout(G, k=1.8, seed=42, iterations=100)

    degrees  = dict(G.degree())
    max_deg  = max(degrees.values()) if degrees else 1
    n_colors = [type_color(G.nodes[n].get("entity_type", "")) for n in G.nodes]
    n_sizes  = [500 + 1500 * (degrees[n] / max_deg) for n in G.nodes]

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, width=1.3,
                           edge_color="#555", arrows=True,
                           arrowsize=14, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=n_colors,
                           node_size=n_sizes, alpha=0.9)

    short = {n: "\n".join(textwrap.wrap(str(n)[:max_label*2], max_label)) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=short, ax=ax, font_size=6, font_weight="bold")

    edge_labels = {}
    for u, v, d in G.edges(data=True):
        kw = str(d.get("keywords", "")).split(",")[0].strip()[:18]
        if kw:
            edge_labels[(u, v)] = kw
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                 font_size=5, font_color="#c0392b",
                                 bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6))

    ax.set_title(f"{title}\n({G.number_of_nodes()} entities, {G.number_of_edges()} relations)",
                 fontsize=11, fontweight="bold", pad=8)
    ax.axis("off")


print("=" * 70)
print("  SO SANH KNOWLEDGE GRAPHS — 3 CHE DO EXTRACTION")
print("=" * 70)

graphs = {}
for label, path in GRAPHS.items():
    if os.path.exists(path):
        G = nx.read_graphml(path)
        graphs[label] = G
    else:
        graphs[label] = nx.Graph()
        print(f"  [WARN] Khong tim thay: {path}")

print(f"\n  {'Mode':<35} {'Entities':>9} {'Relations':>10}")
print(f"  {'-'*56}")
for label, G in graphs.items():
    lbl = label.replace("\n", " ")
    print(f"  {lbl:<35} {G.number_of_nodes():>9} {G.number_of_edges():>10}")

for label, G in graphs.items():
    lbl = label.replace("\n", " ")
    if G.number_of_nodes() == 0:
        continue
    print(f"\n  --- {lbl} ---")
    print(f"  {'Entity':<35} {'Type':<18} {'Deg':>4}")
    print(f"  {'-'*60}")
    for node, deg in sorted(G.degree(), key=lambda x: x[1], reverse=True):
        etype = G.nodes[node].get("entity_type", "?")
        desc  = G.nodes[node].get("description", "")[:70]
        print(f"  {str(node)[:35]:<35} {etype:<18} {deg:>4}")
        if desc:
            for line in textwrap.wrap(desc, 58):
                print(f"    {line}")

print(f"\n  --- RELATIONS ---")
for label, G in graphs.items():
    lbl = label.replace("\n", " ")
    if G.number_of_edges() == 0:
        continue
    print(f"\n  [{lbl}]")
    for u, v, d in G.edges(data=True):
        kw   = d.get("keywords", "")[:35]
        desc = d.get("description", "")[:60]
        print(f"    {str(u)[:22]:<22} --[{kw}]--> {str(v)[:22]}")
        if desc: print(f"      {desc}")

# Side-by-side visualization
fig, axes = plt.subplots(1, 3, figsize=(24, 10))
fig.suptitle("Knowledge Graph Comparison: 3 Extraction Modes\n"
             "(Node size = degree, Color = entity type)",
             fontsize=14, fontweight="bold", y=1.01)

for ax, (label, G) in zip(axes, graphs.items()):
    draw_graph(G, label, ax)

legend_patches = [mpatches.Patch(color=c, label=t.capitalize())
                  for t, c in TYPE_COLORS.items()]
legend_patches.append(mpatches.Patch(color=DEFAULT_COLOR, label="Other"))
fig.legend(handles=legend_patches, loc="lower center", ncol=6,
           fontsize=9, frameon=True, title="Entity Type",
           bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
out = "graphs/graph_3way_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\n  Saved: {out}")
plt.show()
