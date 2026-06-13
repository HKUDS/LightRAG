"""So sánh graph mode=full vs mode=none với entity/edge highlights."""
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap, os

GRAPH_FULL = "graphs/graph_full.graphml"
GRAPH_NONE = "graphs/graph_none.graphml"
OUT_DIR    = "graphs"

# Màu theo entity_type
TYPE_COLORS = {
    "project":      "#e74c3c",
    "method":       "#3498db",
    "concept":      "#2ecc71",
    "organization": "#f39c12",
    "agent":        "#9b59b6",
    "technology":   "#1abc9c",
    "tool":         "#e67e22",
    "database":     "#16a085",
}
DEFAULT_COLOR = "#95a5a6"


def type_color(ntype: str) -> str:
    return TYPE_COLORS.get(str(ntype).lower(), DEFAULT_COLOR)


def draw_graph(G: nx.Graph, title: str, ax, max_label_len=18):
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Graph trống\n(0 nodes)", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")
        return

    # Layout
    if G.number_of_nodes() <= 5:
        pos = nx.spring_layout(G, k=3, seed=42)
    else:
        pos = nx.spring_layout(G, k=2.5, seed=42, iterations=80)

    # Node colors & sizes by degree
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    node_colors = [type_color(G.nodes[n].get("entity_type", "")) for n in G.nodes]
    node_sizes  = [400 + 1200 * (degrees[n] / max_deg) for n in G.nodes]

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.35, width=1.2,
                           edge_color="#555555", arrows=True,
                           arrowsize=12, connectionstyle="arc3,rad=0.1")

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.9)

    # Labels
    short_labels = {n: "\n".join(textwrap.wrap(str(n)[:max_label_len*2], max_label_len))
                    for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=short_labels, ax=ax,
                            font_size=6.5, font_weight="bold")

    # Edge keywords
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        kw = str(d.get("keywords", "")).split(",")[0].strip()[:20]
        if kw:
            edge_labels[(u, v)] = kw
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                 font_size=5.5, font_color="#c0392b",
                                 bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6))

    ax.set_title(f"{title}\n({G.number_of_nodes()} entities, {G.number_of_edges()} relations)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.axis("off")


def print_entity_table(G: nx.Graph, mode: str):
    print(f"\n{'='*65}")
    print(f"  ENTITIES — mode={mode.upper()}")
    print(f"{'='*65}")
    print(f"  {'Entity':<35} {'Type':<14} {'Degree':>6}")
    print(f"  {'-'*57}")
    for node, deg in sorted(G.degree(), key=lambda x: x[1], reverse=True):
        etype = G.nodes[node].get("entity_type", "?")
        desc  = G.nodes[node].get("description", "")[:80]
        print(f"  {str(node)[:35]:<35} {etype:<14} {deg:>6}")
        if desc:
            for line in textwrap.wrap(desc, 55):
                print(f"    {line}")

    print(f"\n  RELATIONS — mode={mode.upper()}")
    print(f"  {'-'*57}")
    for u, v, d in G.edges(data=True):
        kw   = d.get("keywords", "")[:40]
        desc = d.get("description", "")[:70]
        print(f"  {str(u)[:22]:<22} --[{kw[:25]}]--> {str(v)[:20]}")
        if desc:
            print(f"    {desc}")


# ── Load graphs ─────────────────────────────────────────────
G_full = nx.read_graphml(GRAPH_FULL) if os.path.exists(GRAPH_FULL) else nx.Graph()
G_none = nx.read_graphml(GRAPH_NONE) if os.path.exists(GRAPH_NONE) else nx.Graph()

# ── Print tables ────────────────────────────────────────────
print_entity_table(G_full, "full")
print_entity_table(G_none, "none")

# ── Summary comparison ──────────────────────────────────────
print(f"\n{'='*65}")
print(f"  TONG HOP SO SANH")
print(f"{'='*65}")
print(f"  {'':30} {'Frame-full':>12} {'LLM-none':>12}")
print(f"  {'-'*56}")
print(f"  {'Entities (nodes)':<30} {G_full.number_of_nodes():>12} {G_none.number_of_nodes():>12}")
print(f"  {'Relations (edges)':<30} {G_full.number_of_edges():>12} {G_none.number_of_edges():>12}")
ftypes = {G_full.nodes[n].get("entity_type","?") for n in G_full.nodes}
ntypes = {G_none.nodes[n].get("entity_type","?") for n in G_none.nodes}
print(f"  {'Entity types':<30} {len(ftypes):>12} {len(ntypes):>12}")
print(f"  {'Type names':<30} {str(sorted(ftypes))[:25]:>12}")
print(f"  {'':30} {str(sorted(ntypes))[:25]:>12}")

# ── Draw side-by-side ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle("Knowledge Graph Comparison\nFrame-Semantic (mode=full) vs LLM Baseline (mode=none)",
             fontsize=14, fontweight="bold", y=1.01)

draw_graph(G_full, "Frame-Semantic Extraction (mode=full)", axes[0])
draw_graph(G_none, "LLM Baseline (mode=none)",              axes[1])

# Legend
legend_patches = [mpatches.Patch(color=c, label=t.capitalize())
                  for t, c in TYPE_COLORS.items()]
legend_patches.append(mpatches.Patch(color=DEFAULT_COLOR, label="Other"))
fig.legend(handles=legend_patches, loc="lower center", ncol=5,
           fontsize=9, frameon=True, title="Entity Type",
           bbox_to_anchor=(0.5, -0.03))

plt.tight_layout()
out = f"{OUT_DIR}/graph_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\n  Saved: {out}")
plt.show()

# ── Draw mode=none alone (larger, more detail) ──────────────
fig2, ax2 = plt.subplots(figsize=(18, 13))
draw_graph(G_none, "LightRAG Knowledge Graph — LLM Baseline (mode=none)\nNode size = degree, color = entity type", ax2, max_label_len=14)
fig2.legend(handles=legend_patches, loc="lower center", ncol=5,
            fontsize=9, frameon=True, title="Entity Type",
            bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
out2 = f"{OUT_DIR}/graph_none_detail.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"  Saved: {out2}")
plt.show()
