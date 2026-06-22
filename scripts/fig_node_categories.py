"""
Figure (Chapter 1): a single Mapper graph with nodes coloured by their
topological category (Peripheral / Bridge / Connector Hub / Provincial Hub),
using the exact same criteria as graph_metrics.node_category.

Usage:
    python scripts/fig_node_categories.py [FILM] [SUBJECT]

Defaults to an averaged graph that shows all four categories clearly.
Output: <FIGURES_DIR>/ch1_node_categories.png   (FIGURES_DIR set in src/config.py)
"""
import sys
import os
import json
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from graph_metrics import json_to_networkx, node_category, lcc_of  # noqa: E402
from config import FIGURES_DIR, PRIMARY_GRAPHS_DIR  # noqa: E402

GRAPHS_DIR = str(PRIMARY_GRAPHS_DIR)
OUT_PATH   = os.path.join(str(FIGURES_DIR), "ch1_node_categories.png")

# colour-blind-friendly palette, one colour per category
COLORS = {
    "Peripheral":     "#9ecae1",  # light blue  (most numerous, the bulk)
    "Bridge":         "#fdae6b",  # orange       (connectors outside the core)
    "Connector Hub":  "#e6550d",  # dark orange  (core + bridge)
    "Provincial Hub": "#74c476",  # green        (core, no bridging)
}
ORDER = ["Peripheral", "Bridge", "Connector Hub", "Provincial Hub"]


def load_nx(film, subject):
    fname = f"{film}_100_riso_{subject}.json"
    path  = os.path.join(GRAPHS_DIR, fname)
    with open(path) as f:
        G = json_to_networkx(json.load(f))
    return G, fname


def categorize(G):
    core = nx.core_number(G)
    bet  = nx.betweenness_centrality(G, normalized=True)
    cats = {n: node_category(core[n], bet[n]) for n in G.nodes()}
    return cats


def draw(G, cats, title, out_path):
    # plot the largest connected component for a clean, single-piece picture
    H = lcc_of(G)
    cats = {n: cats[n] for n in H.nodes()}

    sizes = np.array([H.nodes[n].get("size", 1) for n in H.nodes()], dtype=float)
    sizes = 60 + 240 * (sizes - sizes.min()) / (np.ptp(sizes) + 1e-9)

    pos = nx.spring_layout(H, seed=42, k=0.6 / np.sqrt(H.number_of_nodes()))

    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(H, pos, ax=ax, edge_color="#cfcfcf", width=0.8, alpha=0.7)
    for cat in ORDER:
        idx = [n for n in H.nodes() if cats[n] == cat]
        if not idx:
            continue
        nx.draw_networkx_nodes(
            H, pos, nodelist=idx,
            node_color=COLORS[cat],
            node_size=[sizes[list(H.nodes()).index(n)] for n in idx],
            edgecolors="#333333", linewidths=0.4, ax=ax,
        )

    counts = {c: sum(1 for v in cats.values() if v == c) for c in ORDER}
    legend = [Patch(facecolor=COLORS[c], edgecolor="#333333",
                    label=f"{c}  (n={counts[c]})") for c in ORDER]
    ax.legend(handles=legend, loc="upper left", frameon=True, fontsize=11)
    ax.set_axis_off()
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"saved -> {out_path}")
    print("category counts (LCC):", counts)


def main():
    film    = sys.argv[1] if len(sys.argv) > 1 else "Sintel"
    subject = sys.argv[2] if len(sys.argv) > 2 else "avg"
    G, fname = load_nx(film, subject)
    cats = categorize(G)
    full_counts = {c: sum(1 for v in cats.values() if v == c) for c in ORDER}
    print(f"{fname}: {G.number_of_nodes()} nodes, full-graph categories: {full_counts}")
    draw(G, cats, title=f"Grafo Mapper — {film} ({subject})", out_path=OUT_PATH)


if __name__ == "__main__":
    main()
