"""
Figure (Chapter 4): a single Mapper graph coloured by time, i.e. each node is
shaded by the mean fMRI timepoint (TR index) of the timepoints it contains.
This is the canonical qualitative view of a Mapper graph: it shows how the
temporal trajectory of the scan folds into the shape of the graph, revealing
branches and recurrences.

Usage:
    python scripts/fig_graph_shape_time.py [FILM] [SUBJECT]

Output: <FIGURES_DIR>/fig_4_graph_shape_time.png   (FIGURES_DIR set in src/config.py)
"""
import sys
import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from graph_metrics import json_to_networkx, lcc_of  # noqa: E402
from config import FIGURES_DIR, PRIMARY_GRAPHS_DIR  # noqa: E402

GRAPHS_DIR = str(PRIMARY_GRAPHS_DIR)
OUT_PATH   = os.path.join(str(FIGURES_DIR), "fig_4_graph_shape_time.png")
SEED = 4


def load(film, subject):
    fname = f"{film}_100_riso_{subject}.json"
    with open(os.path.join(GRAPHS_DIR, fname)) as f:
        return json_to_networkx(json.load(f)), fname


def main():
    film    = sys.argv[1] if len(sys.argv) > 1 else "BigBuckBunny"
    subject = sys.argv[2] if len(sys.argv) > 2 else "sub-S11"
    G_full, fname = load(film, subject)
    G = lcc_of(G_full)  # largest connected component for a clean, single-piece shape
    nodes = list(G.nodes())
    print(f"{fname}: {G_full.number_of_nodes()} nodes total, "
          f"{len(nodes)} in LCC, {G.number_of_edges()} edges")

    # node colour = mean timepoint (TR) of its members; size = number of TRs
    time_val = np.array([np.mean(G.nodes[n]["members"]) for n in nodes])
    sizes    = np.array([G.nodes[n].get("size", 1) for n in nodes], dtype=float)
    node_sz  = 60 + 340 * (sizes - sizes.min()) / (np.ptp(sizes) + 1e-9)

    pos = nx.spring_layout(G, seed=SEED, k=1.1 / np.sqrt(len(nodes)), iterations=200)

    fig, ax = plt.subplots(figsize=(9, 8))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#c9c9c9", width=0.9, alpha=0.7)
    sc = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=time_val, cmap="viridis",
        node_size=node_sz, edgecolors="#333333", linewidths=0.4,
    )
    ax.set_axis_off()
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Time (mean TR index)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
