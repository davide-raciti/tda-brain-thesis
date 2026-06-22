"""
Figure (Chapter 5.3, qualitative): the same Mapper graph shown three times with a
shared layout, coloured by the three mesoscale descriptors used in the analysis:
k-core number, betweenness centrality, and Louvain community. Showing one fixed
layout lets the reader see WHERE each property sits on the graph (the dense
core, the bridging nodes, the modules).

Usage:
    python scripts/fig_5_3_meso_ongraph.py [FILM] [SUBJECT]

Output: <FIGURES_DIR>/fig_5_3_meso_ongraph.png   (FIGURES_DIR set in src/config.py)
"""
import sys
import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import louvain_communities, modularity

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from graph_metrics import json_to_networkx, lcc_of  # noqa: E402
from config import FIGURES_DIR, PRIMARY_GRAPHS_DIR  # noqa: E402

GRAPHS_DIR = str(PRIMARY_GRAPHS_DIR)
OUT_PATH   = os.path.join(str(FIGURES_DIR), "fig_5_3_meso_ongraph.png")
SEED   = 4
N_RUNS = 20


def best_louvain(G):
    best_part, best_Q = None, -np.inf
    for seed in range(N_RUNS):
        part = louvain_communities(G, seed=seed)
        Q = modularity(G, part)
        if Q > best_Q:
            best_Q, best_part = Q, part
    return best_part, best_Q


def load(film, subject):
    fname = f"{film}_100_riso_{subject}.json"
    with open(os.path.join(GRAPHS_DIR, fname)) as f:
        return json_to_networkx(json.load(f)), fname


def main():
    film    = sys.argv[1] if len(sys.argv) > 1 else "Sintel"
    subject = sys.argv[2] if len(sys.argv) > 2 else "sub-S13"
    G_full, fname = load(film, subject)
    G = lcc_of(G_full)
    nodes = list(G.nodes())
    print(f"{fname}: {len(nodes)} nodes in LCC, {G.number_of_edges()} edges")

    pos   = nx.spring_layout(G, seed=SEED, k=1.1 / np.sqrt(len(nodes)), iterations=200)
    sizes = np.array([G.nodes[n].get("size", 1) for n in nodes], dtype=float)
    nsz   = 40 + 260 * (sizes - sizes.min()) / (np.ptp(sizes) + 1e-9)

    core = nx.core_number(G)
    bet  = nx.betweenness_centrality(G, normalized=True)
    part, Q = best_louvain(G)
    comm_id = {n: i for i, com in enumerate(part) for n in com}
    print(f"  k_max={max(core.values())}, n_communities={len(part)}, Q={Q:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))

    # --- panel 1: k-core ---
    cvals = np.array([core[n] for n in nodes])
    sc1 = nx.draw_networkx_nodes(G, pos, ax=axes[0], node_color=cvals, cmap="viridis",
                                 node_size=nsz, edgecolors="#333", linewidths=0.3)
    nx.draw_networkx_edges(G, pos, ax=axes[0], edge_color="#cfcfcf", width=0.7, alpha=0.6)
    axes[0].set_title("k-core number", fontsize=13)
    cb1 = fig.colorbar(sc1, ax=axes[0], fraction=0.045, pad=0.02)
    cb1.set_label("coreness $k$", fontsize=10)

    # --- panel 2: betweenness ---
    bvals = np.array([bet[n] for n in nodes])
    sc2 = nx.draw_networkx_nodes(G, pos, ax=axes[1], node_color=bvals, cmap="plasma",
                                 node_size=nsz, edgecolors="#333", linewidths=0.3)
    nx.draw_networkx_edges(G, pos, ax=axes[1], edge_color="#cfcfcf", width=0.7, alpha=0.6)
    axes[1].set_title("Betweenness centrality", fontsize=13)
    cb2 = fig.colorbar(sc2, ax=axes[1], fraction=0.045, pad=0.02)
    cb2.set_label(r"$C^{B}$", fontsize=10)

    # --- panel 3: communities ---
    mvals = np.array([comm_id[n] for n in nodes])
    nx.draw_networkx_nodes(G, pos, ax=axes[2], node_color=mvals, cmap="tab10",
                           node_size=nsz, edgecolors="#333", linewidths=0.3)
    nx.draw_networkx_edges(G, pos, ax=axes[2], edge_color="#cfcfcf", width=0.7, alpha=0.6)
    axes[2].set_title(f"Louvain communities ($Q={Q:.2f}$, {len(part)} comm.)", fontsize=13)

    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
