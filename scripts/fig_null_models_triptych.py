"""
Figure (Chapter 2, null models): a single real Mapper graph shown next to an
Erdos-Renyi and a Barabasi-Albert null built on the SAME (n, density / n_edges),
so the visual comparison is fair. Illustrative only: the quantitative test is
the null boxplot in results/5_2b/null_boxplot.png.

Usage:
    python scripts/fig_null_models_triptych.py [FILM] [SUBJECT]

Output: <FIGURES_DIR>/fig_2_null_models.png   (FIGURES_DIR set in src/config.py)
"""
import sys
import os
import json
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from graph_metrics import json_to_networkx, make_er, make_ba  # noqa: E402
from config import FIGURES_DIR, PRIMARY_GRAPHS_DIR  # noqa: E402

GRAPHS_DIR = str(PRIMARY_GRAPHS_DIR)
OUT_PATH   = os.path.join(str(FIGURES_DIR), "fig_2_null_models.png")

# same palette as the null boxplot (5_2b)
COLOR_REAL = "#4C72B0"   # blue
COLOR_ER   = "#C44E52"   # red
COLOR_BA   = "#2CA02C"   # green

SEED = 7


def load_real(film, subject):
    fname = f"{film}_100_riso_{subject}.json"
    with open(os.path.join(GRAPHS_DIR, fname)) as f:
        return json_to_networkx(json.load(f)), fname


def lcc_ratio(G):
    if G.number_of_nodes() == 0:
        return 0.0
    return len(max(nx.connected_components(G), key=len)) / G.number_of_nodes()


def draw(ax, G, color, title, dmax):
    pos = nx.spring_layout(G, seed=SEED, k=0.9 / np.sqrt(max(G.number_of_nodes(), 1)))
    # node size proportional to degree, on a shared scale across all three panels,
    # so the hub structure (BA) vs homogeneity (ER) is directly readable
    deg   = dict(G.degree())
    sizes = [25 + 230 * (deg[n] / dmax) for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cfcfcf", width=0.7, alpha=0.7)
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=color, node_size=sizes,
        edgecolors="#333333", linewidths=0.3,
    )
    ncc = nx.number_connected_components(G)
    ax.set_title(f"{title}\n(LCC {lcc_ratio(G):.2f}, {ncc} comp., grado max {max(deg.values())})",
                 fontsize=12)
    ax.set_axis_off()


def main():
    film    = sys.argv[1] if len(sys.argv) > 1 else "Payload"
    subject = sys.argv[2] if len(sys.argv) > 2 else "sub-S01"

    G, fname = load_real(film, subject)
    n = G.number_of_nodes()
    E = G.number_of_edges()
    density = nx.density(G)
    print(f"{fname}: n={n}, edges={E}, density={density:.4f}, LCC={lcc_ratio(G):.3f}")

    ER = make_er(n, density, seed=SEED)
    BA = make_ba(n, E, seed=SEED)

    dmax = max(max(dict(g.degree()).values()) for g in (G, ER, BA))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    draw(axes[0], G,  COLOR_REAL, "Grafo Mapper empirico", dmax)
    draw(axes[1], ER, COLOR_ER,   "Null Erdős–Rényi", dmax)
    draw(axes[2], BA, COLOR_BA,   "Null Barabási–Albert", dmax)
    fig.suptitle(
        f"Grafo empirico vs null model con stessi $(n, p)$  —  $n={n}$, densità ${density:.3f}$",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
