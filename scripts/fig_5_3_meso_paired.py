"""
Figure (Chapter 5.3): paired Rest vs Film comparison of the three mesoscale
descriptors that reach significance, in a single three-panel figure. For each
subject the film value is the mean over that subject's films, paired with the
subject's single rest value; significance is a paired Wilcoxon signed-rank test.

Panels:
    max core number (k-core), Gini of betweenness, number of communities.

Output: <FIGURES_DIR>/fig_5_3_meso_paired.png   (FIGURES_DIR set in src/config.py)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import FIGURES_DIR, RESULTS_DIR  # noqa: E402

BASE = str(RESULTS_DIR)
OUT_PATH = os.path.join(str(FIGURES_DIR), "fig_5_3_meso_paired.png")

COL_REST = "#4C72B0"
COL_FILM = "#C44E52"

PANELS = [
    # (film_csv, rest_csv, column, nice label)
    (f"{BASE}/5_3a/betweenness_film.csv", f"{BASE}/5_3a/betweenness_rest.csv",
     "gini_betweenness", "Gini of betweenness"),
    (f"{BASE}/5_3a/kcore_film.csv", f"{BASE}/5_3a/kcore_rest.csv",
     "max_core", "Max core number"),
    (f"{BASE}/5_3b/community_film.csv", f"{BASE}/5_3b/community_rest.csv",
     "n_communities", "Number of communities"),
]


def paired_arrays(film_csv, rest_csv, col):
    film = pd.read_csv(film_csv).groupby("subject_id")[col].mean()
    rest = pd.read_csv(rest_csv).set_index("subject_id")[col]
    common = film.index.intersection(rest.index)
    return rest.loc[common].values, film.loc[common].values


def draw_panel(ax, rest, film, label):
    stat, p = wilcoxon(film, rest, alternative="two-sided")
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    # per-subject paired lines
    for r, f in zip(rest, film):
        ax.plot([0, 1], [r, f], color="#999999", lw=0.8, alpha=0.5, zorder=1)
    ax.scatter(np.zeros_like(rest), rest, color=COL_REST, s=28, zorder=3, label="Rest")
    ax.scatter(np.ones_like(film), film, color=COL_FILM, s=28, zorder=3, label="Film")
    # group means
    ax.plot([-0.18, 0.18], [rest.mean()] * 2, color=COL_REST, lw=2.5, zorder=4)
    ax.plot([0.82, 1.18], [film.mean()] * 2, color=COL_FILM, lw=2.5, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Rest", "Film"], fontsize=11)
    ax.set_xlim(-0.4, 1.4)
    ax.set_title(f"{label}\n$p = {p:.3f}$ {sig}", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", ls="--", alpha=0.3)
    print(f"  {label}: rest={rest.mean():.3f} film={film.mean():.3f} p={p:.4f} {sig} (n={len(rest)})")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, (fcsv, rcsv, col, label) in zip(axes, PANELS):
        rest, film = paired_arrays(fcsv, rcsv, col)
        draw_panel(ax, rest, film, label)
    axes[0].set_ylabel("per-subject value", fontsize=11)
    fig.suptitle("Mesoscale structure: Rest vs Film (paired, 100_riso)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
