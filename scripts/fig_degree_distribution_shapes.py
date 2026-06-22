"""
Figure (Chapter 2, conceptual): the four candidate degree-distribution shapes
compared on a single log-log axis -- Poisson, pure power law, truncated power
law, exponential. Purely analytical (no data), to illustrate tail behaviour.

Output: <FIGURES_DIR>/fig_2_degree_shapes.png   (FIGURES_DIR set in src/config.py)
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import FIGURES_DIR  # noqa: E402

OUT_PATH = os.path.join(str(FIGURES_DIR), "fig_2_degree_shapes.png")

# --- shared support and parameters ------------------------------------------
K_FULL = np.arange(1, 2001)          # normalise over a long range to capture tail mass
K_PLOT = np.arange(1, 121)           # show this window
ALPHA  = 2.4                         # heavy-tail exponent (ties to empirical alpha ~ 2.41)
LAM_T  = 0.015                       # truncated power-law cutoff rate (gentle cutoff)
LAM_E  = 0.22                        # exponential rate (fast decay, light tail)
Z      = 8.0                         # Poisson mean

COL = {
    "Poisson":             "#6A51A3",  # purple
    "Power law":           "#C44E52",  # crimson
    "Truncated power law": "#E08214",  # orange
    "Exponential":         "#3182BD",  # blue
}


def normalise(p):
    return p / p.sum()


def pmfs(k):
    poisson   = np.exp(-Z + k * np.log(Z) - gammaln(k + 1))
    power     = k.astype(float) ** (-ALPHA)
    truncated = k.astype(float) ** (-ALPHA) * np.exp(-LAM_T * k)
    exponent  = np.exp(-LAM_E * k)
    return {
        "Poisson":             normalise(poisson),
        "Power law":           normalise(power),
        "Truncated power law": normalise(truncated),
        "Exponential":         normalise(exponent),
    }


def ccdf(p):
    """Complementary CDF  P(K >= k)  from a normalised PMF on K_FULL."""
    return np.cumsum(p[::-1])[::-1]


def main():
    full = pmfs(K_FULL)              # normalisation constants from the long range
    surv = {lbl: ccdf(p) for lbl, p in full.items()}
    sl   = K_PLOT - 1               # indices into the full arrays

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    styles = {
        "Poisson":             dict(ls="-",  lw=2.2),
        "Power law":           dict(ls="-",  lw=2.2),
        "Truncated power law": dict(ls="--", lw=2.2),
        "Exponential":         dict(ls="-.", lw=2.2),
    }
    for label in ["Power law", "Truncated power law", "Exponential", "Poisson"]:
        ax.plot(K_PLOT, surv[label][sl], color=COL[label], label=label, **styles[label])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 120)
    ax.set_ylim(1e-7, 1.2)
    ax.set_xlabel("Degree $k$", fontsize=12)
    ax.set_ylabel(r"$P(K \geq k)$", fontsize=12)
    ax.set_title("Candidate degree-distribution shapes", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, frameon=True)
    ax.grid(which="both", ls="--", alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
