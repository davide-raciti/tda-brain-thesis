"""
Figure (Chapter 5.5): two aggregated Graph-Portrait-Divergence (JSD) matrices
that visualise the within-subject vs within-film fingerprint.

From the 420 cached portrait matrices we form the full pairwise JSD, then
aggregate it two ways, each using the SAME shared colour scale:

  S[i,j]  (subjects, 30x30) = mean over film pairs a != b of
          JSD( graph[subject_i, film_a],  graph[subject_j, film_b] )
          -> diagonal = within-subject JSD

  F[a,b]  (films, 14x14)    = mean over subject pairs i != j of
          JSD( graph[subject_i, film_a],  graph[subject_j, film_b] )
          -> diagonal = within-film JSD

A more pronounced diagonal means a stronger fingerprint. The film diagonal
stands out more than the subject diagonal: the stimulus drives topology more
than individual identity.

Output: <FIGURES_DIR>/fig_5_5_fingerprint_matrices.png   (FIGURES_DIR set in src/config.py)
"""
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import FILMS, SUBJECT_IDS, FIGURES_DIR, RESULTS_DIR  # noqa: E402

CACHE = RESULTS_DIR / "5_5" / "portraits_cache.pkl"
OUT_PATH = str(FIGURES_DIR / "fig_5_5_fingerprint_matrices.png")


def jsd(p, q):
    """Jensen-Shannon divergence (base 2) between two normalised vectors."""
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    with open(CACHE, "rb") as f:
        portraits = pickle.load(f)
    print(f"  {len(portraits)} portraits loaded")

    subjects = [s for s in SUBJECT_IDS if any((s, f) in portraits for f in FILMS)]
    films    = list(FILMS)
    keys     = [(s, f) for s in subjects for f in films if (s, f) in portraits]
    kidx     = {k: i for i, k in enumerate(keys)}
    nK = len(keys)

    # pad every portrait to a common shape and flatten + normalise
    maxK = max(B.shape[0] for B in portraits.values())
    maxJ = max(B.shape[1] for B in portraits.values())
    V = np.zeros((nK, maxK * maxJ))
    for k, i in kidx.items():
        B = portraits[k]
        P = np.zeros((maxK, maxJ)); P[:B.shape[0], :B.shape[1]] = B
        v = P.flatten(); s = v.sum()
        V[i] = v / s if s > 0 else v

    # full pairwise JSD
    print(f"  computing {nK}x{nK} JSD matrix ...")
    D = np.zeros((nK, nK))
    for i in range(nK):
        for j in range(i + 1, nK):
            d = jsd(V[i], V[j])
            D[i, j] = D[j, i] = d
        if i % 60 == 0:
            print(f"    row {i}/{nK}")

    si = {s: i for i, s in enumerate(subjects)}
    fi = {f: i for i, f in enumerate(films)}

    # subject x subject: average over film pairs a != b
    S = np.full((len(subjects), len(subjects)), np.nan)
    for s1 in subjects:
        for s2 in subjects:
            vals = [D[kidx[(s1, a)], kidx[(s2, b)]]
                    for a in films for b in films
                    if a != b and (s1, a) in kidx and (s2, b) in kidx]
            if vals:
                S[si[s1], si[s2]] = np.mean(vals)

    # film x film: average over subject pairs i != j
    F = np.full((len(films), len(films)), np.nan)
    for f1 in films:
        for f2 in films:
            vals = [D[kidx[(s1, f1)], kidx[(s2, f2)]]
                    for s1 in subjects for s2 in subjects
                    if s1 != s2 and (s1, f1) in kidx and (s2, f2) in kidx]
            if vals:
                F[fi[f1], fi[f2]] = np.mean(vals)

    print(f"  within-subject (S diag mean) = {np.nanmean(np.diag(S)):.4f}  (expected 0.355)")
    print(f"  within-film    (F diag mean) = {np.nanmean(np.diag(F)):.4f}  (expected 0.331)")

    # shared colour scale for fair comparison
    allv = np.concatenate([S[~np.isnan(S)], F[~np.isnan(F)]])
    vmin, vmax = np.percentile(allv, 2), np.percentile(allv, 98)

    def outline_diagonal(ax, n):
        for i in range(n):
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="#d62728", linewidth=1.1, zorder=5))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6),
                             gridspec_kw={"width_ratios": [30, 14]})

    im0 = axes[0].imshow(S, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Subject × Subject\n(diagonal = within-subject, "
                      f"mean {np.nanmean(np.diag(S)):.3f})", fontsize=12)
    axes[0].set_xticks(range(len(subjects)))
    axes[0].set_yticks(range(len(subjects)))
    axes[0].set_xticklabels([s.replace("sub-", "") for s in subjects], rotation=90, fontsize=6)
    axes[0].set_yticklabels([s.replace("sub-", "") for s in subjects], fontsize=6)
    outline_diagonal(axes[0], len(subjects))

    im1 = axes[1].imshow(F, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Film × Film\n(diagonal = within-film, "
                      f"mean {np.nanmean(np.diag(F)):.3f})", fontsize=12)
    axes[1].set_xticks(range(len(films)))
    axes[1].set_yticks(range(len(films)))
    axes[1].set_xticklabels([f[:10] for f in films], rotation=90, fontsize=7)
    axes[1].set_yticklabels([f[:10] for f in films], fontsize=7)
    outline_diagonal(axes[1], len(films))

    cb = fig.colorbar(im1, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Portrait divergence (JSD)", fontsize=11)
    fig.suptitle("Graph Portrait Divergence: within-subject vs within-film (100_riso)",
                 fontsize=13, fontweight="bold")
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
