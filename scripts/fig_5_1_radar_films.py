"""
fig_5_1_radar_films.py
======================
Figure 5.1 (thesis): "Affective signatures of three example films".

Three-panel radar chart (Sintel, Chatter, ToClaireFromSonny) over the seven
discrete emotion axes of the All50 set, on a shared radial scale. Uses the
SAME computation as 5_1_emotional_profile.py: for each film the bold line is
the mean over the film's timepoints and the shaded band its +/-1 temporal
standard deviation. All50 annotations are film-level, so no graphs are involved.

The three films are chosen to span the entropy ranking of Section 5.1:
Sintel (broad profile), Chatter (intermediate), ToClaireFromSonny (lowest
entropy, lopsided profile).

All fig_*.py scripts (this one included) write to FIGURES_DIR, defined in
src/config.py. Edit FIGURES_DIR there to point at your thesis image folder.

Output: <FIGURES_DIR>/radar_films.png  (300 dpi)
        <FIGURES_DIR>/radar_films.pdf  (vector)
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import FIGURES_DIR, DISCRETE_EMOTIONS
from data_loader import load_all50

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

FILMS_TO_PLOT = ["Sintel", "Chatter", "ToClaireFromSonny"]
DISPLAY_NAME = {
    "Sintel":            "Sintel",
    "Chatter":           "Chatter",
    "ToClaireFromSonny": "To Claire From Sonny",
}

EMO_LABELS = list(DISCRETE_EMOTIONS.keys())
N_EMO      = len(EMO_LABELS)
ANGLES     = np.linspace(0, 2 * np.pi, N_EMO, endpoint=False).tolist()
ANGLES_C   = ANGLES + ANGLES[:1]   # closed polygon

COLOR_MAIN = "#4C72B0"
OUT_DIR    = FIGURES_DIR


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────

def _setup_ax(ax, title, ylim):
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(ANGLES), EMO_LABELS, fontsize=12)
    ax.tick_params(axis="x", pad=8)
    ax.set_ylim(*ylim)
    yr = ylim[1] - ylim[0]
    yt = [ylim[0] + yr * f for f in (0.25, 0.5, 0.75)]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v:.2f}" for v in yt], fontsize=8, color="gray")
    ax.grid(color="gray", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.spines["polar"].set_visible(False)
    ax.set_title(title, size=14, fontweight="bold", pad=22)


def main():
    emo      = load_all50()
    disc_idx = list(DISCRETE_EMOTIONS.values())

    # temporal mean +/- std per film (identical to 5_1_emotional_profile.py)
    stats = {}
    for film in FILMS_TO_PLOT:
        arr = emo[film][:, disc_idx]               # (T, 7) film-level annotation
        stats[film] = dict(mean=arr.mean(axis=0), std=arr.std(axis=0))

    # shared radial scale across the three panels
    lo = min((stats[f]["mean"] - stats[f]["std"]).min() for f in FILMS_TO_PLOT)
    hi = max((stats[f]["mean"] + stats[f]["std"]).max() for f in FILMS_TO_PLOT)
    margin = (hi - lo) * 0.08
    ylim = (lo - margin, hi + margin)
    print(f"Shared radial scale: ({ylim[0]:.3f}, {ylim[1]:.3f})")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), subplot_kw=dict(polar=True))

    line = band = None
    for ax, film in zip(axes, FILMS_TO_PLOT):
        mean, std = stats[film]["mean"], stats[film]["std"]
        _setup_ax(ax, DISPLAY_NAME[film], ylim)

        upper = np.clip(mean + std, *ylim)
        lower = np.clip(mean - std, *ylim)
        band = ax.fill_between(
            ANGLES_C, list(lower) + [lower[0]], list(upper) + [upper[0]],
            color=COLOR_MAIN, alpha=0.15, label=r"$\pm 1$ SD (over time)",
        )

        m_c = list(mean) + [mean[0]]
        line, = ax.plot(ANGLES_C, m_c, color=COLOR_MAIN, linewidth=2.4,
                        label="Temporal mean")
        ax.fill(ANGLES_C, m_c, color=COLOR_MAIN, alpha=0.25)

    fig.legend(handles=[line, band], loc="lower center", ncol=2, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.86, bottom=0.10, wspace=0.45)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "radar_films.png"
    pdf_path = OUT_DIR / "radar_films.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved -> {png_path}")
    print(f"Saved -> {pdf_path}")


if __name__ == "__main__":
    main()
