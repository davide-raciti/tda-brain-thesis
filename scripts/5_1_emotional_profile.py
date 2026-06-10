"""
5_1_emotional_profile.py
========================
Emotional characterization of the 14 film stimuli (thesis section 5.1).

Outputs
-------
results/5_1/
    entropy_ranking.csv         — per-film Shannon entropy (bits), sorted
    radar_<film>.png            — film-average radar chart (discrete emotions)
    alignment_matrix.csv        — inter-subject correlation matrix per film
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from config import FILMS, FILM_LENGTHS, DISCRETE_EMOTIONS
from data_loader import load_all50

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR = Path(__file__).resolve().parent.parent / 'results' / '5_1'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# RADAR CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────

EMO_LABELS = list(DISCRETE_EMOTIONS.keys())
N_EMO      = len(EMO_LABELS)
ANGLES     = np.linspace(0, 2 * np.pi, N_EMO, endpoint=False).tolist()
ANGLES_C   = ANGLES + ANGLES[:1]   # closed polygon

COLOR_MAIN = '#4C72B0'


def _setup_radar_ax(ax, title='', ylim=None):
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(ANGLES), EMO_LABELS, fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.45)
    ax.spines['polar'].set_visible(False)
    if title:
        ax.set_title(title, size=11, fontweight='bold', pad=16)


def _draw_polygon(ax, values, color, alpha_fill=0.25, lw=2, label=None):
    vals = list(values) + [values[0]]
    ax.plot(ANGLES_C, vals, color=color, linewidth=lw, label=label)
    ax.fill(ANGLES_C, vals, color=color, alpha=alpha_fill)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — ENTROPY RANKING
# ─────────────────────────────────────────────────────────────────────────────

def compute_entropy_ranking(emo_data):
    """
    Film-level emotional dynamism. For each of the 7 discrete emotions, build a
    10-bin histogram of its values over the film's TRs and take the Shannon
    entropy (bits) of that histogram; the film score is the mean entropy over the
    7 emotions. A film whose emotions sweep through many intensity levels over
    time scores high; one that stays in a narrow emotional band scores low.

    This is a measure of temporal variability, distinct from the node-level
    Emotional Complexity of Section 4.3 (which is the entropy of a single mean
    profile). The two answer different questions and are not the same metric.
    """
    rows     = []
    disc_idx = list(DISCRETE_EMOTIONS.values())
    n_bins   = 10

    for film in FILMS:
        arr  = emo_data[film][:, disc_idx]   # (T, 7)
        ents = []
        for j in range(arr.shape[1]):
            sig    = arr[:, j]
            lo, hi = float(sig.min()), float(sig.max())
            if hi > lo:
                hist, _ = np.histogram(sig, bins=n_bins, range=(lo, hi))
                p = hist[hist > 0] / hist.sum()
                ents.append(float(-np.sum(p * np.log2(p))))
            else:
                ents.append(0.0)
        rows.append({'Film': film, 'Entropy_bits': round(float(np.mean(ents)), 4)})

    df = pd.DataFrame(rows).sort_values('Entropy_bits', ascending=False).reset_index(drop=True)
    df.to_csv(OUT_DIR / 'entropy_ranking.csv', index=False)

    print('\nEntropy ranking (bits):')
    for _, row in df.iterrows():
        print(f"  {row['Film']:<25} {row['Entropy_bits']:.3f}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — RADAR CHARTS (film-average, discrete emotions)
# ─────────────────────────────────────────────────────────────────────────────

def make_radar_charts(emo_data):
    """
    Film-average radar chart for each film on the 7 discrete emotion axes.
    Mean ± 1 std across subjects shown as shaded band.
    """
    disc_idx = list(DISCRETE_EMOTIONS.values())

    # global ylim for comparability across films
    all_vals = np.vstack([emo_data[f][:, disc_idx] for f in FILMS])
    margin   = (all_vals.max() - all_vals.min()) * 0.05
    ylim     = (float(all_vals.min()) - margin, float(all_vals.max()) + margin)

    for film in FILMS:
        arr  = emo_data[film][:, disc_idx]   # (T, 7)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        upper = np.clip(mean + std, *ylim)
        lower = np.clip(mean - std, *ylim)

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        _setup_radar_ax(ax, title=film, ylim=ylim)

        up_c = list(upper) + [upper[0]]
        lo_c = list(lower) + [lower[0]]
        ax.fill_between(ANGLES_C, lo_c, up_c, color=COLOR_MAIN, alpha=0.15)
        _draw_polygon(ax, mean, COLOR_MAIN, label='Film mean')
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=8)

        fig.tight_layout()
        fig.savefig(OUT_DIR / f'radar_{film}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f'\n{len(FILMS)} radar charts saved -> {OUT_DIR}')


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — INTER-SUBJECT ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def compute_alignment(emo_data):
    """
    For each film, measure how strongly the 7 discrete emotion dimensions
    co-vary over time — a proxy for the internal coherence of the film's
    affective structure.

    Since All50 annotations are film-level (same for all subjects), true
    inter-subject variability cannot be measured here. Instead we compute
    the mean pairwise Pearson correlation between the 7 emotion timeseries,
    which captures how synchronised the emotional dimensions are within each
    film. Films with high mean correlation have a tightly coupled affective
    profile; low correlation signals emotionally diverse, multi-dimensional
    content.
    """
    disc_idx = list(DISCRETE_EMOTIONS.values())
    rows     = []

    for film in FILMS:
        T   = FILM_LENGTHS[film]
        arr = emo_data[film][:T, disc_idx]   # (T, 7)

        corr_matrix = np.corrcoef(arr.T)     # (7, 7)
        mask        = ~np.eye(N_EMO, dtype=bool)
        off_diag    = corr_matrix[mask]

        rows.append({
            'Film':     film,
            'Mean_r':   round(float(off_diag.mean()), 4),
            'Std_r':    round(float(off_diag.std()),  4),
        })

    df = pd.DataFrame(rows).sort_values('Mean_r', ascending=False).reset_index(drop=True)
    df.to_csv(OUT_DIR / 'emotion_coherence.csv', index=False)

    print('\nIntra-film emotion coherence (mean pairwise r across 7 dimensions):')
    for _, row in df.iterrows():
        print(f"  {row['Film']:<25} r = {row['Mean_r']:.3f} ± {row['Std_r']:.3f}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('Loading All50 emotion data...')
    emo_data = load_all50()
    print(f'  {len(emo_data)} films loaded.')

    print('\n[Task 1] Entropy ranking...')
    compute_entropy_ranking(emo_data)

    print('\n[Task 2] Radar charts...')
    make_radar_charts(emo_data)

    print('\n[Task 3] Inter-subject alignment...')
    compute_alignment(emo_data)

    print(f'\nAll outputs -> {OUT_DIR}')


if __name__ == '__main__':
    main()
