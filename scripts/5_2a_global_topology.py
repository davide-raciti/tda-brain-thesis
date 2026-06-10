"""
5_2a_global_topology.py
=======================
Macro-scale analysis: global topology distributions and Rest vs Film
comparison (thesis section 5.2, first half).

Pipeline
--------
1. Compute topology metrics for all 420 film graphs + 30 Rest graphs
   from JSON, save metrics cache to CSV.
2. Descriptive statistics + boxplot of the four macro-metrics across
   the 420 film networks.
3. Paired Wilcoxon signed-rank test (Film mean vs Rest, per subject)
   for each metric.

Outputs  results/5_2a/
    metrics_film.csv          topology metrics for all 420 film graphs
    metrics_rest.csv          topology metrics for all 30 Rest graphs
    distributions.png         boxplot of 4 metrics across 420 networks
    wilcoxon_results.csv      Wilcoxon test results Film vs Rest
    paired_comparison.png     paired boxplot Film vs Rest per metric
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wilcoxon

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from config import FILMS, SUBJECT_IDS, GRAPHS_DIR
from graph_metrics import load_graph, compute_topology

OUT_DIR = Path(__file__).resolve().parent.parent / 'results' / '5_2a'
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = {
    'density':    'Density',
    'lcc_ratio':  'LCC Ratio',
    'diam_ratio': 'Diameter Ratio',
    'max_degree': 'Max Degree',
}

COLOR_FILM = '#DD8452'
COLOR_REST = '#4C72B0'
COLOR_LINE = '#AAAAAA'


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — COMPUTE METRICS FROM JSON
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics():
    """
    Compute topology metrics for all film and Rest graphs.
    Returns two DataFrames: df_film (420 rows), df_rest (30 rows).
    """
    cfg_dir = GRAPHS_DIR / '100_riso' / 'graphs'

    # ── film graphs ──────────────────────────────────────────────────────────
    film_rows = []
    n_total   = len(SUBJECT_IDS) * len(FILMS)
    n_done    = 0

    for subject_id in SUBJECT_IDS:
        for film_name in FILMS:
            n_done += 1
            path = cfg_dir / f'{film_name}_100_riso_{subject_id}.json'
            if not path.exists():
                continue

            topo = compute_topology(load_graph(path))
            topo['lcc_ratio']  = topo['nodes_in_giant_component'] / topo['num_nodes'] if topo['num_nodes'] > 0 else 0
            topo['diam_ratio'] = topo['diameter_giant_component'] / topo['num_nodes'] if topo['num_nodes'] > 0 else 0
            topo['subject_id'] = subject_id
            topo['film']       = film_name
            film_rows.append(topo)

            if n_done % 60 == 0:
                print(f'  film: {n_done}/{n_total}')

    df_film = pd.DataFrame(film_rows)
    df_film.to_csv(OUT_DIR / 'metrics_film.csv', index=False)
    print(f'  Film graphs: {len(df_film)} rows saved.')

    # ── rest graphs ──────────────────────────────────────────────────────────
    rest_rows = []
    for subject_id in SUBJECT_IDS:
        path = cfg_dir / f'Rest_100_riso_{subject_id}.json'
        if not path.exists():
            continue
        topo = compute_topology(load_graph(path))
        topo['lcc_ratio']  = topo['nodes_in_giant_component'] / topo['num_nodes'] if topo['num_nodes'] > 0 else 0
        topo['diam_ratio'] = topo['diameter_giant_component'] / topo['num_nodes'] if topo['num_nodes'] > 0 else 0
        topo['subject_id'] = subject_id
        rest_rows.append(topo)

    df_rest = pd.DataFrame(rest_rows)
    df_rest.to_csv(OUT_DIR / 'metrics_rest.csv', index=False)
    print(f'  Rest graphs:  {len(df_rest)} rows saved.')

    return df_film, df_rest


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — DISTRIBUTIONS (420 film networks)
# ─────────────────────────────────────────────────────────────────────────────

def plot_distributions(df_film):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for ax, (key, label) in zip(axes, METRICS.items()):
        vals = df_film[key].dropna().values
        ax.boxplot(vals, patch_artist=True,
                   boxprops=dict(facecolor=COLOR_FILM, alpha=0.7),
                   medianprops=dict(color='black', linewidth=2),
                   whiskerprops=dict(linewidth=1.2),
                   capprops=dict(linewidth=1.2),
                   flierprops=dict(marker='o', markersize=3, alpha=0.4))
        ax.set_title(label, fontweight='bold', fontsize=11)
        ax.set_xticks([])
        ax.set_xlabel(f'mean={vals.mean():.3f}  std={vals.std():.3f}', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Global Topology Distributions — 420 Film Networks (100_riso)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'distributions.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: distributions.png')


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — WILCOXON FILM vs REST (paired per subject)
# ─────────────────────────────────────────────────────────────────────────────

def wilcoxon_film_vs_rest(df_film, df_rest):
    # film mean per subject across 14 films
    film_mean = df_film.groupby('subject_id')[list(METRICS.keys())].mean().reset_index()
    rest_sub  = df_rest.set_index('subject_id')[list(METRICS.keys())]

    stat_rows = []
    print('\nWilcoxon Film vs Rest:')
    print(f"  {'Metric':<16} {'Film mean':>10} {'Rest mean':>10} {'W':>8} {'p':>10} {'sig':>5}")

    for key, label in METRICS.items():
        film_vals = film_mean.set_index('subject_id')[key].reindex(rest_sub.index).values
        rest_vals = rest_sub[key].values

        # drop pairs where either value is NaN
        mask      = ~(np.isnan(film_vals) | np.isnan(rest_vals))
        film_vals = film_vals[mask]
        rest_vals = rest_vals[mask]

        stat, p = wilcoxon(film_vals, rest_vals, alternative='two-sided')
        sig     = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        direction = 'film>rest' if film_vals.mean() > rest_vals.mean() else 'rest>film'

        stat_rows.append(dict(
            metric=label, mean_film=round(film_vals.mean(), 4),
            mean_rest=round(rest_vals.mean(), 4), direction=direction,
            W=round(stat, 1), p_value=round(p, 6), significance=sig,
        ))
        print(f"  {label:<16} {film_vals.mean():>10.4f} {rest_vals.mean():>10.4f}"
              f" {stat:>8.1f} {p:>10.4f} {sig:>5}")

    df_stats = pd.DataFrame(stat_rows)
    df_stats.to_csv(OUT_DIR / 'wilcoxon_results.csv', index=False)
    return df_stats


def plot_paired_comparison(df_film, df_rest):
    film_mean = df_film.groupby('subject_id')[list(METRICS.keys())].mean().reset_index()
    rest_sub  = df_rest.set_index('subject_id')[list(METRICS.keys())].reset_index()

    fig, axes = plt.subplots(1, 4, figsize=(16, 6))

    for ax, (key, label) in zip(axes, METRICS.items()):
        film_vals = film_mean.set_index('subject_id')[key]
        rest_vals = rest_sub.set_index('subject_id')[key]
        common    = film_vals.index.intersection(rest_vals.index)

        fv = film_vals[common].values
        rv = rest_vals[common].values

        # subject connecting lines
        for f, r in zip(fv, rv):
            ax.plot([1, 2], [r, f], color=COLOR_LINE, alpha=0.4, linewidth=0.8)

        ax.boxplot([rv, fv], patch_artist=True,
                   boxprops=dict(alpha=0.7),
                   medianprops=dict(color='black', linewidth=2))
        ax.set_facecolor('white')
        # recolor boxes
        for patch, color in zip(ax.patches, [COLOR_REST, COLOR_FILM]):
            patch.set_facecolor(color)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Rest', 'Film'], fontsize=10)
        ax.set_title(label, fontweight='bold', fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Film vs Rest — Paired Wilcoxon (100_riso, n=30 subjects)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'paired_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: paired_comparison.png')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('[Step 1] Computing topology metrics from JSON graphs...')
    df_film, df_rest = compute_all_metrics()

    print('\n[Step 2] Plotting distributions...')
    plot_distributions(df_film)

    print('\n[Step 3] Wilcoxon Film vs Rest...')
    wilcoxon_film_vs_rest(df_film, df_rest)
    plot_paired_comparison(df_film, df_rest)

    print(f'\nAll outputs -> {OUT_DIR}')


if __name__ == '__main__':
    main()
