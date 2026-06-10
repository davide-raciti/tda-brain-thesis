"""
5_3a_kcore_betweenness.py
=========================
Meso-scale analysis: k-core decomposition and betweenness centrality,
with Rest vs Film comparison (thesis section 5.3, first half).

Pipeline
--------
1. Compute per-graph k-core and betweenness metrics for all 420 film
   graphs and 30 Rest graphs from JSON.
2. Paired Wilcoxon signed-rank test (Film mean vs Rest, per subject)
   for k-core metrics: max_core, mean_core, frac_innermost.
3. Paired Wilcoxon for betweenness metrics: mean_betweenness,
   max_betweenness, gini_betweenness.
4. Plots: paired boxplots for each metric set.

Outputs  results/5_3a/
    kcore_film.csv            k-core metrics for 420 film graphs
    kcore_rest.csv            k-core metrics for 30 Rest graphs
    betweenness_film.csv      betweenness metrics for 420 film graphs
    betweenness_rest.csv      betweenness metrics for 30 Rest graphs
    kcore_wilcoxon.csv        Wilcoxon results for k-core metrics
    betweenness_wilcoxon.csv  Wilcoxon results for betweenness metrics
    kcore_paired.png          paired boxplot Film vs Rest (k-core)
    betweenness_paired.png    paired boxplot Film vs Rest (betweenness)
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
from graph_metrics import load_graph, json_to_networkx, kcore_metrics, betweenness_metrics

OUT_DIR = Path(__file__).resolve().parent.parent / 'results' / '5_3a'
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_FILM = '#DD8452'
COLOR_REST = '#4C72B0'
COLOR_LINE = '#AAAAAA'


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — COMPUTE METRICS FROM JSON
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics():
    cfg_dir   = GRAPHS_DIR / '100_riso' / 'graphs'
    n_total   = len(SUBJECT_IDS) * len(FILMS)
    n_done    = 0

    kcore_film_rows = []
    bet_film_rows   = []

    for subject_id in SUBJECT_IDS:
        for film_name in FILMS:
            n_done += 1
            path = cfg_dir / f'{film_name}_100_riso_{subject_id}.json'
            if not path.exists():
                continue

            G  = json_to_networkx(load_graph(path))
            km = kcore_metrics(G)
            bm = betweenness_metrics(G)

            km.update(subject_id=subject_id, film=film_name)
            bm.update(subject_id=subject_id, film=film_name)
            kcore_film_rows.append(km)
            bet_film_rows.append(bm)

            if n_done % 60 == 0:
                print(f'  film: {n_done}/{n_total}')

    df_kcore_film = pd.DataFrame(kcore_film_rows)
    df_bet_film   = pd.DataFrame(bet_film_rows)
    df_kcore_film.to_csv(OUT_DIR / 'kcore_film.csv', index=False)
    df_bet_film.to_csv(OUT_DIR / 'betweenness_film.csv', index=False)
    print(f'  Film graphs: {len(df_kcore_film)} rows saved.')

    # ── Rest graphs ──────────────────────────────────────────────────────────
    kcore_rest_rows = []
    bet_rest_rows   = []

    for subject_id in SUBJECT_IDS:
        path = cfg_dir / f'Rest_100_riso_{subject_id}.json'
        if not path.exists():
            continue
        G  = json_to_networkx(load_graph(path))
        km = kcore_metrics(G)
        bm = betweenness_metrics(G)
        km.update(subject_id=subject_id)
        bm.update(subject_id=subject_id)
        kcore_rest_rows.append(km)
        bet_rest_rows.append(bm)

    df_kcore_rest = pd.DataFrame(kcore_rest_rows)
    df_bet_rest   = pd.DataFrame(bet_rest_rows)
    df_kcore_rest.to_csv(OUT_DIR / 'kcore_rest.csv', index=False)
    df_bet_rest.to_csv(OUT_DIR / 'betweenness_rest.csv', index=False)
    print(f'  Rest graphs:  {len(df_kcore_rest)} rows saved.')

    return df_kcore_film, df_kcore_rest, df_bet_film, df_bet_rest


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2-3 — WILCOXON PAIRED TESTS
# ─────────────────────────────────────────────────────────────────────────────

def run_wilcoxon(df_film, df_rest, metric_dict, out_name):
    """
    Paired Wilcoxon: per-subject film mean vs Rest for each metric.
    """
    film_mean = df_film.groupby('subject_id')[list(metric_dict.keys())].mean().reset_index()
    rest_sub  = df_rest.set_index('subject_id')[list(metric_dict.keys())]

    rows = []
    print(f'\nWilcoxon Film vs Rest ({out_name}):')
    print(f"  {'Metric':<22} {'Film':>8} {'Rest':>8} {'W':>8} {'p':>10} {'sig':>5}")

    for key, label in metric_dict.items():
        fv = film_mean.set_index('subject_id')[key].reindex(rest_sub.index).values
        rv = rest_sub[key].values
        mask = ~(np.isnan(fv) | np.isnan(rv))
        fv, rv = fv[mask], rv[mask]

        stat, p = wilcoxon(fv, rv, alternative='two-sided')
        sig     = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        direction = 'film>rest' if fv.mean() > rv.mean() else 'rest>film'
        rows.append(dict(metric=label, mean_film=round(fv.mean(), 4),
                         mean_rest=round(rv.mean(), 4), direction=direction,
                         W=round(stat, 1), p_value=round(p, 6), significance=sig))
        print(f"  {label:<22} {fv.mean():>8.4f} {rv.mean():>8.4f}"
              f" {stat:>8.1f} {p:>10.4f} {sig:>5}")

    df_stats = pd.DataFrame(rows)
    df_stats.to_csv(OUT_DIR / f'{out_name}_wilcoxon.csv', index=False)
    return df_stats


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_paired(df_film, df_rest, metric_dict, out_name, title):
    film_mean = df_film.groupby('subject_id')[list(metric_dict.keys())].mean().reset_index()
    rest_sub  = df_rest.set_index('subject_id')[list(metric_dict.keys())].reset_index()

    n_metrics = len(metric_dict)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    for ax, (key, label) in zip(axes, metric_dict.items()):
        fv = film_mean.set_index('subject_id')[key]
        rv = rest_sub.set_index('subject_id')[key]
        common = fv.index.intersection(rv.index)
        fv, rv = fv[common].values, rv[common].values

        for f, r in zip(fv, rv):
            ax.plot([1, 2], [r, f], color=COLOR_LINE, alpha=0.4, linewidth=0.8)

        ax.boxplot([rv, fv], patch_artist=True,
                   boxprops=dict(alpha=0.7),
                   medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(ax.patches, [COLOR_REST, COLOR_FILM]):
            patch.set_facecolor(color)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Rest', 'Film'], fontsize=10)
        ax.set_title(label, fontweight='bold', fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_DIR / f'{out_name}_paired.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_name}_paired.png')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

KCORE_METRICS = {
    'max_core':       'Max Core Number',
    'mean_core':      'Mean Core Number',
    'frac_innermost': 'Frac. Innermost',
}

BET_METRICS = {
    'mean_betweenness': 'Mean Betweenness',
    'max_betweenness':  'Max Betweenness',
    'gini_betweenness': 'Gini (Betweenness)',
}


def main():
    print('[Step 1] Computing k-core and betweenness metrics...')
    df_kf, df_kr, df_bf, df_br = compute_all_metrics()

    print('\n[Step 2] Wilcoxon k-core Film vs Rest...')
    run_wilcoxon(df_kf, df_kr, KCORE_METRICS, 'kcore')

    print('\n[Step 3] Wilcoxon betweenness Film vs Rest...')
    run_wilcoxon(df_bf, df_br, BET_METRICS, 'betweenness')

    print('\n[Step 4] Plots...')
    plot_paired(df_kf, df_kr, KCORE_METRICS, 'kcore',
                'K-core Film vs Rest — Paired Wilcoxon (100_riso, n=30)')
    plot_paired(df_bf, df_br, BET_METRICS, 'betweenness',
                'Betweenness Film vs Rest — Paired Wilcoxon (100_riso, n=30)')

    print(f'\nAll outputs -> {OUT_DIR}')


if __name__ == '__main__':
    main()
