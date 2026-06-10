"""
5_4_nodal_emotion.py
====================
Micro-scale analysis: nodal topological centrality vs emotional content
(thesis section 5.4).

For each node in all 420 Mapper graphs (100_riso):
  - Compute Core Number, Degree, Betweenness directly from JSON.
  - Assign the functional node category (Peripheral / Bridge / Connector Hub /
    Provincial Hub) from coreness and betweenness.
  - Compute Emotional Activation and Emotional Complexity from All50.

Then run Spearman rank correlations between the 3 topological predictors
and the 2 emotional metrics (6 tests total), and summarise the emotional
profile by node category.

Pipeline
--------
1. Pre-compute within-film Z-scores of the All50 matrix.
2. Loop over all 420 graphs and all nodes: compute per-node metrics + category.
3. Spearman correlations (pooled ~44k nodes).
4. Emotional profile by node category.
5. Plots: 3-panel scatter (topology vs Activation) and 2x2 centerpiece
   (Activation / Complexity by node category and by core depth).

Outputs  results/5_4/
    nodal_data.csv          per-node dataset (~44k rows)
    spearman_results.csv    6-row correlation table
    category_summary.csv    Activation/Complexity means + counts per category
    kruskal_results.csv     Kruskal-Wallis test across node categories
    scatter_activation.png  3-panel scatter (Core / Degree / Betweenness)
    centerpiece.png         2x2 panel: emotion by category and by core depth
"""

import sys
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, kruskal

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from config import FILMS, FILM_LENGTHS, SUBJECT_IDS, GRAPHS_DIR
from data_loader import load_all50
from graph_metrics import load_graph, json_to_networkx, node_category

CATEGORIES = ['Peripheral', 'Bridge', 'Provincial Hub', 'Connector Hub']
CAT_COLORS = {
    'Peripheral':     '#AAAAAA',
    'Bridge':         '#E84040',
    'Provincial Hub': '#4C72B0',
    'Connector Hub':  '#FF8C00',
}

OUT_DIR  = Path(__file__).resolve().parent.parent / 'results' / '5_4'
OUT_DIR.mkdir(parents=True, exist_ok=True)

Z_THRESH = 1.5
N_EMO    = 50


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — PRE-COMPUTE EMOTION MATRICES
# ─────────────────────────────────────────────────────────────────────────────

def prepare_emotion_matrices(emo_raw):
    """
    For each film return:
      emo_z[film]    : (T, 50) within-film Z-scored matrix  (for Activation)
      emo_norm[film] : (T, 50) raw values                   (for Complexity)
    """
    emo_z, emo_norm = {}, {}
    for film in FILMS:
        arr   = emo_raw[film]           # (T, 50)
        mean  = arr.mean(axis=0)
        std   = arr.std(axis=0)
        std[std == 0] = 1.0             # avoid division by zero
        emo_z[film]    = (arr - mean) / std
        emo_norm[film] = arr
    return emo_z, emo_norm


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — PER-NODE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_nodal_dataset(emo_z, emo_norm):
    """
    For each node in the 420 film graphs compute:
      Core_Number          : k-core shell
      Degree               : node degree
      Betweenness          : normalised betweenness centrality
      Emotional_Activation : fraction of 50 Z-scored features with |Z| > 1.5
      Emotional_Complexity : Shannon entropy (bits) of the raw mean profile
    """
    cfg_dir  = GRAPHS_DIR / '100_riso' / 'graphs'
    rows     = []
    n_total  = len(SUBJECT_IDS) * len(FILMS)
    n_done   = 0

    for film in FILMS:
        z_mat    = emo_z[film]      # (T, 50)
        norm_mat = emo_norm[film]   # (T, 50)
        T        = FILM_LENGTHS[film]

        for subject_id in SUBJECT_IDS:
            n_done += 1
            path = cfg_dir / f'{film}_100_riso_{subject_id}.json'
            if not path.exists():
                continue

            G     = json_to_networkx(load_graph(path))
            cores = nx.core_number(G)
            bet   = nx.betweenness_centrality(G, normalized=True)

            if n_done % 60 == 0:
                print(f'  {n_done}/{n_total}  nodes so far: {len(rows):,}')

            for node_id, node_data in G.nodes(data=True):
                members = [t for t in node_data.get('members', []) if t < T]
                if not members:
                    continue

                idxs = np.array(members, dtype=int)

                # Emotional Activation
                z_mean     = z_mat[idxs].mean(axis=0)       # (50,)
                activation = float((np.abs(z_mean) > Z_THRESH).sum()) / N_EMO

                # Emotional Complexity
                profile = norm_mat[idxs].mean(axis=0)        # (50,)
                shifted = profile - profile.min()
                total   = shifted.sum()
                p       = shifted / total if total > 0 else np.full(N_EMO, 1.0 / N_EMO)
                p       = np.clip(p, 1e-12, None)
                entropy = float(-np.sum(p * np.log2(p)))

                core_num = int(cores.get(node_id, 0))
                betw     = float(bet.get(node_id, 0.0))

                rows.append(dict(
                    Subject_ID           = subject_id,
                    Film_Name            = film,
                    Node_ID              = node_id,
                    Core_Number          = core_num,
                    Degree               = int(G.degree(node_id)),
                    Betweenness          = round(betw, 6),
                    Category             = node_category(core_num, betw),
                    N_members            = len(members),
                    Emotional_Activation = round(activation, 4),
                    Emotional_Complexity = round(entropy, 4),
                ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'nodal_data.csv', index=False)
    print(f'\n  Total nodes: {len(df):,}')
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — SPEARMAN CORRELATIONS
# ─────────────────────────────────────────────────────────────────────────────

def run_spearman(df):
    predictors = ['Core_Number', 'Degree', 'Betweenness']
    metrics    = [('Emotional_Activation', 'Activation'),
                  ('Emotional_Complexity', 'Complexity')]

    rows = []
    print('\nSpearman correlations (pooled nodes):')
    print(f"  {'Predictor':<16} {'Metric':<14} {'r':>8}  {'p':>12}  {'sig':>5}")

    for pred in predictors:
        for metric_col, metric_label in metrics:
            r, p = spearmanr(df[pred], df[metric_col])
            sig  = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            rows.append(dict(Predictor=pred.replace('_', ' '),
                             Metric=metric_label,
                             Spearman_r=round(r, 4),
                             p_value=round(p, 8),
                             significance=sig))
            print(f"  {pred:<16} {metric_label:<14} {r:>+8.4f}  {p:>12.2e}  {sig}")

    df_corr = pd.DataFrame(rows)
    df_corr.to_csv(OUT_DIR / 'spearman_results.csv', index=False)
    return df_corr


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — EMOTIONAL PROFILE BY NODE CATEGORY
# ─────────────────────────────────────────────────────────────────────────────

def category_summary(df):
    """
    Mean Emotional Activation and Complexity per node category, with counts.
    Tests whether topological role is associated with emotional engagement.
    """
    rows = []
    print('\nEmotional profile by node category:')
    print(f"  {'Category':<16} {'N':>8} {'%':>6} {'Activation':>12} {'Complexity':>12}")

    total = len(df)
    for cat in CATEGORIES:
        sub = df[df['Category'] == cat]
        if sub.empty:
            continue
        rows.append(dict(
            Category=cat, N=len(sub),
            Pct=round(100 * len(sub) / total, 2),
            Mean_Activation=round(sub['Emotional_Activation'].mean(), 4),
            Mean_Complexity=round(sub['Emotional_Complexity'].mean(), 4),
        ))
        print(f"  {cat:<16} {len(sub):>8,} {100*len(sub)/total:>5.1f}% "
              f"{sub['Emotional_Activation'].mean():>12.4f} "
              f"{sub['Emotional_Complexity'].mean():>12.4f}")

    df_cat = pd.DataFrame(rows)
    df_cat.to_csv(OUT_DIR / 'category_summary.csv', index=False)
    return df_cat


def kruskal_categories(df):
    """
    Kruskal-Wallis test of whether Emotional Activation and Emotional Complexity
    differ across the node categories. Non-parametric analogue of one-way ANOVA,
    appropriate since the per-node metrics are not normally distributed.
    """
    cats = [c for c in CATEGORIES if c in df['Category'].unique()]
    rows = []
    print('\nKruskal-Wallis across node categories:')

    for metric, label in [('Emotional_Activation', 'Activation'),
                          ('Emotional_Complexity', 'Complexity')]:
        groups = [df[df['Category'] == c][metric].values for c in cats]
        H, p   = kruskal(*groups)
        sig    = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        rows.append(dict(metric=label, n_groups=len(cats),
                         H=round(float(H), 4), p_value=p, significance=sig))
        print(f"  {label:<12} H={H:.2f}  p={p:.2e}  {sig}  (k={len(cats)} categories)")

    pd.DataFrame(rows).to_csv(OUT_DIR / 'kruskal_results.csv', index=False)


def plot_centerpiece(df):
    """
    2x2 centerpiece: Emotional Activation and Complexity broken down by
    functional node category (top row) and by core depth (bottom row).
    Visualises the center-periphery affective polarization.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    cats        = [c for c in CATEGORIES if c in df['Category'].unique()]
    core_levels = sorted(df['Core_Number'].unique())

    # top row — by node category
    for ax, metric, label in [
        (axes[0, 0], 'Emotional_Activation', 'Emotional Activation'),
        (axes[0, 1], 'Emotional_Complexity', 'Emotional Complexity'),
    ]:
        data = [df[df['Category'] == c][metric].values for c in cats]
        bp   = ax.boxplot(data, patch_artist=True, showfliers=False,
                          medianprops=dict(color='black', linewidth=2))
        for patch, c in zip(bp['boxes'], cats):
            patch.set_facecolor(CAT_COLORS[c])
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(cats) + 1))
        ax.set_xticklabels(cats, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f'{label} by node category', fontweight='bold', fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # bottom row — by core depth
    for ax, metric, label in [
        (axes[1, 0], 'Emotional_Activation', 'Emotional Activation'),
        (axes[1, 1], 'Emotional_Complexity', 'Emotional Complexity'),
    ]:
        data = [df[df['Core_Number'] == k][metric].values for k in core_levels]
        bp   = ax.boxplot(data, patch_artist=True, showfliers=False,
                          medianprops=dict(color='black', linewidth=2),
                          boxprops=dict(facecolor='#4C72B0', alpha=0.7))
        ax.set_xticks(range(1, len(core_levels) + 1))
        ax.set_xticklabels(core_levels, fontsize=9)
        ax.set_xlabel('Core Number (k-shell)', fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f'{label} by core depth', fontweight='bold', fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Center-Periphery Affective Polarization — 100_riso',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'centerpiece.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: centerpiece.png')


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — SCATTER PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter_activation(df, df_corr):
    """
    3-panel scatter: Core / Degree / Betweenness vs Emotional Activation.
    Points coloured by Emotional Complexity.
    """
    predictors = [
        ('Core_Number', 'Core Number'),
        ('Degree',      'Degree'),
        ('Betweenness', 'Betweenness'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # subsample for plotting performance (~5k points)
    df_plot = df.sample(min(5000, len(df)), random_state=42)
    vmin    = df_plot['Emotional_Complexity'].quantile(0.05)
    vmax    = df_plot['Emotional_Complexity'].quantile(0.95)

    for ax, (col, label) in zip(axes, predictors):
        sc = ax.scatter(df_plot[col], df_plot['Emotional_Activation'],
                        c=df_plot['Emotional_Complexity'],
                        cmap='viridis', vmin=vmin, vmax=vmax,
                        alpha=0.35, s=8, rasterized=True)

        # annotate Spearman r
        r_row = df_corr[(df_corr['Predictor'] == col.replace('_', ' ')) &
                        (df_corr['Metric'] == 'Activation')]
        if not r_row.empty:
            r   = r_row['Spearman_r'].values[0]
            sig = r_row['significance'].values[0]
            ax.text(0.05, 0.93, f'r = {r:+.3f}  {sig}',
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('Emotional Activation', fontsize=10)
        ax.set_title(f'{label} vs Activation', fontweight='bold', fontsize=11)
        ax.grid(linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.colorbar(sc, ax=axes[-1], label='Emotional Complexity (bits)')
    fig.suptitle('Nodal Topology vs Emotional Activation — 100_riso (~44k nodes)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'scatter_activation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: scatter_activation.png')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('[Step 1] Loading All50 emotion data...')
    emo_raw = load_all50()
    emo_z, emo_norm = prepare_emotion_matrices(emo_raw)
    print(f'  Z-score matrices ready for {len(emo_z)} films.')

    print('\n[Step 2] Computing per-node metrics...')
    df = compute_nodal_dataset(emo_z, emo_norm)

    print('\n[Step 3] Spearman correlations...')
    df_corr = run_spearman(df)

    print('\n[Step 4] Emotional profile by node category...')
    category_summary(df)
    kruskal_categories(df)

    print('\n[Step 5] Plots...')
    plot_scatter_activation(df, df_corr)
    plot_centerpiece(df)

    print(f'\nAll outputs -> {OUT_DIR}')


if __name__ == '__main__':
    main()
