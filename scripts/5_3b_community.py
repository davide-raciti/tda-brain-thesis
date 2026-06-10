"""
5_3b_community.py
=================
Meso-scale analysis: Louvain community structure on the LCC, Rest vs Film
comparison, pairwise film comparison, and cross-resolution consistency
(thesis section 5.3, second half).

Pipeline
--------
1. Louvain community detection on the LCC of each film graph + ER null model.
   Metrics: modularity_Q, n_communities, lcc_ratio.
2. Rest vs Film paired Wilcoxon for Q and n_communities.
3. Pairwise film comparison for n_communities (all C(14,2)=91 pairs).
4. Cross-resolution consistency: Spearman r of k-core and betweenness
   metrics between 100_riso (reference) and 200/400/800_riso.

Outputs  results/5_3b/
    community_film.csv          per-graph Louvain metrics (420 rows)
    community_rest.csv          per-graph Louvain metrics Rest (30 rows)
    community_wilcoxon.csv      Wilcoxon Film vs Rest
    film_pairs_ncommunities.csv pairwise Wilcoxon n_communities (91 pairs)
    cross_resolution.csv        per-graph meso metrics across 4 configs
    cross_resolution_spearman.csv  Spearman r vs 100_riso reference
    community_paired.png        paired boxplot Film vs Rest
    cross_resolution_plot.png   Spearman r heatmap across configs
"""

import sys
import warnings
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wilcoxon, spearmanr, false_discovery_control
from networkx.algorithms.community import louvain_communities, modularity

warnings.filterwarnings('ignore')
np.random.seed(42)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from config import FILMS, SUBJECT_IDS, GRAPHS_DIR
from graph_metrics import (
    load_graph, json_to_networkx,
    kcore_metrics, betweenness_metrics,
    make_er, make_ba,
)

OUT_DIR = Path(__file__).resolve().parent.parent / 'results' / '5_3b'
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_RUNS = 10   # Louvain repeats — keep the partition with highest Q
N_NULL = 8    # ER realisations per graph

COLOR_FILM = '#DD8452'
COLOR_REST = '#4C72B0'
COLOR_LINE = '#AAAAAA'
COLOR_ER   = '#C44E52'


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def best_louvain(G):
    """Run Louvain N_RUNS times; return (partition, Q) with highest modularity."""
    best_partition, best_Q = None, -np.inf
    for seed in range(N_RUNS):
        try:
            partition = louvain_communities(G, seed=seed)
            Q         = modularity(G, partition)
            if Q > best_Q:
                best_Q, best_partition = Q, partition
        except Exception:
            continue
    return best_partition, best_Q


def lcc_subgraph(G):
    """Return the largest connected component as a new graph."""
    if G.number_of_nodes() == 0:
        return G
    lcc = max(nx.connected_components(G), key=len)
    return G.subgraph(lcc).copy()


def community_metrics(G):
    """
    Louvain metrics on the LCC of G.
    Returns dict with modularity_Q, n_communities, lcc_ratio.
    """
    n_total = G.number_of_nodes()
    G_lcc   = lcc_subgraph(G)
    n_lcc   = G_lcc.number_of_nodes()

    if n_lcc < 2:
        return dict(modularity_Q=0.0, n_communities=1,
                    lcc_ratio=n_lcc / n_total if n_total > 0 else 0.0,
                    lcc_size=n_lcc)

    partition, Q = best_louvain(G_lcc)
    n_comm       = len(partition) if partition else 1

    return dict(
        modularity_Q   = round(Q, 4),
        n_communities  = n_comm,
        lcc_ratio      = round(n_lcc / n_total, 4),
        lcc_size       = n_lcc,
    )


def null_community_metrics(n_lcc, density_lcc, n_edges_lcc, model):
    """
    Mean Louvain metrics over N_NULL null realisations matched to the LCC.
    Both nulls are built on the LCC, since modularity is computed on the LCC.
        model='er' : Erdős–Rényi matched to LCC size and density
        model='ba' : Barabási–Albert matched to LCC size and edge count
    """
    Q_vals, nc_vals = [], []
    for s in range(N_NULL):
        if model == 'er':
            G = make_er(n_lcc, density_lcc, seed=s)
        else:
            G = make_ba(n_lcc, n_edges_lcc, seed=s)
        G = lcc_subgraph(G)
        if G.number_of_nodes() < 2:
            continue
        part, Q = best_louvain(G)
        Q_vals.append(Q)
        nc_vals.append(len(part) if part else 1)
    return (round(np.mean(Q_vals), 4) if Q_vals else 0.0,
            round(np.mean(nc_vals), 2) if nc_vals else 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — COMMUNITY METRICS (film + Rest)
# ─────────────────────────────────────────────────────────────────────────────

def compute_community_metrics():
    cfg_dir = GRAPHS_DIR / '100_riso' / 'graphs'
    n_total = len(SUBJECT_IDS) * len(FILMS)
    n_done  = 0

    film_rows = []
    for subject_id in SUBJECT_IDS:
        for film_name in FILMS:
            n_done += 1
            path = cfg_dir / f'{film_name}_100_riso_{subject_id}.json'
            if not path.exists():
                continue

            G  = json_to_networkx(load_graph(path))
            cm = community_metrics(G)

            # ER and BA nulls, both matched to the LCC
            G_lcc       = lcc_subgraph(G)
            lcc_density = nx.density(G_lcc) if cm['lcc_size'] > 1 else 0.0
            lcc_edges   = G_lcc.number_of_edges()
            er_Q, er_nc = null_community_metrics(cm['lcc_size'], lcc_density, lcc_edges, 'er')
            ba_Q, ba_nc = null_community_metrics(cm['lcc_size'], lcc_density, lcc_edges, 'ba')

            cm.update(subject_id=subject_id, film=film_name,
                      er_modularity_Q=er_Q, er_n_communities=er_nc,
                      ba_modularity_Q=ba_Q, ba_n_communities=ba_nc)
            film_rows.append(cm)

            if n_done % 60 == 0:
                print(f'  film: {n_done}/{n_total}')

    df_film = pd.DataFrame(film_rows)
    df_film.to_csv(OUT_DIR / 'community_film.csv', index=False)
    print(f'  Film graphs: {len(df_film)} rows saved.')

    # ── Rest ─────────────────────────────────────────────────────────────────
    rest_rows = []
    for subject_id in SUBJECT_IDS:
        path = cfg_dir / f'Rest_100_riso_{subject_id}.json'
        if not path.exists():
            continue
        G  = json_to_networkx(load_graph(path))
        cm = community_metrics(G)
        cm.update(subject_id=subject_id)
        rest_rows.append(cm)

    df_rest = pd.DataFrame(rest_rows)
    df_rest.to_csv(OUT_DIR / 'community_rest.csv', index=False)
    print(f'  Rest graphs:  {len(df_rest)} rows saved.')

    return df_film, df_rest


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — WILCOXON FILM vs REST
# ─────────────────────────────────────────────────────────────────────────────

def wilcoxon_rest_film(df_film, df_rest):
    metrics = {'modularity_Q': 'Modularity Q', 'n_communities': 'N Communities'}
    film_mean = df_film.groupby('subject_id')[list(metrics.keys())].mean().reset_index()
    rest_sub  = df_rest.set_index('subject_id')[list(metrics.keys())]

    rows = []
    print('\nWilcoxon Film vs Rest (community):')
    for key, label in metrics.items():
        fv   = film_mean.set_index('subject_id')[key].reindex(rest_sub.index).values
        rv   = rest_sub[key].values
        mask = ~(np.isnan(fv) | np.isnan(rv))
        fv, rv = fv[mask], rv[mask]
        stat, p = wilcoxon(fv, rv, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        rows.append(dict(metric=label, mean_film=round(fv.mean(), 4),
                         mean_rest=round(rv.mean(), 4), W=round(stat, 1),
                         p_value=round(p, 6), significance=sig))
        print(f'  {label:<20} film={fv.mean():.4f}  rest={rv.mean():.4f}'
              f'  W={stat:.0f}  p={p:.4f}  {sig}')

    pd.DataFrame(rows).to_csv(OUT_DIR / 'community_wilcoxon.csv', index=False)


def community_vs_null(df_film):
    """
    Paired Wilcoxon of empirical community metrics against the ER and BA nulls,
    across all film graphs. Both nulls are matched to the LCC.
    """
    rows = []
    print('\nCommunity empirical vs null (per-graph paired Wilcoxon):')
    print(f"  {'Metric':<16} {'Null':<4} {'Emp':>8} {'Null':>8} {'p':>10} {'sig':>5}")

    for emp_col, label in [('modularity_Q', 'Modularity Q'),
                           ('n_communities', 'N Communities')]:
        for model in ('er', 'ba'):
            null_col = f'{model}_{emp_col}'
            sub = df_film[[emp_col, null_col]].dropna()
            ev, nv = sub[emp_col].values, sub[null_col].values
            stat, p = wilcoxon(ev, nv, alternative='two-sided')
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            rows.append(dict(metric=label, null=model.upper(),
                             mean_emp=round(ev.mean(), 4),
                             mean_null=round(nv.mean(), 4),
                             W=round(stat, 1), p_value=round(p, 6),
                             significance=sig))
            print(f"  {label:<16} {model.upper():<4} {ev.mean():>8.4f} "
                  f"{nv.mean():>8.4f} {p:>10.4f} {sig:>5}")

    pd.DataFrame(rows).to_csv(OUT_DIR / 'community_null_wilcoxon.csv', index=False)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — PAIRWISE FILM COMPARISON (n_communities)
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_film_comparison(df_film):
    """
    For each pair of films, Wilcoxon test on n_communities across 30 subjects.
    Raw p-values are corrected for the many simultaneous comparisons with both
    Benjamini-Hochberg (FDR) and Bonferroni (FWER).
    """
    pivot = df_film.pivot_table(index='subject_id', columns='film',
                                values='n_communities')
    rows  = []
    pairs = list(itertools.combinations(FILMS, 2))

    for fa, fb in pairs:
        if fa not in pivot.columns or fb not in pivot.columns:
            continue
        merged = pivot[[fa, fb]].dropna()
        if len(merged) < 5:
            continue
        try:
            stat, p = wilcoxon(merged[fa].values, merged[fb].values,
                               alternative='two-sided')
        except ValueError:
            continue
        rows.append(dict(Film_A=fa, Film_B=fb, W=round(stat, 1), p_raw=p))

    df_pairs = pd.DataFrame(rows)

    # multiple comparison corrections over the tested pairs
    s = len(df_pairs)
    p_raw = df_pairs['p_raw'].values
    df_pairs['p_BH']         = false_discovery_control(p_raw, method='bh')
    df_pairs['p_Bonferroni'] = np.minimum(p_raw * s, 1.0)
    df_pairs['sig_raw']      = df_pairs['p_raw']         < 0.05
    df_pairs['sig_BH']       = df_pairs['p_BH']          < 0.05
    df_pairs['sig_Bonf']     = df_pairs['p_Bonferroni']  < 0.05
    df_pairs['p_raw']        = df_pairs['p_raw'].round(6)
    df_pairs['p_BH']         = df_pairs['p_BH'].round(6)
    df_pairs['p_Bonferroni'] = df_pairs['p_Bonferroni'].round(6)

    print(f'\n  Pairwise n_communities ({s} pairs):')
    print(f"    significant raw     : {df_pairs['sig_raw'].sum()}/{s}")
    print(f"    significant BH/FDR  : {df_pairs['sig_BH'].sum()}/{s}")
    print(f"    significant Bonf.   : {df_pairs['sig_Bonf'].sum()}/{s}")
    df_pairs.to_csv(OUT_DIR / 'film_pairs_ncommunities.csv', index=False)
    return df_pairs


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — CROSS-RESOLUTION CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS    = ['100_riso', '200_riso', '400_riso', '800_riso']
MESO_KEYS  = ['max_core', 'mean_core', 'frac_innermost',
              'mean_betweenness', 'max_betweenness', 'gini_betweenness']


def compute_cross_resolution():
    """
    For each (subject, film) pair compute meso metrics across all 4 riso
    configurations. Returns a DataFrame with config, subject_id, film, metrics.
    """
    all_rows = []

    for cfg in CONFIGS:
        cfg_dir = GRAPHS_DIR / cfg / 'graphs'
        n_done  = 0
        print(f'  Computing {cfg}...')

        for subject_id in SUBJECT_IDS:
            for film_name in FILMS:
                path = cfg_dir / f'{film_name}_{cfg}_{subject_id}.json'
                if not path.exists():
                    continue
                G  = json_to_networkx(load_graph(path))
                km = kcore_metrics(G)
                bm = betweenness_metrics(G)
                row = dict(config=cfg, subject_id=subject_id, film=film_name)
                row.update(km)
                row.update(bm)
                all_rows.append(row)
                n_done += 1

        print(f'    {n_done} graphs processed.')

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / 'cross_resolution.csv', index=False)
    return df


def spearman_consistency(df_cross):
    """
    Spearman r between 100_riso (reference) and each other config,
    computed across all (subject, film) pairs.
    """
    ref = df_cross[df_cross['config'] == '100_riso'].set_index(
        ['subject_id', 'film'])[MESO_KEYS]

    rows = []
    print('\nSpearman r vs 100_riso:')
    print(f"  {'Config':<12} {'Metric':<22} {'r':>7}  {'p':>10}")

    for cfg in ['200_riso', '400_riso', '800_riso']:
        other = df_cross[df_cross['config'] == cfg].set_index(
            ['subject_id', 'film'])[MESO_KEYS]
        merged = ref.join(other, lsuffix='_ref', rsuffix='_other').dropna()

        for key in MESO_KEYS:
            r, p = spearmanr(merged[f'{key}_ref'], merged[f'{key}_other'])
            sig  = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            rows.append(dict(config=cfg, metric=key, spearman_r=round(r, 4),
                             p_value=round(p, 6), significance=sig))
            print(f'  {cfg:<12} {key:<22} {r:>7.4f}  {p:>10.4f}  {sig}')

    df_spearman = pd.DataFrame(rows)
    df_spearman.to_csv(OUT_DIR / 'cross_resolution_spearman.csv', index=False)
    return df_spearman


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_community_paired(df_film, df_rest):
    metrics  = {'modularity_Q': 'Modularity Q', 'n_communities': 'N Communities'}
    film_mean = df_film.groupby('subject_id')[list(metrics.keys())].mean().reset_index()
    rest_sub  = df_rest.set_index('subject_id')[list(metrics.keys())].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    for ax, (key, label) in zip(axes, metrics.items()):
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

    fig.suptitle('Community Structure Film vs Rest — Paired Wilcoxon (n=30)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'community_paired.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: community_paired.png')


def plot_cross_resolution(df_spearman):
    configs = ['200_riso', '400_riso', '800_riso']
    n_cfg   = len(configs)
    n_met   = len(MESO_KEYS)

    mat = np.zeros((n_met, n_cfg))
    for j, cfg in enumerate(configs):
        for i, key in enumerate(MESO_KEYS):
            row = df_spearman[(df_spearman['config'] == cfg) &
                              (df_spearman['metric'] == key)]
            if not row.empty:
                mat[i, j] = row['spearman_r'].values[0]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(mat, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(n_cfg))
    ax.set_xticklabels(configs, fontsize=9)
    ax.set_yticks(range(n_met))
    ax.set_yticklabels(MESO_KEYS, fontsize=9)
    for i in range(n_met):
        for j in range(n_cfg):
            ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center',
                    fontsize=8, color='black' if abs(mat[i, j]) < 0.8 else 'white')
    fig.colorbar(im, ax=ax, label='Spearman r')
    ax.set_title('Cross-Resolution Consistency vs 100_riso', fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'cross_resolution_plot.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: cross_resolution_plot.png')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('[Step 1] Community detection (Louvain on LCC) + ER and BA nulls...')
    df_film, df_rest = compute_community_metrics()

    print('\n[Step 2] Wilcoxon Film vs Rest + empirical vs null...')
    wilcoxon_rest_film(df_film, df_rest)
    community_vs_null(df_film)

    print('\n[Step 3] Pairwise film comparison (n_communities)...')
    pairwise_film_comparison(df_film)

    print('\n[Step 4] Cross-resolution consistency...')
    df_cross    = compute_cross_resolution()
    df_spearman = spearman_consistency(df_cross)

    print('\nGenerating plots...')
    plot_community_paired(df_film, df_rest)
    plot_cross_resolution(df_spearman)

    print(f'\nAll outputs -> {OUT_DIR}')


if __name__ == '__main__':
    main()
