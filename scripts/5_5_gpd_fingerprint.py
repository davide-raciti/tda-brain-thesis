"""
5_5_gpd_fingerprint.py
======================
Geometric fingerprinting via Graph Portrait Divergence (thesis section 5.5).

The network portrait of a graph G is a matrix B where:
    B[k, j] = number of nodes u such that exactly j other nodes lie at
              hop-distance k from u.
Portrait Divergence (JSD, base-2, squared, range [0,1]) between two graphs
measures how geometrically different their multi-scale distance structure is.

Two comparisons:
  (A) Within-subject  : same subject, all 91 film pairs  — captures how much
                        each subject's topology changes across films.
  (B) Within-film     : same film, all subject pairs     — captures how much
                        subjects differ from each other in the same film.

If within-subject > within-film, the external stimulus drives topology
more than individual brain signatures do.

Fingerprinting: for each of the 91 film pairs (A, B) build a 30×30
subject-by-subject JSD matrix M[i,j] = JSD(subject_i in A, subject_j in B).
A subject is correctly identified if argmin(M[i,:]) == i (nearest neighbour
in film B is yourself).

Pipeline
--------
1. Compute portrait matrices for all 420 graphs (cached to pkl).
2. Global Mann-Whitney U test: within-subject vs within-film JSD.
3. Build 91 JSD matrices; compute per-pair identification rate and
   diagonal vs off-diagonal Wilcoxon test.
4. Pooled fingerprinting: Mann-Whitney diagonal vs all off-diagonal.
5. Plots: violin JSD distributions, identification-rate heatmap.

Outputs  results/5_5/
    portraits_cache.pkl         420 portrait matrices (reused if present)
    global_distances.csv        per-pair within-subject and within-film JSD
    global_mwu.csv              Mann-Whitney U test result
    fingerprint_stats.csv       per-pair id_rate, Wilcoxon result
    pooled_fingerprint.csv      pooled Mann-Whitney diagonal vs off-diagonal
    violin_global.png           within-subject vs within-film JSD violin
    id_rate_heatmap.png         14×14 identification rate per film pair
"""

import sys
import pickle
import warnings
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from scipy.spatial.distance import jensenshannon
from scipy.stats import mannwhitneyu, wilcoxon

warnings.filterwarnings('ignore')
np.random.seed(42)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from config import FILMS, SUBJECT_IDS, GRAPHS_DIR
from graph_metrics import load_graph, json_to_networkx

OUT_DIR    = Path(__file__).resolve().parent.parent / 'results' / '5_5'
CACHE_FILE = OUT_DIR / 'portraits_cache.pkl'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHORT = {f: f[:10] for f in FILMS}


# ─────────────────────────────────────────────────────────────────────────────
# PORTRAIT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_portrait(G):
    """
    Compute the network portrait matrix B for graph G.
    B[k, j] = number of nodes u such that exactly j nodes lie at
              BFS hop-distance k from u.
    Works on disconnected graphs (unreachable nodes are absent from BFS trees).
    """
    G = G.copy()
    G.remove_edges_from(nx.selfloop_edges(G))
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros((1, 1))

    all_pairs = dict(nx.all_pairs_shortest_path_length(G))
    max_dist  = max(max(d.values()) for d in all_pairs.values() if d)

    B = np.zeros((max_dist + 1, n + 1), dtype=float)
    for _, dists in all_pairs.items():
        for k, cnt in Counter(dists.values()).items():
            B[k, cnt] += 1
    return B


def portrait_divergence(B1, B2):
    """
    Squared Jensen-Shannon Divergence between two portrait matrices,
    zero-padded to the same shape. Returns JSD in [0, 1].
    """
    K = max(B1.shape[0], B2.shape[0])
    J = max(B1.shape[1], B2.shape[1])
    P1, P2 = np.zeros((K, J)), np.zeros((K, J))
    P1[:B1.shape[0], :B1.shape[1]] = B1
    P2[:B2.shape[0], :B2.shape[1]] = B2

    p1 = P1.flatten(); s1 = p1.sum()
    p2 = P2.flatten(); s2 = p2.sum()
    if s1 > 0: p1 = p1 / s1
    if s2 > 0: p2 = p2 / s2

    return float(jensenshannon(p1, p2, base=2) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — COMPUTE PORTRAITS (or load from cache)
# ─────────────────────────────────────────────────────────────────────────────

def get_portraits():
    if CACHE_FILE.exists():
        print('  Loading portraits from cache...')
        with open(CACHE_FILE, 'rb') as f:
            portraits = pickle.load(f)
        print(f'  {len(portraits)} portraits loaded.')
        return portraits

    cfg_dir  = GRAPHS_DIR / '100_riso' / 'graphs'
    n_total  = len(SUBJECT_IDS) * len(FILMS)
    portraits = {}
    n_done   = 0

    print(f'  Computing portraits for {n_total} graphs...')
    for subject_id in SUBJECT_IDS:
        for film in FILMS:
            n_done += 1
            path = cfg_dir / f'{film}_100_riso_{subject_id}.json'
            if not path.exists():
                continue
            G = json_to_networkx(load_graph(path))
            G.remove_edges_from(nx.selfloop_edges(G))
            if G.number_of_nodes() > 1:
                portraits[(subject_id, film)] = compute_portrait(G)
            if n_done % 60 == 0:
                print(f'    {n_done}/{n_total}  cached: {len(portraits)}')

    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(portraits, f)
    print(f'  Done. {len(portraits)} portraits cached.')
    return portraits


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — GLOBAL TEST: within-subject vs within-film JSD
# ─────────────────────────────────────────────────────────────────────────────

def global_distance_test(portraits):
    """
    Within-subject  : for each subject, average JSD across all 91 film pairs.
    Within-film     : for each film, average JSD across all C(30,2) subject pairs.
    Mann-Whitney U test: within-subject > within-film?
    """
    within_subject = []   # one value per (subject, film_pair)
    within_film    = []   # one value per (film, subject_pair)

    film_pairs = list(itertools.combinations(FILMS, 2))
    subj_pairs = list(itertools.combinations(SUBJECT_IDS, 2))

    # within-subject: same subject, different films
    for subject_id in SUBJECT_IDS:
        for fa, fb in film_pairs:
            if (subject_id, fa) in portraits and (subject_id, fb) in portraits:
                jsd = portrait_divergence(portraits[(subject_id, fa)],
                                          portraits[(subject_id, fb)])
                within_subject.append(jsd)

    # within-film: same film, different subjects
    for film in FILMS:
        for si, sj in subj_pairs:
            if (si, film) in portraits and (sj, film) in portraits:
                jsd = portrait_divergence(portraits[(si, film)],
                                          portraits[(sj, film)])
                within_film.append(jsd)

    ws = np.array(within_subject)
    wf = np.array(within_film)

    stat, p = mannwhitneyu(ws, wf, alternative='greater')
    ratio   = round(ws.mean() / wf.mean(), 4) if wf.mean() > 0 else np.nan
    sig     = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

    print(f'\n  Within-subject JSD : {ws.mean():.4f} ± {ws.std():.4f}  (n={len(ws):,})')
    print(f'  Within-film JSD    : {wf.mean():.4f} ± {wf.std():.4f}  (n={len(wf):,})')
    print(f'  Entrainment ratio  : {ratio:.4f}')
    print(f'  Mann-Whitney U     : U={stat:.0f}  p={p:.2e}  {sig}')

    pd.DataFrame([dict(mean_within_subject=round(ws.mean(), 4),
                       mean_within_film=round(wf.mean(), 4),
                       entrainment_ratio=ratio, U=round(stat, 1),
                       p_value=round(p, 10), significance=sig)]
                 ).to_csv(OUT_DIR / 'global_mwu.csv', index=False)

    return ws, wf


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — FINGERPRINTING: 91 subject×subject JSD matrices
# ─────────────────────────────────────────────────────────────────────────────

def compute_fingerprint_stats(portraits):
    """
    For each of the 91 film pairs, build the 30×30 subject-by-subject JSD
    matrix and compute:
      mean_diag     : mean within-subject JSD (diagonal)
      mean_offdiag  : mean between-subject JSD (off-diagonal)
      id_rate       : fraction of subjects correctly identified (argmin = self)
      p_wilcoxon    : Wilcoxon test diagonal < off-diagonal
    """
    film_pairs = list(itertools.combinations(FILMS, 2))
    rows       = []
    all_diag, all_offdiag = [], []

    print(f'\n  Building {len(film_pairs)} subject×subject matrices...')
    for fa, fb in film_pairs:
        subjs = [s for s in SUBJECT_IDS
                 if (s, fa) in portraits and (s, fb) in portraits]
        if len(subjs) < 2:
            continue

        n = len(subjs)
        M = np.zeros((n, n))
        for i, si in enumerate(subjs):
            for j, sj in enumerate(subjs):
                M[i, j] = portrait_divergence(portraits[(si, fa)],
                                              portraits[(sj, fb)])

        diag    = np.array([M[i, i] for i in range(n)])
        offdiag = np.array([M[i, j] for i in range(n)
                            for j in range(n) if i != j])

        all_diag.extend(diag.tolist())
        all_offdiag.extend(offdiag.tolist())

        # identification rate
        id_rate = float(sum(np.argmin(M[i, :]) == i for i in range(n))) / n

        # Wilcoxon diagonal vs per-row off-diagonal mean
        row_off_means = np.array([offdiag[i * (n-1): (i+1) * (n-1)].mean()
                                   for i in range(n)])
        try:
            _, p_wil = wilcoxon(diag, row_off_means, alternative='less')
        except ValueError:
            p_wil = 1.0

        sig = '***' if p_wil < 0.001 else '**' if p_wil < 0.01 else \
              '*' if p_wil < 0.05 else 'n.s.'
        rows.append(dict(Film_A=fa, Film_B=fb, n_subjects=n,
                         mean_diag=round(diag.mean(), 4),
                         mean_offdiag=round(offdiag.mean(), 4),
                         id_rate=round(id_rate, 4),
                         diag_minimized=bool(diag.mean() < offdiag.mean()),
                         p_wilcoxon=round(p_wil, 6),
                         significance=sig,
                         significant=(p_wil < 0.05)))

    df = pd.DataFrame(rows)
    n_dmin = df['diag_minimized'].sum()
    n_sig  = df['significant'].sum()
    print(f'  Diagonal minimization (mean_diag < mean_offdiag): '
          f'{n_dmin}/{len(df)} ({100*n_dmin/len(df):.1f}%)')
    print(f'  Per-pair Wilcoxon significant (stricter):          '
          f'{n_sig}/{len(df)} ({100*n_sig/len(df):.1f}%)')
    df.to_csv(OUT_DIR / 'fingerprint_stats.csv', index=False)
    return df, np.array(all_diag), np.array(all_offdiag)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — POOLED FINGERPRINT TEST
# ─────────────────────────────────────────────────────────────────────────────

def pooled_fingerprint_test(df_stats, all_diag, all_offdiag):
    stat, p = mannwhitneyu(all_diag, all_offdiag, alternative='less')
    mean_id  = df_stats['id_rate'].mean()   # mean identification rate across pairs
    chance   = 1.0 / len(SUBJECT_IDS)
    sig      = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

    print(f'\n  Pooled diagonal JSD    : {all_diag.mean():.4f} +/- {all_diag.std():.4f}')
    print(f'  Pooled off-diagonal JSD: {all_offdiag.mean():.4f} +/- {all_offdiag.std():.4f}')
    print(f'  Mann-Whitney (diag<off): U={stat:.0f}  p={p:.2e}  {sig}')
    print(f'  Mean id rate           : {mean_id:.4f}  (chance = {chance:.4f})')

    pd.DataFrame([dict(mean_diag=round(all_diag.mean(), 4),
                       mean_offdiag=round(all_offdiag.mean(), 4),
                       U=round(stat, 1), p_value=round(p, 10),
                       significance=sig, mean_id_rate=round(mean_id, 4),
                       chance_level=round(chance, 4))]
                 ).to_csv(OUT_DIR / 'pooled_fingerprint.csv', index=False)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_violin_global(ws, wf):
    fig, ax = plt.subplots(figsize=(6, 6))
    data    = [wf, ws]
    labels  = ['Within-film\n(same film,\ndiff. subjects)',
               'Within-subject\n(same subject,\ndiff. films)']
    vp = ax.violinplot(data, positions=[1, 2], showmedians=True, showextrema=True)
    colors = ['#4C72B0', '#DD8452']
    for body, color in zip(vp['bodies'], colors):
        body.set_facecolor(color)
        body.set_alpha(0.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Portrait Divergence (JSD)', fontsize=10)
    ax.set_title('Film Entrainment vs Individual Signatures\n(100_riso)',
                 fontweight='bold', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'violin_global.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: violin_global.png')


def plot_id_rate_heatmap(df_stats):
    n  = len(FILMS)
    mat = np.full((n, n), np.nan)
    idx = {f: i for i, f in enumerate(FILMS)}

    for _, row in df_stats.iterrows():
        i, j = idx[row['Film_A']], idx[row['Film_B']]
        mat[i, j] = mat[j, i] = row['id_rate']

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(mat, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    short = [SHORT[f] for f in FILMS]
    ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=7)
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                        fontsize=6, color='black' if mat[i, j] < 0.7 else 'white')
    fig.colorbar(im, ax=ax, label='Identification Rate')
    ax.set_title('Fingerprint Identification Rate per Film Pair (100_riso)',
                 fontweight='bold', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'id_rate_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: id_rate_heatmap.png')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('[Step 1] Computing/loading portrait matrices...')
    portraits = get_portraits()

    print('\n[Step 2] Global distance test (within-subject vs within-film)...')
    ws, wf = global_distance_test(portraits)

    print('\n[Step 3] Fingerprinting — 91 subject×subject matrices...')
    df_stats, all_diag, all_offdiag = compute_fingerprint_stats(portraits)

    print('\n[Step 4] Pooled fingerprint test...')
    pooled_fingerprint_test(df_stats, all_diag, all_offdiag)

    print('\nGenerating plots...')
    plot_violin_global(ws, wf)
    plot_id_rate_heatmap(df_stats)

    print(f'\nAll outputs -> {OUT_DIR}')


if __name__ == '__main__':
    main()
