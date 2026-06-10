"""
5_2b_power_law.py
=================
Degree distribution analysis and null model validation (thesis section
5.2, second half).

Pipeline
--------
1. Pool degree sequences from all 420 film graphs.
2. MLE fit: power law, truncated power law, exponential (Clauset method).
3. Vuong likelihood ratio tests between candidate distributions.
4. Bootstrap goodness-of-fit p-value for the power law.
5. Null model comparison (ER and BA) with substrate matching: each null is
   built on the same substrate as the empirical metric it is compared to.
       - lcc_ratio, max_degree : full-graph property -> null on the full graph
       - diameter              : LCC property        -> null on the LCC
   Wilcoxon test empirical vs ER and empirical vs BA for each metric.

Requires
--------
    pip install powerlaw

Outputs  results/5_2b/
    powerlaw_summary.csv       alpha, xmin, KS, bootstrap p
    lrt_results.csv            Vuong likelihood ratio results
    null_comparison.csv        per-graph empirical / ER / BA means
    null_wilcoxon.csv          Wilcoxon empirical vs ER and vs BA per metric
    ccdf_fits.png              CCDF log-log with power law + truncated fits
    null_boxplot.png           Empirical / ER / BA boxplot per metric
"""

import sys
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wilcoxon

warnings.filterwarnings('ignore')
np.random.seed(42)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from config import GRAPHS_DIR, SUBJECT_IDS, FILMS
from graph_metrics import load_graph, json_to_networkx, make_er, make_ba, lcc_of

import powerlaw

OUT_DIR      = Path(__file__).resolve().parent.parent / 'results' / '5_2b'
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_NULL       = 8     # null realisations per graph (each model)
N_BOOT       = 100   # bootstrap iterations for goodness-of-fit (p is robust; raise for a finer estimate)

COLOR_REAL   = '#4C72B0'
COLOR_ER     = '#C44E52'
COLOR_BA     = '#2CA02C'


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — POOL DEGREE SEQUENCES
# ─────────────────────────────────────────────────────────────────────────────

def collect_degrees():
    cfg_dir = GRAPHS_DIR / '100_riso' / 'graphs'
    all_degrees = []

    for subject_id in SUBJECT_IDS:
        for film_name in FILMS:
            path = cfg_dir / f'{film_name}_100_riso_{subject_id}.json'
            if not path.exists():
                continue
            g = load_graph(path)
            all_degrees.extend(node['degree'] for node in g['nodes'])

    degrees = np.array(all_degrees, dtype=int)
    # isolated nodes (degree=0) are uninformative for power law fitting
    return degrees[degrees > 0]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — MLE FIT
# ─────────────────────────────────────────────────────────────────────────────

def fit_distributions(degrees):
    print('  Fitting power law (automatic x_min, Clauset MLE)...')
    fit   = powerlaw.Fit(degrees, discrete=True, verbose=False)
    alpha = fit.power_law.alpha
    xmin  = fit.power_law.xmin
    sigma = fit.power_law.sigma
    D_ks  = fit.power_law.D

    print(f'    alpha = {alpha:.4f}  (sigma = {sigma:.4f})')
    print(f'    xmin  = {xmin:.0f}')
    print(f'    KS D  = {D_ks:.4f}')
    print(f'    tail  = {(degrees >= xmin).sum():,} / {len(degrees):,} nodes')

    return fit, alpha, xmin, sigma, D_ks


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — VUONG LRT
# ─────────────────────────────────────────────────────────────────────────────

def run_lrt(fit):
    comparisons = [
        ('truncated_power_law', 'Truncated Power Law'),
        ('exponential',         'Exponential'),
        ('lognormal',           'Lognormal'),
    ]

    rows = []
    print('\n  Vuong LRT (power law vs alternatives):')
    for dist_name, dist_label in comparisons:
        try:
            R, p = fit.distribution_compare('power_law', dist_name,
                                             normalized_ratio=True)
            sig  = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            if R > 0 and p < 0.05:
                verdict = 'Power Law favoured'
            elif R < 0 and p < 0.05:
                verdict = f'{dist_label} favoured'
            else:
                verdict = 'Indistinguishable'
            print(f'    PL vs {dist_label:<22}  R={R:+.3f}  p={p:.4f}  {sig}  -> {verdict}')
            rows.append(dict(comparison=f'PL vs {dist_label}', R=round(R, 4),
                             p_value=round(p, 6), significance=sig, verdict=verdict))
        except Exception as e:
            print(f'    PL vs {dist_label:<22}  ERROR: {e}')

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — BOOTSTRAP GOODNESS-OF-FIT
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_gof(fit, degrees, D_ks, xmin):
    print(f'\n  Bootstrap goodness-of-fit ({N_BOOT} iterations)...')
    tail_data = degrees[degrees >= xmin]
    ks_boot   = []

    for _ in range(N_BOOT):
        synthetic = fit.power_law.generate_random(len(tail_data)).astype(int)
        synthetic = np.maximum(synthetic, int(xmin))
        fit_boot  = powerlaw.Fit(synthetic, discrete=True, xmin=xmin, verbose=False)
        ks_boot.append(fit_boot.power_law.D)

    ks_boot = np.array(ks_boot)
    p_boot  = float(np.mean(ks_boot >= D_ks))
    print(f'    D observed  = {D_ks:.4f}')
    print(f'    D bootstrap = {ks_boot.mean():.4f} +/- {ks_boot.std():.4f}')
    print(f'    p-value     = {p_boot:.4f}  '
          f'-> {"power law compatible (p>=0.1)" if p_boot >= 0.1 else "power law rejected (p<0.1)"}')
    return p_boot


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — NULL MODELS (ER + BA, substrate matching)
# ─────────────────────────────────────────────────────────────────────────────

def _full_metrics(G):
    """lcc_ratio and max_degree, defined on the full graph."""
    n = G.number_of_nodes()
    if n == 0:
        return 0.0, 0
    lcc_n   = len(max(nx.connected_components(G), key=len))
    degrees = [d for _, d in G.degree()]
    return lcc_n / n, (max(degrees) if degrees else 0)


def _lcc_diameter(G):
    """Diameter of the largest connected component of G."""
    H = lcc_of(G)
    return nx.diameter(H) if H.number_of_nodes() > 1 else 0


def null_comparison():
    """
    Compare each empirical graph against ER and BA nulls, matching every null
    to the substrate on which the empirical metric is defined:
        lcc_ratio, max_degree  -> full graph
        diameter               -> largest connected component
    Each null value is the mean over N_NULL realisations.
    """
    cfg_dir = GRAPHS_DIR / '100_riso' / 'graphs'
    rows    = []
    n_total = len(SUBJECT_IDS) * len(FILMS)
    n_done  = 0

    for subject_id in SUBJECT_IDS:
        for film_name in FILMS:
            n_done += 1
            path = cfg_dir / f'{film_name}_100_riso_{subject_id}.json'
            if not path.exists():
                continue

            G = json_to_networkx(load_graph(path))
            n = G.number_of_nodes()
            E = G.number_of_edges()
            if n < 3 or E == 0:
                continue
            density = nx.density(G)

            # empirical
            emp_lcc, emp_deg = _full_metrics(G)
            emp_diam         = _lcc_diameter(G)

            # full-graph nulls: lcc_ratio, max_degree
            er_lcc, er_deg, ba_lcc, ba_deg = [], [], [], []
            for s in range(N_NULL):
                l, d = _full_metrics(make_er(n, density, seed=s))
                er_lcc.append(l); er_deg.append(d)
                l, d = _full_metrics(make_ba(n, E, seed=s))
                ba_lcc.append(l); ba_deg.append(d)

            # LCC-substrate nulls: diameter
            H     = lcc_of(G)
            n_lcc = H.number_of_nodes()
            e_lcc = H.number_of_edges()
            d_lcc = nx.density(H) if n_lcc > 1 else 0.0
            er_diam, ba_diam = [], []
            if n_lcc > 2 and e_lcc > 0:
                for s in range(N_NULL):
                    er_diam.append(_lcc_diameter(make_er(n_lcc, d_lcc, seed=s)))
                    ba_diam.append(_lcc_diameter(make_ba(n_lcc, e_lcc, seed=s)))

            rows.append(dict(
                subject_id=subject_id, film=film_name,
                emp_lcc=emp_lcc, er_lcc=np.mean(er_lcc), ba_lcc=np.mean(ba_lcc),
                emp_max=emp_deg, er_max=np.mean(er_deg), ba_max=np.mean(ba_deg),
                emp_diam=emp_diam,
                er_diam=np.mean(er_diam) if er_diam else np.nan,
                ba_diam=np.mean(ba_diam) if ba_diam else np.nan,
            ))

            if n_done % 60 == 0:
                print(f'    {n_done}/{n_total}...')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'null_comparison.csv', index=False)

    # Wilcoxon empirical vs each null, per metric
    stat_rows = []
    print('\n  Wilcoxon empirical vs null:')
    print(f"    {'Metric':<14} {'Null':<4} {'Emp':>8} {'Null':>8} {'p':>10} {'sig':>5}")
    metrics = [
        ('emp_lcc',  'lcc',  'LCC Ratio'),
        ('emp_max',  'max',  'Max Degree'),
        ('emp_diam', 'diam', 'Diameter'),
    ]
    for emp_col, suffix, label in metrics:
        for model in ('er', 'ba'):
            sub = df[[emp_col, f'{model}_{suffix}']].dropna()
            ev  = sub[emp_col].values
            nv  = sub[f'{model}_{suffix}'].values
            stat, p = wilcoxon(ev, nv, alternative='two-sided')
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            stat_rows.append(dict(metric=label, null=model.upper(),
                                  mean_emp=round(ev.mean(), 4),
                                  mean_null=round(nv.mean(), 4),
                                  W=round(stat, 1), p_value=round(p, 6),
                                  significance=sig))
            print(f"    {label:<14} {model.upper():<4} {ev.mean():>8.4f} "
                  f"{nv.mean():>8.4f} {p:>10.4f} {sig:>5}")

    pd.DataFrame(stat_rows).to_csv(OUT_DIR / 'null_wilcoxon.csv', index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_ccdf(fit, degrees):
    fig, ax = plt.subplots(figsize=(7, 5))
    fit.plot_ccdf(ax=ax, color='steelblue', linewidth=1.2,
                  label='Empirical CCDF', original_data=True)
    fit.power_law.plot_ccdf(ax=ax, color='crimson', linestyle='--',
                            linewidth=2, label=f'Power law  α={fit.power_law.alpha:.3f}')
    fit.truncated_power_law.plot_ccdf(ax=ax, color='darkorange', linestyle='-.',
                                      linewidth=2, label='Truncated power law')
    ax.set_xlabel('Degree', fontsize=11)
    ax.set_ylabel('P(X ≥ x)', fontsize=11)
    ax.set_title('Degree Distribution — CCDF (pooled 420 graphs, 100_riso)',
                 fontweight='bold', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'ccdf_fits.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: ccdf_fits.png')


def plot_null_boxplot(df):
    metrics = [
        ('emp_lcc',  'er_lcc',  'ba_lcc',  'LCC Ratio'),
        ('emp_max',  'er_max',  'ba_max',  'Max Degree'),
        ('emp_diam', 'er_diam', 'ba_diam', 'Diameter'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (ec, rc, bc, label) in zip(axes, metrics):
        data = [df[ec].dropna().values, df[rc].dropna().values, df[bc].dropna().values]
        ax.boxplot(data, patch_artist=True, showfliers=False,
                   medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(ax.patches, [COLOR_REAL, COLOR_ER, COLOR_BA]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Empirical', 'ER', 'BA'], fontsize=10)
        ax.set_title(label, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.suptitle('Empirical vs ER and BA Null Models (100_riso)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'null_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: null_boxplot.png')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('[Step 1] Collecting degree sequences...')
    degrees = collect_degrees()
    print(f'  {len(degrees):,} nodes (degree > 0)  |  '
          f'range {degrees.min()}–{degrees.max()}  |  mean {degrees.mean():.2f}')

    print('\n[Step 2] MLE fit...')
    fit, alpha, xmin, sigma, D_ks = fit_distributions(degrees)

    print('\n[Step 3] Vuong LRT...')
    lrt_rows = run_lrt(fit)

    print('\n[Step 4] Bootstrap goodness-of-fit...')
    p_boot = bootstrap_gof(fit, degrees, D_ks, xmin)

    # save summary CSV
    summary = dict(alpha=round(alpha, 4), sigma=round(sigma, 4),
                   xmin=int(xmin), KS_D=round(D_ks, 4),
                   p_bootstrap=round(p_boot, 4), n_nodes=len(degrees))
    pd.DataFrame([summary]).to_csv(OUT_DIR / 'powerlaw_summary.csv', index=False)
    pd.DataFrame(lrt_rows).to_csv(OUT_DIR / 'lrt_results.csv', index=False)

    print('\n[Step 5] Null model comparison (ER + BA, substrate matching)...')
    df_null = null_comparison()

    print('\nGenerating plots...')
    plot_ccdf(fit, degrees)
    plot_null_boxplot(df_null)

    print(f'\nAll outputs -> {OUT_DIR}')


if __name__ == '__main__':
    main()
