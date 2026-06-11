# TDA Brain Thesis — Dynamical Mapping of Emotional Brain States during Film Viewing

Code for my bachelor thesis (Mathematics for Engineering, Politecnico di Torino).
It applies **Topological Data Analysis** (the *Mapper* algorithm, Saggar et al. 2018)
to naturalistic fMRI from the **Emo-FilM** dataset, building one topological graph per
(subject, film) and relating its structure to the emotional content of the stimulus.

- **30 subjects × 14 short films + Rest**, Schaefer atlas at 4 resolutions × 2 lenses (7 configurations).
- Reference configuration for all analyses: `100_riso` (100 ROI, Reciprocal-Isomap lens).
- Each graph node = a cluster of fMRI timepoints; each edge = temporal overlap between clusters.

## Repository structure

```
src/
  config.py          # all paths, constants and Mapper parameters (single source of truth)
  data_loader.py     # fMRI loading, VAD/3FA emotions, All50 annotations, frame building
  mapper_builder.py  # Mapper construction: lens, cover, clustering, node annotation, JSON export
  graph_metrics.py   # graph I/O + topology, k-core, betweenness, node categories, null models

scripts/
  run_pipeline.py            # batch: build all Mapper graphs (7 configs × 30 subjects × 14 films)
  5_1_emotional_profile.py   # emotional variability ranking + per-film VAD radars
  5_2a_global_topology.py    # global metrics, Rest vs Film (Wilcoxon)
  5_2b_power_law.py          # degree distribution, power-law fit, ER/BA null models
  5_3a_kcore_betweenness.py  # k-core & betweenness, Rest vs Film
  5_3b_community.py          # Louvain modularity & communities, ER/BA null, cross-resolution
  5_4_nodal_emotion.py       # node centrality vs emotional Activation/Complexity, node categories
  5_5_gpd_fingerprint.py     # Graph Portrait Divergence (within-subject vs within-film)
  fig_node_categories.py     # helper: render a Mapper graph coloured by node category

results/                     # generated CSVs and figures (git-ignored)
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (use: source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

Python 3.9+ recommended. Community detection uses `networkx>=3.0` (`louvain_communities`);
the power-law analysis (`5_2b`) requires `powerlaw`.

## Data

Raw data and the generated graphs are **not** included in the repository. The code expects
the paths defined in `src/config.py` (edit `PROJECT_ROOT` for your machine):

```
data/raw/3FA_films/3FA_13_<film>_stim.tsv   # continuous VAD emotion annotations
data/raw/emotion_ts_dict.pkl                # All50 (50-feature) emotion time series
data/raw/TC_emofilms/<resolution>_sub/      # fMRI BOLD time series per subject
0_graphs_html/<resolution>_<lens>/graphs/   # Mapper graphs as JSON (pipeline output)
```

The Emo-FilM dataset is described in Morgenroth et al. (2022). The analysis scripts read
the saved JSON graphs in `0_graphs_html/`; to reproduce the Rest-vs-Film comparison, the
Rest graphs must sit alongside the film graphs in `0_graphs_html/100_riso/graphs/`.

## Usage

Build all Mapper graphs from the raw fMRI (long-running, only needed once):

```bash
python scripts/run_pipeline.py
```

Run any analysis section independently (each loads the JSON graphs, caches a CSV, runs the
statistics and writes figures to `results/<section>/`):

```bash
python scripts/5_2a_global_topology.py
python scripts/5_3b_community.py
python scripts/5_4_nodal_emotion.py
python scripts/5_5_gpd_fingerprint.py
```

All metrics are recomputed from the JSON graphs via `graph_metrics`, so the repository is
self-contained and does not depend on any pre-computed spreadsheet.

## Author

**Davide Raciti** — Bachelor thesis, Mathematics for Engineering, Politecnico di Torino.
Supervisor: Andrea Santoro (ISI Foundation, Turin).

## References

- M. Saggar et al., *Towards a new approach to reveal dynamical organization of the brain
  using topological data analysis*, Nature Communications, 2018.
- Emo-FilM dataset (Morgenroth et al., 2022).
