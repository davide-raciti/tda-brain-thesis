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
  config.py          # all paths (incl. FIGURES_DIR output dir), constants, Mapper parameters
  data_loader.py     # fMRI loading, VAD/3FA emotions, All50 annotations, frame building
  mapper_builder.py  # Mapper construction: lens, cover, clustering, node annotation, JSON export
  graph_metrics.py   # graph I/O + topology, k-core, betweenness, node categories, null models

scripts/
  run_pipeline.py            # batch: build all Mapper graphs (7 configs × 30 subjects × 14 films)
  5_1_emotional_profile.py   # entropy ranking + per-film radar charts + emotion coherence
  5_2a_global_topology.py    # global metrics, Rest vs Film (Wilcoxon)
  5_2b_power_law.py          # degree distribution, power-law fit, ER/BA null models
  5_3a_kcore_betweenness.py  # k-core & betweenness, Rest vs Film
  5_3b_community.py          # Louvain modularity & communities, ER/BA null, cross-resolution
  5_4_nodal_emotion.py       # node centrality vs emotional Activation/Complexity, node categories
  5_5_gpd_fingerprint.py     # Graph Portrait Divergence (within-subject vs within-film)

  # figure generators — each writes a thesis image into FIGURES_DIR (see src/config.py):
  fig_5_1_radar_films.py            # Fig 5.1: three-panel affective-signature radars
  fig_degree_distribution_shapes.py # Ch 2: four candidate degree-distribution shapes (analytical)
  fig_null_models_triptych.py       # Ch 2: empirical Mapper graph vs ER vs BA null
  fig_graph_shape_time.py           # Ch 4: Mapper graph coloured by time (TR index)
  fig_node_categories.py            # Ch 1/4: Mapper graph coloured by node category
  fig_5_3_meso_ongraph.py           # Ch 5.3: k-core / betweenness / communities on one graph
  fig_5_3_meso_paired.py            # Ch 5.3: paired Rest vs Film mesoscale comparison
  fig_5_5_fingerprint_matrices.py   # Ch 5.5: subject×subject and film×film GPD matrices

results/                     # generated CSVs and figures, organised by section (5_1 ... 5_5)
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (use: source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

Python 3.9+ recommended. Community detection uses `networkx>=3.0` (`louvain_communities`);
the power-law analysis (`5_2b`) requires `powerlaw`. Building the Mapper graphs
(`src/mapper_builder.py`, `run_pipeline.py`) additionally requires `tda-mapper`
(module `tdamapper`) and `reciprocal-isomap` (module `reciprocal_isomap`); the analysis
and figure scripts, which read the saved JSON graphs, do not need them.

## Data

Raw data and the generated graphs are **not** included in the repository. The code expects
the paths defined in `src/config.py` (edit `PROJECT_ROOT`, and `FIGURES_DIR` to point the
figure scripts at your thesis image folder, for your machine):

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
Advisor (relatore): Francesco Vaccarino (Politecnico di Torino).
Co-advisor: Andrea Santoro (ISI Foundation, Turin).

## References

- M. Saggar et al., *Towards a new approach to reveal dynamical organization of the brain
  using topological data analysis*, Nature Communications, 2018.
- Emo-FilM dataset (Morgenroth et al., 2022).
