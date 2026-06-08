"""
run_pipeline.py
===============
Runs the full Mapper pipeline across all 7 configurations, subjects, and films.

Seven configurations (resolution x lens):
    100_riso | 100_tsne
    200_riso | 200_tsne
    400_riso | 400_tsne
    800_riso

For each (configuration, subject, film) triplet:
    1. Load fMRI and emotion data
    2. Build the Mapper graph
    3. Annotate nodes with dominant emotion proportions
    4. Save the annotated graph as JSON

Output structure (mirrors existing 0_graphs_html/):
    GRAPHS_DIR / {resolution}_{lens} / graphs / {film}_{resolution}_{lens}_{subject}.json
"""

import sys
import json
from pathlib import Path

# make src/ importable from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from config import GRAPHS_DIR, FILMS, SUBJECT_IDS

from mapper_builder import build_graph


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATIONS
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS = [
    ('100', 'riso'),
    ('100', 'tsne'),
    ('200', 'riso'),
    ('200', 'tsne'),
    ('400', 'riso'),
    ('400', 'tsne'),
    ('800', 'riso'),
]

ALL_SUBJECTS = SUBJECT_IDS + ['avg']


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    total_configs = len(CONFIGS)

    for cfg_idx, (resolution, lens) in enumerate(CONFIGS, 1):
        cfg_name = f"{resolution}_{lens}"
        out_dir  = GRAPHS_DIR / cfg_name / 'graphs'
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{cfg_idx}/{total_configs}] Configuration: {cfg_name}")
        print(f"  Output -> {out_dir}")
        print(f"  Subjects: {len(ALL_SUBJECTS)}  |  Films: {len(FILMS)}")

        n_total = len(ALL_SUBJECTS) * len(FILMS)
        n_done  = 0
        n_skip  = 0

        for subject_id in ALL_SUBJECTS:
            for film_name in FILMS:
                n_done += 1

                result = build_graph(subject_id, film_name, resolution, lens)

                if result is None:
                    n_skip += 1
                    continue

                tda_graph, graph_json = result

                fname = f"{film_name}_{resolution}_{lens}_{subject_id}.json"
                with open(out_dir / fname, 'w') as f:
                    json.dump(graph_json, f)

                print(f"  [{n_done:>4}/{n_total}] {subject_id} / {film_name} "
                      f"-> nodes={tda_graph.number_of_nodes()}  "
                      f"edges={tda_graph.number_of_edges()}")

        print(f"  Done. {n_done - n_skip} graphs saved, {n_skip} skipped.")

    print("\n=== Pipeline complete ===")


if __name__ == '__main__':
    main()
