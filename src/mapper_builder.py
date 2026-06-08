import json
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from tdamapper.core import MapperAlgorithm
from tdamapper.cover import CubicalCover
from reciprocal_isomap import ReciprocalIsomap

from config import (
    COVER_OVERLAP, REF_DENSITY,
    CLUSTERER_K, CLUSTERER_P,
    RISOMAP_NEIGHBORS,
)
from data_loader import build_frame


# ─────────────────────────────────────────────────────────────────────────────
# MAPPER CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

# configuration of the clusterer used in the 3rd step of the Mapper algorithm
# the value of the cutoff is not arbitrary but depends on the local geometry of the data
def build_clusterer(X, k=CLUSTERER_K, p=CLUSTERER_P):
    """
    Compute the optimal distance threshold for agglomerative clustering
    following Saggar et al. 2018: fit a k-NN graph on X and take the
    p-th percentile of the k-th nearest-neighbour distances as threshold.

    Returns a fitted AgglomerativeClustering object (single linkage, Manhattan).
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='manhattan').fit(X)
    distances, _ = nbrs.kneighbors(X)
    threshold = np.percentile(distances[:, -1], p) # as suggested by Saggar
    return AgglomerativeClustering(
        n_clusters=None,
        linkage='single', # two clusters become one as soon as they share one item
        metric='manhattan',
        distance_threshold=threshold,
    )

# builds the mapper graph for given inputs of subject (or average actvity), film, lens and resolution
def build_mapper(X, lens='riso'):
    """
    Build the Mapper graph for a single (subject, film) fMRI matrix.

    Parameters
    ----------
    X    : ndarray (T, n_rois) — standardized fMRI time series
    lens : str — dimensionality reduction for the filter function:
                 'riso' (ReciprocalIsomap) or 'tsne' (t-SNE)

    Returns
    -------
    tda_graph : nx.Graph — Mapper graph with node attribute 'ids' (TR indices)
    """
    # number of cover intervals scales with film length (Saggar 2018)
    n_intervals = int(round(X.shape[0] / REF_DENSITY))
    cover = CubicalCover(n_intervals=n_intervals, overlap_frac=COVER_OVERLAP)
    clusterer = build_clusterer(X)
    mapper = MapperAlgorithm(cover=cover, clustering=clusterer)

    if lens == 'riso':
        X_lens = ReciprocalIsomap(n_neighbors=RISOMAP_NEIGHBORS).fit_transform(X)
    elif lens == 'tsne':
        X_lens = TSNE(n_components=2, init='pca', random_state=42).fit_transform(X)
    else:
        raise ValueError(f"Unknown lens '{lens}'. Choose 'riso' or 'tsne'.")

    return mapper.fit_transform(X, X_lens)


# ─────────────────────────────────────────────────────────────────────────────
# EMOTION ANNOTATION
# ─────────────────────────────────────────────────────────────────────────────

# annotates each node with the dominant emotional value computed in y
# for graphical purposes
def annotate_nodes(tda_graph, y):
    """
    Attach emotion proportions to each node in the Mapper graph.

    For each node, the members (TR indices) are looked up in y (the one-hot
    dominant-emotion DataFrame). The proportion of TRs belonging to each
    emotion category is stored as a node attribute 'proportions'.

    Parameters
    ----------
    tda_graph : nx.Graph — Mapper graph with node attribute 'ids'
    y         : DataFrame (T, 3) — one-hot dominant emotion per TR

    Returns
    -------
    tda_graph : nx.Graph — same graph with 'proportions' added to each node
    """
    emotion_cols = y.columns.tolist()

    for node_id in tda_graph.nodes():
        members = list(tda_graph.nodes[node_id].get('ids', []))
        if not members:
            tda_graph.nodes[node_id]['proportions'] = {col: 0.0 for col in emotion_cols}
            continue
        node_emotions = y.iloc[members]
        proportions = (node_emotions.sum() / len(members)).to_dict()
        tda_graph.nodes[node_id]['proportions'] = proportions

    return tda_graph


# ─────────────────────────────────────────────────────────────────────────────
# SERIALIZATION
# for further analysis, we will need the graph in a json format
# so that we can easily extract the metrics
# ─────────────────────────────────────────────────────────────────────────────

def to_json(tda_graph):
    """
    Serialize the annotated Mapper graph to a node-link dictionary
    compatible with downstream analysis scripts.

    Node fields : id, members, size, degree, proportions
    Link fields : source, target, value (overlap), strength (Jaccard)
    """
    node_members = {
        node_id: [int(m) for m in tda_graph.nodes[node_id].get('ids', [])]
        for node_id in tda_graph.nodes()
    }

    nodes = [
        {
            'id':          f'cube_{node_id}',
            'members':     node_members[node_id],
            'size':        len(node_members[node_id]),
            'degree':      tda_graph.degree(node_id),
            'proportions': tda_graph.nodes[node_id].get('proportions', {}),
        }
        for node_id in tda_graph.nodes()
    ]

    links = []
    for u, v in tda_graph.edges():
        overlap = len(set(node_members[u]) & set(node_members[v]))
        denom   = min(len(node_members[u]), len(node_members[v]))
        links.append({
            'source':   f'cube_{u}',
            'target':   f'cube_{v}',
            'value':    overlap,
            'strength': round(overlap / denom, 4) if denom > 0 else 0.0,
        })

    return {'nodes': nodes, 'links': links}


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

# main function: takes subject, film, resolution and lens as input 
# and returns networkx graph with emotional annotation and json graph for further analysis
def build_graph(subject_id, film_name, resolution='100', lens='riso'):
    """
    Full pipeline for one (subject, film) pair: load data, build Mapper,
    annotate nodes, return graph and serializable dict.

    Returns
    -------
    tda_graph : nx.Graph — annotated Mapper graph
    graph_json: dict     — node-link dict ready for json.dump()
    None if input data is missing.
    """
    frame = build_frame(subject_id, film_name, resolution)
    if frame is None:
        return None

    X = frame['fMRI_frame']
    y = frame['labels_frame']

    tda_graph  = build_mapper(X, lens=lens)
    tda_graph  = annotate_nodes(tda_graph, y)
    graph_json = to_json(tda_graph)

    return tda_graph, graph_json
