import json
import numpy as np
import networkx as nx


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_graph(path):
    """
    Load a Mapper graph from a JSON file produced by to_json().

    Returns
    -------
    dict with keys 'nodes' and 'links'
    """
    with open(path) as f:
        return json.load(f)


def json_to_networkx(graph_dict):
    """
    Convert a node-link JSON dict (as produced by to_json()) back to a
    NetworkX graph, restoring node attributes 'members' and 'proportions'.

    Returns
    -------
    nx.Graph
    """
    G = nx.Graph()

    for node in graph_dict['nodes']:
        G.add_node(
            node['id'],
            members=node['members'],
            size=node['size'],
            proportions=node.get('proportions', {}),
        )

    for link in graph_dict['links']:
        G.add_edge(
            link['source'],
            link['target'],
            value=link['value'],
            strength=link['strength'],
        )

    return G


# ─────────────────────────────────────────────────────────────────────────────
# TOPOLOGY
# ─────────────────────────────────────────────────────────────────────────────

def compute_topology(G):
    """
    Compute global topological metrics for a Mapper graph.

    Parameters
    ----------
    G : nx.Graph — accepts both NetworkX graphs and JSON dicts
        (JSON dicts are converted automatically)

    Returns
    -------
    dict with keys:
        num_nodes, num_edges, density, connected_components,
        nodes_in_giant_component, diameter_giant_component,
        mean_degree, max_degree, mean_closeness
    """
    if isinstance(G, dict):
        G = json_to_networkx(G)

    n = G.number_of_nodes()
    m = G.number_of_edges()
    degrees   = [d for _, d in G.degree()]
    closeness = nx.closeness_centrality(G)

    if n > 0:
        lcc   = max(nx.connected_components(G), key=len)
        g_sub = G.subgraph(lcc)
        lcc_n = g_sub.number_of_nodes()
        diam  = nx.diameter(g_sub)
    else:
        lcc_n, diam = 0, 0

    return {
        'num_nodes':                n,
        'num_edges':                m,
        'density':                  round(nx.density(G), 6),
        'connected_components':     nx.number_connected_components(G),
        'nodes_in_giant_component': lcc_n,
        'diameter_giant_component': diam,
        'mean_degree':              round(float(np.mean(degrees)), 4) if degrees else 0,
        'max_degree':               int(np.max(degrees)) if degrees else 0,
        'mean_closeness':           round(float(np.mean(list(closeness.values()))), 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MESO-SCALE METRICS  (shared by 5_3a and 5_3b)
# ─────────────────────────────────────────────────────────────────────────────

def gini(arr):
    """Gini coefficient of a non-negative array."""
    arr = np.sort(np.abs(arr))
    n   = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum()) / (n * arr.sum()) - (n + 1) / n)


def kcore_metrics(G):
    """
    K-core decomposition metrics for one graph.
    max_core       : depth of the innermost core
    mean_core      : mean core number across all nodes
    frac_innermost : fraction of nodes in the innermost core
    """
    if G.number_of_nodes() == 0:
        return dict(max_core=0, mean_core=0.0, frac_innermost=0.0)
    cores  = nx.core_number(G)
    values = np.array(list(cores.values()))
    k_max  = int(values.max())
    return dict(
        max_core       = k_max,
        mean_core      = round(float(values.mean()), 4),
        frac_innermost = round(float((values == k_max).sum()) / len(values), 4),
    )


def betweenness_metrics(G):
    """
    Betweenness centrality metrics for one graph.
    mean_betweenness : mean normalised betweenness across nodes
    max_betweenness  : maximum betweenness (most prominent bridge node)
    gini_betweenness : Gini coefficient (inequality of betweenness distribution)
    """
    if G.number_of_nodes() < 2:
        return dict(mean_betweenness=0.0, max_betweenness=0.0, gini_betweenness=0.0)
    bet    = nx.betweenness_centrality(G, normalized=True)
    values = np.array(list(bet.values()))
    return dict(
        mean_betweenness = round(float(values.mean()), 6),
        max_betweenness  = round(float(values.max()),  6),
        gini_betweenness = round(gini(values), 4),
    )


def node_category(coreness, betweenness):
    """
    Classify a node by its core membership and bridging role, crossing
    coreness (core member iff c >= 2) with normalised betweenness
    (bridge iff C^B > 0):

      Peripheral     : C^B = 0 and c <= 1   (no bridging, not in the core)
      Bridge         : C^B > 0 and c <= 1   (bridges, but outside the core)
      Connector Hub  : C^B > 0 and c >= 2   (bridges and sits in the dense core)
      Provincial Hub : C^B = 0 and c >= 2   (core member that bridges nothing,
                                             e.g. nodes inside cliques/triangles)
    """
    in_core = coreness >= 2
    bridges = betweenness > 0
    if not in_core and not bridges:
        return 'Peripheral'
    if not in_core and bridges:
        return 'Bridge'
    if in_core and bridges:
        return 'Connector Hub'
    return 'Provincial Hub'


# ─────────────────────────────────────────────────────────────────────────────
# NULL MODELS  (shared by 5_2b and 5_3b)
# ─────────────────────────────────────────────────────────────────────────────

def make_er(n, density, seed=None):
    """Erdős–Rényi graph G(n, p) with p = density."""
    return nx.erdos_renyi_graph(int(n), float(min(density, 1.0)), seed=seed)


def make_ba(n, n_edges, seed=None):
    """
    Barabási–Albert graph on n nodes whose edge count approximately matches
    n_edges. The attachment parameter m is chosen as round(n_edges / n), since
    a BA graph has about m*n edges; m is clamped to [1, n-1].
    """
    n = int(n)
    m = max(1, min(int(round(n_edges / n)), n - 1))
    return nx.barabasi_albert_graph(n, m, seed=seed)


def lcc_of(G):
    """Return the largest connected component of G as a new graph."""
    if G.number_of_nodes() == 0:
        return G
    nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(nodes).copy()
