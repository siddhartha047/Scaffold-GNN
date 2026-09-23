"""Clustering helpers for clustered joint sparsifiers."""

from collections import deque
import hashlib
import inspect
import json
from pathlib import Path

import networkx as nx


CLUSTER_METHODS = (
    "bfs",
    "louvain",
    "greedy",
    "metis",
    "label",
    "node_prefix",
)


def _sort_key(node):
    return repr(node)


def _sorted_nodes(G):
    return sorted(G.nodes(), key=_sort_key)


def communities_to_node_map(communities, G):
    node_to_cluster = {}
    for cluster_id, community in enumerate(communities):
        for node in community:
            node_to_cluster[node] = cluster_id

    fallback = 0
    for node in G.nodes():
        if node not in node_to_cluster:
            node_to_cluster[node] = fallback
            fallback += 1
    return node_to_cluster


def build_bfs_clusters(G, cluster_count):
    nodes = _sorted_nodes(G)
    if not nodes:
        return {}

    cluster_count = min(max(1, int(cluster_count)), len(nodes))
    seeds = sorted(nodes, key=lambda node: (-G.degree(node), _sort_key(node)))[:cluster_count]
    node_to_cluster = {}
    queue = deque()

    for cluster_id, seed in enumerate(seeds):
        node_to_cluster[seed] = cluster_id
        queue.append(seed)

    while queue:
        node = queue.popleft()
        cluster_id = node_to_cluster[node]
        for nbr in sorted(G.neighbors(node), key=_sort_key):
            if nbr in node_to_cluster:
                continue
            node_to_cluster[nbr] = cluster_id
            queue.append(nbr)

    for idx, node in enumerate(nodes):
        if node not in node_to_cluster:
            node_to_cluster[node] = idx % cluster_count

    return node_to_cluster


def build_louvain_clusters(G, cluster_count, seed=None, weight_key=None):
    communities = nx.algorithms.community.louvain_communities(
        G,
        weight=weight_key,
        seed=seed,
    )
    return communities_to_node_map(communities, G)


def build_greedy_clusters(G, cluster_count, weight_key=None):
    communities = nx.algorithms.community.greedy_modularity_communities(
        G,
        weight=weight_key,
        cutoff=1,
        best_n=max(1, int(cluster_count)),
    )
    return communities_to_node_map(communities, G)


def build_label_clusters(G):
    labels = {}
    next_label = 0
    node_to_cluster = {}
    for node, data in G.nodes(data=True):
        label = None
        for key in ("cluster", "community", "label", "y"):
            if key in data:
                label = data[key]
                break
        if label is None:
            raise ValueError(
                "cluster_method='label' requires a node attribute named one of "
                "'cluster', 'community', 'label', or 'y'."
            )
        if label not in labels:
            labels[label] = next_label
            next_label += 1
        node_to_cluster[node] = labels[label]
    return node_to_cluster


def build_node_prefix_clusters(G):
    labels = {}
    next_label = 0
    node_to_cluster = {}
    for node in G.nodes():
        if not isinstance(node, (tuple, list)) or not node:
            raise ValueError(
                "cluster_method='node_prefix' requires tuple/list node ids, "
                "where node[0] is the cluster id."
            )
        label = node[0]
        if label not in labels:
            labels[label] = next_label
            next_label += 1
        node_to_cluster[node] = labels[label]
    return node_to_cluster


def _canon_edge_for_hash(u, v):
    u_key = repr(u)
    v_key = repr(v)
    return (u_key, v_key) if u_key <= v_key else (v_key, u_key)


def _graph_fingerprint(G, cluster_count):
    digest = hashlib.blake2b(digest_size=16)
    digest.update(f"n={G.number_of_nodes()}|m={G.number_of_edges()}|k={cluster_count}".encode())
    for node in _sorted_nodes(G):
        digest.update(b"n:")
        digest.update(repr(node).encode())
        digest.update(b"\n")
    for u_key, v_key in sorted(_canon_edge_for_hash(u, v) for u, v in G.edges()):
        digest.update(b"e:")
        digest.update(u_key.encode())
        digest.update(b"|")
        digest.update(v_key.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _metis_cache_path(G, cluster_count, cache_dir):
    if cache_dir is None:
        return None
    cache_root = Path(cache_dir)
    fingerprint = _graph_fingerprint(G, cluster_count)
    return cache_root / f"metis_parts_{cluster_count}_{fingerprint}.json"


def _load_cluster_cache(path, G, cluster_count):
    if path is None or not path.exists():
        return None
    nodes = _sorted_nodes(G)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("method") != "metis":
        return None
    if int(payload.get("num_parts", -1)) != int(cluster_count):
        return None
    if int(payload.get("num_nodes", -1)) != G.number_of_nodes():
        return None
    if int(payload.get("num_edges", -1)) != G.number_of_edges():
        return None
    assignments = payload.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(nodes):
        return None
    return {node: int(assignments[idx]) for idx, node in enumerate(nodes)}


def _save_cluster_cache(path, G, cluster_count, node_to_cluster):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes = _sorted_nodes(G)
    payload = {
        "method": "metis",
        "num_parts": int(cluster_count),
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "node_order": [repr(node) for node in nodes],
        "assignments": [int(node_to_cluster[node]) for node in nodes],
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    tmp_path.replace(path)


def build_metis_clusters(G, cluster_count, weight_key=None, cluster_cache_dir=None):
    """Build METIS partitions via PyG ClusterData.

    PyG's ``ClusterData`` is the component that invokes METIS. Newer PyG
    versions expose ``keep_inter_cluster_edges``; when present, pass it as
    ``True``. Older PyG versions do not expose that flag, but this helper only
    reads partition labels and never replaces ``G`` with ClusterData subgraphs,
    so inter-cluster edges remain available to the sparsifier.
    """
    cluster_count = max(1, int(cluster_count))
    if cluster_count <= 1 or G.number_of_nodes() <= 1:
        return {node: 0 for node in G.nodes()}

    cache_path = _metis_cache_path(G, cluster_count, cluster_cache_dir)
    cached = _load_cluster_cache(cache_path, G, cluster_count)
    if cached is not None:
        return cached

    try:
        import torch
        from torch_geometric.data import Data
        from torch_geometric.loader import ClusterData
        from torch_geometric.utils import to_undirected
    except Exception as exc:
        raise ImportError(
            "cluster_method='metis' requires torch_geometric with ClusterData."
        ) from exc

    nodes = _sorted_nodes(G)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    indexed_edges = []
    for u, v in G.edges():
        if u == v:
            continue
        indexed_edges.append((node_to_idx[u], node_to_idx[v]))

    if not indexed_edges:
        return {
            node: idx % min(cluster_count, len(nodes))
            for idx, node in enumerate(nodes)
        }

    edge_index = torch.tensor(indexed_edges, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index, num_nodes=len(nodes))
    data = Data(edge_index=edge_index, num_nodes=len(nodes))

    kwargs = {
        "num_parts": min(cluster_count, len(nodes)),
        "recursive": False,
        "save_dir": None,
        "log": False,
    }
    if "keep_inter_cluster_edges" in inspect.signature(ClusterData).parameters:
        kwargs["keep_inter_cluster_edges"] = True

    try:
        cluster_data = ClusterData(data, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            "PyG ClusterData METIS partitioning failed. Confirm that the "
            "active conda env has torch-sparse/METIS support installed."
        ) from exc

    node_to_cluster = {}
    perm = cluster_data.perm.cpu().tolist()
    partptr = cluster_data.partptr.cpu().tolist()
    for cluster_id in range(len(partptr) - 1):
        for permuted_pos in range(partptr[cluster_id], partptr[cluster_id + 1]):
            original_idx = int(perm[permuted_pos])
            node_to_cluster[nodes[original_idx]] = cluster_id

    _save_cluster_cache(cache_path, G, cluster_count, node_to_cluster)
    return node_to_cluster


def build_node_clusters(
    G,
    method="bfs",
    cluster_count=8,
    seed=None,
    weight_key=None,
    cluster_cache_dir=None,
):
    method = str(method).lower()
    cluster_count = max(1, int(cluster_count))
    if cluster_count <= 1 or G.number_of_nodes() <= 1:
        return {node: 0 for node in G.nodes()}
    if method == "bfs":
        return build_bfs_clusters(G, cluster_count)
    if method == "louvain":
        try:
            return build_louvain_clusters(G, cluster_count, seed=seed, weight_key=weight_key)
        except Exception:
            return build_bfs_clusters(G, cluster_count)
    if method == "greedy":
        try:
            return build_greedy_clusters(G, cluster_count, weight_key=weight_key)
        except Exception:
            return build_bfs_clusters(G, cluster_count)
    if method == "metis":
        return build_metis_clusters(
            G,
            cluster_count,
            weight_key=weight_key,
            cluster_cache_dir=cluster_cache_dir,
        )
    if method == "label":
        return build_label_clusters(G)
    if method == "node_prefix":
        return build_node_prefix_clusters(G)
    raise ValueError(f"Unknown cluster_method: {method!r}. Available: {CLUSTER_METHODS}")


def assign_edges_to_clusters(edges, node_to_cluster):
    edge_cluster = {}
    edge_affinity = {}
    remaining_by_cluster = {}
    cluster_load = {}

    for cluster_id in sorted(set(node_to_cluster.values())):
        remaining_by_cluster[cluster_id] = set()
        cluster_load[cluster_id] = 0

    for edge in sorted(edges, key=repr):
        u, v = edge
        affinity = {
            node_to_cluster.get(u, 0),
            node_to_cluster.get(v, 0),
        }
        cluster_id = min(affinity, key=lambda cid: (cluster_load.get(cid, 0), cid))
        edge_cluster[edge] = cluster_id
        edge_affinity[edge] = affinity
        remaining_by_cluster.setdefault(cluster_id, set()).add(edge)
        cluster_load[cluster_id] = cluster_load.get(cluster_id, 0) + 1

    return edge_cluster, edge_affinity, remaining_by_cluster
