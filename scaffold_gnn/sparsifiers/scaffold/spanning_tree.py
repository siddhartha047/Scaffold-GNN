"""Fast native spanning tree/forest builders for SCAFFOLD variants.

These helpers avoid ``networkx.minimum_spanning_tree`` /
``networkx.random_spanning_tree`` calls which are slow on large graphs. Each
helper operates directly on the input graph representation (either a
``networkx.Graph`` for the greedy/heap loops or the raw ``(src, dst)`` arrays
for the tensor fast path) using an in-place union-find implementation.

Every builder accepts an optional ``max_edges`` budget. When that budget is
smaller than a complete spanning forest, union-find stops immediately after
accepting the requested number of acyclic edges. The result is intentionally
a partial spanning forest; a full tree is never built and truncated later.

Six builders are exposed:

* :func:`build_spanning_forest_nx` -- return an ``nx.Graph`` spanning forest
  for the input ``nx.Graph``. Kruskal-style union-find over ``G.edges()`` with
  optional edge weights.
* :func:`build_spanning_forest_mask` -- return a boolean mask over canonical
  ``(src, dst)`` arrays selecting the edges that form a spanning forest.
* :func:`build_random_spanning_forest_nx` -- Kruskal on random edge priorities
  (fast randomized spanning forest for ``nx.Graph`` inputs).
* :func:`build_random_spanning_forest_mask` -- same idea on ``(src, dst)``
  arrays; used by the tensor SCAFFOLD-Fast backend.
* :func:`build_fast_weighted_spanning_forest_nx` -- the bucketed approximate
  weighted Kruskal implementation shared with Benchmark's ``fast-maxst`` and
  ``fast-mst`` selectors.
* :func:`build_fast_weighted_spanning_forest_mask` -- the same bucketed
  implementation over canonical ``(src, dst)`` arrays.

The exact and approximate weighted builders share the same compiled,
budget-aware union-find scan. Weight bucketing is parallelized with Numba;
the dependency-sensitive union-find scan stays serial so it remains
deterministic and correct.
"""

from __future__ import annotations

import math
import os
import numpy as np


try:
    import numba
    from numba import njit, prange
except Exception:  # pragma: no cover - fallback when numba is unavailable.
    numba = None
    njit = None
    prange = range


if njit is not None:

    @njit(cache=True, nogil=True)
    def _dsu_find(parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    @njit(cache=True, nogil=True)
    def _spanning_forest_mask_numba(num_nodes, src, dst, max_edges):
        parent = np.arange(num_nodes, dtype=np.int64)
        rank = np.zeros(num_nodes, dtype=np.int8)
        mask = np.zeros(src.shape[0], dtype=np.bool_)
        added = 0
        limit = min(max(0, int(max_edges)), max(0, int(num_nodes) - 1))
        if limit <= 0:
            return mask
        for i in range(src.shape[0]):
            u = int(src[i])
            v = int(dst[i])
            if u == v:
                continue
            ru = _dsu_find(parent, u)
            rv = _dsu_find(parent, v)
            if ru == rv:
                continue
            if rank[ru] < rank[rv]:
                parent[ru] = rv
            elif rank[ru] > rank[rv]:
                parent[rv] = ru
            else:
                parent[rv] = ru
                rank[ru] += 1
            mask[i] = True
            added += 1
            if added >= limit:
                break
        return mask

    @njit(cache=True, nogil=True, parallel=True)
    def _priority_buckets_numba(scores, low, high, bucket_count, maximum):
        """Parallel Benchmark-compatible weight quantization."""
        buckets = np.empty(scores.shape[0], dtype=np.int32)
        scale = float(bucket_count - 1) / max(float(high - low), 1e-20)
        for edge_id in prange(scores.shape[0]):
            bucket = int((float(scores[edge_id]) - low) * scale)
            bucket = max(0, min(bucket_count - 1, bucket))
            buckets[edge_id] = bucket_count - 1 - bucket if maximum else bucket
        return buckets

    @njit(cache=True, nogil=True)
    def _counting_order_numba(priority_buckets, bucket_count):
        counts = np.zeros(bucket_count, dtype=np.int64)
        for edge_id in range(priority_buckets.shape[0]):
            counts[int(priority_buckets[edge_id])] += 1
        offsets = np.empty(bucket_count, dtype=np.int64)
        position = 0
        for bucket in range(bucket_count):
            offsets[bucket] = position
            position += int(counts[bucket])
        cursors = offsets.copy()
        order = np.empty(priority_buckets.shape[0], dtype=np.int64)
        for edge_id in range(priority_buckets.shape[0]):
            bucket = int(priority_buckets[edge_id])
            order[cursors[bucket]] = edge_id
            cursors[bucket] += 1
        return order

    @njit(cache=True, nogil=True)
    def _spanning_forest_ids_numba(num_nodes, src, dst, order, max_edges):
        """Compiled, early-stopping Kruskal scan over an edge-id order."""
        parent = np.arange(num_nodes, dtype=np.int64)
        rank = np.zeros(num_nodes, dtype=np.int8)
        limit = min(
            max(0, int(max_edges)),
            max(0, int(num_nodes) - 1),
            src.shape[0],
        )
        if limit <= 0:
            return np.empty(0, dtype=np.int64)
        selected = np.empty(limit, dtype=np.int64)
        selected_count = 0
        for position in range(order.shape[0]):
            edge_id = int(order[position])
            u = int(src[edge_id])
            v = int(dst[edge_id])
            if u == v:
                continue
            root_u = _dsu_find(parent, u)
            root_v = _dsu_find(parent, v)
            if root_u == root_v:
                continue
            if rank[root_u] < rank[root_v]:
                parent[root_u] = root_v
            elif rank[root_u] > rank[root_v]:
                parent[root_v] = root_u
            else:
                parent[root_v] = root_u
                rank[root_u] += 1
            selected[selected_count] = edge_id
            selected_count += 1
            if selected_count >= limit:
                break
        return selected[:selected_count]


FAST_TREE_DEFAULT_BUCKETS = 256
FAST_TREE_MAX_BUCKETS = 65_536
_PARALLEL_BUCKET_THRESHOLD = 100_000


# Every backbone has a "forest" spelling alongside its historical "tree"
# spelling, and the two are the same thing. The construct really is a spanning
# FOREST -- one tree per component, which is what build_spanning_forest_mask
# returns -- and on an unweighted graph under support_weight_method=uniform the
# max/min distinction is vacuous anyway, so `sf` alone resolves to the default
# deterministic forest. The tree spelling stays canonical internally so no
# existing cell, cache key or .done marker changes meaning.
SUPPORT_NAME_ALIASES = {
    # historical
    "minst": "mst",
    "fast_minst": "fast_mst",
    # deterministic forest
    "maxsf": "maxst",
    "msf": "mst",
    "minsf": "mst",
    "fast_maxsf": "fast_maxst",
    "fast_msf": "fast_mst",
    "fast_minsf": "fast_mst",
    # random forest
    "randsf": "randst",
    "fast_randsf": "fast_randst",
    # low-stretch and friends
    "slsf": "slst",
    "glsf": "glst",
    "llsf": "llst",
    "randspf": "randspt",
    # bare names resolve to the default deterministic forest
    "sf": "maxst",
    "st": "maxst",
}

# Sample keeps hyphenated compound names and compares them literally, so it
# needs a map that preserves hyphens rather than going through the
# underscore-normalizing path above.
SAMPLE_BACKBONE_ALIASES = {
    "fixed-maxsf": "fixed-maxst",
    "fixed-msf": "fixed-maxst",
    "fixed-slsf": "fixed-slst",
    "rotate-randsf": "rotate-randst",
    "fixed-sf": "fixed-maxst",
    "fixed-st": "fixed-maxst",
}


def canonical_support_name(name):
    """Normalize CLI aliases while retaining existing internal names."""
    if not isinstance(name, str):
        return name
    normalized = name.strip().lower().replace("-", "_")
    return SUPPORT_NAME_ALIASES.get(normalized, normalized)


def canonical_sample_backbone(name):
    """Normalize Sample's backbone aliases, keeping its hyphenated spelling."""
    if not isinstance(name, str):
        return name
    normalized = name.strip().lower().replace("_", "-")
    return SAMPLE_BACKBONE_ALIASES.get(normalized, normalized)


def _available_cpu_count():
    candidates = []
    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        affinity = None
    if affinity:
        candidates.append(len(affinity))
    if os.cpu_count():
        candidates.append(int(os.cpu_count()))
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "LSB_DJOB_NUMPROC", "NSLOTS"):
        raw_value = os.environ.get(name)
        if not raw_value:
            continue
        try:
            value = int(raw_value)
        except ValueError:
            continue
        if value > 0:
            candidates.append(value)
    return max(1, min(candidates)) if candidates else 1


def _configure_numba_threads(parallel_workers=None):
    """Respect scheduler/affinity limits when running the parallel bucket kernel."""
    if numba is None:
        return 1
    workers = _available_cpu_count()
    if parallel_workers is not None and int(parallel_workers) > 0:
        workers = min(workers, int(parallel_workers))
    workers = max(1, min(workers, int(numba.config.NUMBA_NUM_THREADS)))
    try:
        numba.set_num_threads(workers)
    except (RuntimeError, ValueError):
        pass
    return workers


def _validate_bucket_count(bucket_count):
    bucket_count = int(bucket_count)
    if not 2 <= bucket_count <= FAST_TREE_MAX_BUCKETS:
        raise ValueError(
            f"fast-tree bucket_count must be between 2 and {FAST_TREE_MAX_BUCKETS}"
        )
    return bucket_count


def resolve_edge_budget(num_edges, target_ratio=None, max_edges=None):
    """Resolve an optional edge cap without exceeding the input edge count."""
    edge_count = max(0, int(num_edges))
    if max_edges is not None:
        return min(edge_count, max(0, int(max_edges)))
    if target_ratio is None:
        return None
    ratio = max(0.0, min(1.0, float(target_ratio)))
    return min(edge_count, int(math.ceil(ratio * edge_count - 1e-12)))


def _spanning_forest_mask_python(num_nodes, src, dst, max_edges=None):
    parent = np.arange(int(num_nodes), dtype=np.int64)
    rank = np.zeros(int(num_nodes), dtype=np.int8)
    mask = np.zeros(src.shape[0], dtype=bool)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    added = 0
    requested = resolve_edge_budget(src.shape[0], max_edges=max_edges)
    limit = max(0, int(num_nodes) - 1) if requested is None else min(
        max(0, int(num_nodes) - 1), requested
    )
    if limit <= 0:
        return mask
    for i in range(src.shape[0]):
        u = int(src[i])
        v = int(dst[i])
        if u == v:
            continue
        ru = find(u)
        rv = find(v)
        if ru == rv:
            continue
        if rank[ru] < rank[rv]:
            parent[ru] = rv
        elif rank[ru] > rank[rv]:
            parent[rv] = ru
        else:
            parent[rv] = ru
            rank[ru] += 1
        mask[i] = True
        added += 1
        if added >= limit:
            break
    return mask


def _spanning_forest_ids_python(num_nodes, src, dst, order, max_edges):
    parent = np.arange(int(num_nodes), dtype=np.int64)
    rank = np.zeros(int(num_nodes), dtype=np.int8)
    limit = min(
        max(0, int(max_edges)),
        max(0, int(num_nodes) - 1),
        int(src.shape[0]),
    )
    selected = []

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for position in order:
        edge_id = int(position)
        u = int(src[edge_id])
        v = int(dst[edge_id])
        if u == v:
            continue
        root_u = find(u)
        root_v = find(v)
        if root_u == root_v:
            continue
        if rank[root_u] < rank[root_v]:
            parent[root_u] = root_v
        elif rank[root_u] > rank[root_v]:
            parent[root_v] = root_u
        else:
            parent[root_v] = root_u
            rank[root_u] += 1
        selected.append(edge_id)
        if len(selected) >= limit:
            break
    return np.asarray(selected, dtype=np.int64)


def _selected_ids_from_order(num_nodes, src, dst, order, max_edges):
    if int(max_edges) <= 0 or src.size == 0:
        return np.empty(0, dtype=np.int64)
    if njit is not None:
        return _spanning_forest_ids_numba(
            int(num_nodes), src, dst, order, int(max_edges)
        )
    return _spanning_forest_ids_python(
        int(num_nodes), src, dst, order, int(max_edges)
    )


def _mask_from_selected_ids(num_edges, selected_ids):
    mask = np.zeros(int(num_edges), dtype=bool)
    if selected_ids.size:
        mask[np.asarray(selected_ids, dtype=np.int64)] = True
    return mask


def _priority_buckets_numpy(scores, bucket_count, maximum):
    low = float(scores.min())
    high = float(scores.max())
    if high - low <= 1e-20:
        return np.zeros(scores.shape[0], dtype=np.int32)
    buckets = np.floor(
        (scores - low) * ((bucket_count - 1) / (high - low))
    ).clip(0, bucket_count - 1).astype(np.int32)
    return bucket_count - 1 - buckets if maximum else buckets


def build_fast_weighted_spanning_forest_mask(
    num_nodes,
    src,
    dst,
    scores=None,
    *,
    maximum,
    bucket_count=FAST_TREE_DEFAULT_BUCKETS,
    max_edges=None,
    parallel_workers=None,
):
    """Benchmark-style approximate weighted forest with strict early stopping.

    Scores are quantized into ``bucket_count`` stable priority buckets and
    scanned with compiled Kruskal union-find. Edges inside a bucket retain
    input order. When ``scores`` is ``None`` or constant, this becomes the
    fastest deterministic unweighted spanning-forest scan.
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    if src.shape != dst.shape:
        raise ValueError("src and dst must have identical shapes")
    edge_count = int(src.shape[0])
    requested = resolve_edge_budget(edge_count, max_edges=max_edges)
    limit = max(0, int(num_nodes) - 1) if requested is None else min(
        max(0, int(num_nodes) - 1), requested
    )
    if edge_count == 0 or limit <= 0:
        return np.zeros(edge_count, dtype=bool)

    bucket_count = _validate_bucket_count(bucket_count)
    if scores is None:
        order = np.arange(edge_count, dtype=np.int64)
    else:
        scores = np.asarray(scores, dtype=np.float32)
        if scores.shape != src.shape:
            raise ValueError("scores must contain one value per edge")
        low = float(scores.min())
        high = float(scores.max())
        if high - low <= 1e-20:
            order = np.arange(edge_count, dtype=np.int64)
        else:
            if njit is not None and edge_count >= _PARALLEL_BUCKET_THRESHOLD:
                _configure_numba_threads(parallel_workers)
                buckets = _priority_buckets_numba(
                    scores,
                    low,
                    high,
                    bucket_count,
                    bool(maximum),
                )
            else:
                buckets = _priority_buckets_numpy(
                    scores,
                    bucket_count,
                    bool(maximum),
                )
            if njit is not None:
                order = _counting_order_numba(buckets, bucket_count)
            else:
                order = np.argsort(buckets, kind="stable").astype(
                    np.int64, copy=False
                )

    selected = _selected_ids_from_order(
        int(num_nodes), src, dst, order, int(limit)
    )
    return _mask_from_selected_ids(edge_count, selected)


def build_weighted_spanning_forest_mask(
    num_nodes,
    src,
    dst,
    scores,
    *,
    maximum,
    max_edges=None,
):
    """Exact stable weighted Kruskal mask for canonical edge arrays."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if src.shape != dst.shape or scores.shape != src.shape:
        raise ValueError("src, dst, and scores must have identical shapes")
    requested = resolve_edge_budget(src.shape[0], max_edges=max_edges)
    limit = max(0, int(num_nodes) - 1) if requested is None else min(
        max(0, int(num_nodes) - 1), requested
    )
    if src.size == 0 or limit <= 0:
        return np.zeros(src.shape[0], dtype=bool)
    priority = -scores if bool(maximum) else scores
    order = np.argsort(priority, kind="stable").astype(np.int64, copy=False)
    selected = _selected_ids_from_order(
        int(num_nodes), src, dst, order, int(limit)
    )
    return _mask_from_selected_ids(src.shape[0], selected)


def _networkx_edge_arrays(G, weight_key=None):
    nodes = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    edge_rows = list(G.edges(data=True))
    src = np.fromiter(
        (node_to_idx[u] for u, _, _ in edge_rows),
        dtype=np.int64,
        count=len(edge_rows),
    )
    dst = np.fromiter(
        (node_to_idx[v] for _, v, _ in edge_rows),
        dtype=np.int64,
        count=len(edge_rows),
    )
    scores = None
    if weight_key is not None:
        scores = np.fromiter(
            (float(data.get(weight_key, 1.0)) for _, _, data in edge_rows),
            dtype=np.float64,
            count=len(edge_rows),
        )
    return nodes, edge_rows, src, dst, scores


def _networkx_forest_from_mask(G, nodes, edge_rows, mask):
    import networkx as nx

    forest = nx.Graph()
    forest.add_nodes_from(G.nodes(data=True))
    for edge_id in np.flatnonzero(mask):
        u, v, data = edge_rows[int(edge_id)]
        forest.add_edge(u, v, **(data or {}))
    return forest


def build_spanning_forest_nx(G, weight_key=None, maximum=False, max_edges=None):
    """Return an ``nx.Graph`` spanning forest built directly from ``G``.

    Parameters
    ----------
    G : ``networkx.Graph``
    weight_key : str or ``None``
        If given, sort edges by ``G[u][v][weight_key]`` ascending (or
        descending when ``maximum=True``); otherwise use insertion order,
        matching the deterministic Kruskal-on-hop-count spanning forest.
    maximum : bool
        If True, prefer heavier edges.
    max_edges : int or ``None``
        Stop after accepting this many acyclic edges. ``None`` builds the
        complete spanning forest.
    """
    nodes, edge_rows, src, dst, scores = _networkx_edge_arrays(
        G, weight_key=weight_key
    )
    requested = resolve_edge_budget(G.number_of_edges(), max_edges=max_edges)
    limit = max(0, len(nodes) - 1) if requested is None else min(
        max(0, len(nodes) - 1), requested
    )
    if limit <= 0:
        return _networkx_forest_from_mask(
            G, nodes, edge_rows, np.zeros(len(edge_rows), dtype=bool)
        )
    if scores is None:
        order = np.arange(len(edge_rows), dtype=np.int64)
    else:
        sort_scores = -scores if maximum else scores
        order = np.argsort(sort_scores, kind="stable").astype(np.int64, copy=False)
    selected = _selected_ids_from_order(len(nodes), src, dst, order, limit)
    return _networkx_forest_from_mask(
        G, nodes, edge_rows, _mask_from_selected_ids(len(edge_rows), selected)
    )


def build_fast_weighted_spanning_forest_nx(
    G,
    weight_key=None,
    *,
    maximum=False,
    bucket_count=FAST_TREE_DEFAULT_BUCKETS,
    max_edges=None,
    parallel_workers=None,
):
    """NetworkX adapter for the Benchmark fast approximate tree kernel."""
    nodes, edge_rows, src, dst, scores = _networkx_edge_arrays(
        G, weight_key=weight_key
    )
    mask = build_fast_weighted_spanning_forest_mask(
        len(nodes),
        src,
        dst,
        scores,
        maximum=maximum,
        bucket_count=bucket_count,
        max_edges=max_edges,
        parallel_workers=parallel_workers,
    )
    return _networkx_forest_from_mask(G, nodes, edge_rows, mask)


def build_spanning_forest_mask(num_nodes, src, dst, max_edges=None):
    """Numpy union-find spanning forest mask on canonical ``(src, dst)`` arrays.

    Uses the numba-compiled union-find when numba is importable, falling back
    to the pure-numpy implementation otherwise.
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    requested = resolve_edge_budget(src.shape[0], max_edges=max_edges)
    limit = max(0, int(num_nodes) - 1) if requested is None else min(
        max(0, int(num_nodes) - 1), requested
    )
    if njit is not None:
        return _spanning_forest_mask_numba(int(num_nodes), src, dst, int(limit))
    return _spanning_forest_mask_python(int(num_nodes), src, dst, max_edges=limit)


def build_random_spanning_forest_nx(G, seed=None, max_edges=None):
    """Return an ``nx.Graph`` random spanning forest built directly from ``G``.

    This is Kruskal's algorithm on random edge priorities: each edge draws a
    uniform priority, edges are processed in priority order, and the standard
    union-find accepts the ``|V| - cc(G)`` edges that connect new components.
    This produces the classical "random-weight MST" distribution over spanning
    forests -- structurally close to Wilson's uniform sample for most graphs
    but orders of magnitude cheaper (``O(m log m)`` vs. mean-commute-time).

    Parameters
    ----------
    G : ``networkx.Graph``
    seed : int or ``None``
        Seed for the priority draw. ``None`` uses fresh entropy.
    max_edges : int or ``None``
        Stop after accepting this many acyclic edges.
    """
    import networkx as nx

    nodes = list(G.nodes())
    num_nodes = len(nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    edges = list(G.edges())
    m = len(edges)
    rng = np.random.default_rng(seed)
    priorities = rng.random(m)
    order = np.argsort(priorities, kind="stable")

    parent = list(range(num_nodes))
    rank = [0] * num_nodes

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    T = nx.Graph()
    T.add_nodes_from(G.nodes(data=True))
    added = 0
    requested = resolve_edge_budget(m, max_edges=max_edges)
    limit = max(0, num_nodes - 1) if requested is None else min(
        max(0, num_nodes - 1), requested
    )
    if limit <= 0:
        return T
    for pos in order:
        u, v = edges[int(pos)]
        if u == v:
            continue
        ru = find(node_to_idx[u])
        rv = find(node_to_idx[v])
        if ru == rv:
            continue
        if rank[ru] < rank[rv]:
            parent[ru] = rv
        elif rank[ru] > rank[rv]:
            parent[rv] = ru
        else:
            parent[rv] = ru
            rank[ru] += 1
        edge_data = G.get_edge_data(u, v) or {}
        T.add_edge(u, v, **edge_data)
        added += 1
        if added >= limit:
            break
    return T


def build_random_spanning_forest_mask(num_nodes, src, dst, seed=None, max_edges=None):
    """Random-priority Kruskal spanning forest mask over ``(src, dst)`` arrays.

    Draws a uniform priority per edge and reuses the same numba (or python
    fallback) union-find used by :func:`build_spanning_forest_mask`. Each call
    with a distinct ``seed`` produces an independent sample from the
    random-weight MST distribution.
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    m = int(src.shape[0])
    rng = np.random.default_rng(seed)
    order = np.argsort(rng.random(m), kind="stable").astype(np.int64, copy=False)
    src_perm = src[order]
    dst_perm = dst[order]
    requested = resolve_edge_budget(m, max_edges=max_edges)
    limit = max(0, int(num_nodes) - 1) if requested is None else min(
        max(0, int(num_nodes) - 1), requested
    )
    if njit is not None:
        mask_perm = _spanning_forest_mask_numba(
            int(num_nodes), src_perm, dst_perm, int(limit)
        )
    else:
        mask_perm = _spanning_forest_mask_python(
            int(num_nodes), src_perm, dst_perm, max_edges=limit
        )
    mask = np.zeros(m, dtype=bool)
    mask[order] = mask_perm
    return mask
