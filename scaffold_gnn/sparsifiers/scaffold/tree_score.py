"""Exact SCAFFOLD candidate scoring against a fixed spanning forest.

Every term of the SCAFFOLD score is a path aggregate on a tree, and every path
aggregate on a tree is a root-prefix difference. That replaces the ``O(m * n)``
per-round shortest-path scan in :mod:`.scaffold_greedy` with an
``O(m log n + n)`` closed form whose values are *identical* to the Greedy
variant's round-1 numbers -- in round 1 the support graph is the tree, so
``d_H == d_T``.

Three traps are handled explicitly, because each produces plausible-looking
wrong numbers rather than a crash:

1. Candidates whose endpoints lie in different components have no tree path.
   ``scaffold_greedy`` skips them *before* accumulating congestion, so they must
   be excluded from ``diff`` / ``lcaCnt`` / ``degI`` and not merely flagged with
   an infinite dilation.
2. ``Sp`` indexes tree *edges* on ``root -> x`` (so ``Sp[root] = 0``) while
   ``Nq`` indexes *nodes* inclusively (so ``Nq[root] = (vCon[root]+eps)**q``).
   A single ``S[parent[x]] + term`` recurrence self-references at a root,
   because roots satisfy ``parent[root] == root``.
3. The subtree-sum loop must skip roots for the same reason, otherwise
   ``eCon[root] += eCon[root]`` silently doubles it.

The module is deliberately dependency-light -- numpy plus optional numba, no
torch and no torch_geometric -- so the precompute script and the equivalence
tests can import it without pulling in the training stack.

Reference: ``Brainstrom/others/notes/2026-08-17_plan_scaffold_sampling.md`` sections 2, 8.2.
"""

import math

import numpy as np

from .parallel_utils import auto_worker_count, parallel_threads

try:
    from numba import njit
    from numba import prange as _prange
except Exception:  # pragma: no cover - fallback for environments without numba.
    njit = None
    _prange = range

# Written as ``prange`` in the kernel bodies; degrades to the builtin ``range``
# without numba, so one source serves both backends.
prange = _prange


def _kernel(func):
    """JIT a kernel when numba is present, else keep the pure-Python body.

    The bodies are written in explicit-loop style so both paths are the same
    code. Without numba they stay correct but are only fast enough for the
    small graphs used by the equivalence tests.
    """
    if njit is None:
        return func
    return njit(cache=True, nogil=True)(func)


def _kernel_parallel(func):
    """As :func:`_kernel`, with numba's auto-parallelizer over ``prange``.

    Only for loops whose iterations write disjoint output slots and perform no
    cross-iteration float reduction, so the result is identical at any thread
    count. The count itself comes from
    :func:`..parallel_utils.parallel_threads`; one thread is a fine setting, and
    measured no slower than a separately compiled serial build.

    **Never decorate the same function with both** :func:`_kernel` **and this.**
    numba keys its on-disk cache on the function's code location, so the two
    compilations collide and one silently loads the other -- correct results, no
    speedup, no warning.
    """
    if njit is None:
        return func
    return njit(cache=True, nogil=True, parallel=True)(func)


# ----------------------------------------------------------------------
# union-find
# ----------------------------------------------------------------------
@_kernel
def _dsu_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@_kernel
def _forest_mask_kernel(num_nodes, src, dst, visit, max_edges):
    parent = np.arange(num_nodes)
    rank = np.zeros(num_nodes, dtype=np.int8)
    mask = np.zeros(src.shape[0], dtype=np.bool_)
    added = 0
    for j in range(visit.shape[0]):
        if added >= max_edges:
            break
        i = visit[j]
        u = src[i]
        v = dst[i]
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
    return mask


@_kernel
def _component_count_kernel(num_nodes, src, dst):
    parent = np.arange(num_nodes)
    components = num_nodes
    for i in range(src.shape[0]):
        ru = _dsu_find(parent, src[i])
        rv = _dsu_find(parent, dst[i])
        if ru != rv:
            parent[ru] = rv
            components -= 1
    return components


# ----------------------------------------------------------------------
# rooted forest index (parent / depth / root / binary lifting)
# ----------------------------------------------------------------------
@_kernel
def _tree_index_kernel(num_nodes, tree_src, tree_dst, levels):
    m = tree_src.shape[0]
    rowptr = np.zeros(num_nodes + 1, dtype=np.int64)
    for i in range(m):
        rowptr[tree_src[i] + 1] += 1
        rowptr[tree_dst[i] + 1] += 1
    for i in range(1, num_nodes + 1):
        rowptr[i] += rowptr[i - 1]
    nbr = np.empty(2 * m, dtype=np.int64)
    cursor = np.empty(num_nodes, dtype=np.int64)
    for i in range(num_nodes):
        cursor[i] = rowptr[i]
    for i in range(m):
        u = tree_src[i]
        v = tree_dst[i]
        nbr[cursor[u]] = v
        cursor[u] += 1
        nbr[cursor[v]] = u
        cursor[v] += 1

    parent = np.arange(num_nodes)
    depth = np.zeros(num_nodes, dtype=np.int32)
    root = np.arange(num_nodes)
    tin = np.zeros(num_nodes, dtype=np.int64)
    seen = np.zeros(num_nodes, dtype=np.bool_)
    stack = np.empty(num_nodes, dtype=np.int64)
    clock = 0

    for start in range(num_nodes):
        if seen[start]:
            continue
        seen[start] = True
        parent[start] = start
        depth[start] = 0
        root[start] = start
        top = 0
        stack[top] = start
        top += 1
        while top > 0:
            top -= 1
            node = stack[top]
            tin[node] = clock  # DFS preorder: the tree-locality key
            clock += 1
            for pos in range(rowptr[node], rowptr[node + 1]):
                nb = nbr[pos]
                if seen[nb]:
                    continue
                seen[nb] = True
                parent[nb] = node
                depth[nb] = depth[node] + 1
                root[nb] = start
                stack[top] = nb
                top += 1

    up = np.empty((levels, num_nodes), dtype=np.int64)
    for i in range(num_nodes):
        up[0, i] = parent[i]
    for lvl in range(1, levels):
        for i in range(num_nodes):
            up[lvl, i] = up[lvl - 1, up[lvl - 1, i]]
    return depth, root, up, parent, tin


@_kernel_parallel
def _lca_kernel(src, dst, depth, root, up):
    """One LCA per candidate pair. Iterations write only ``out[i]``."""
    n = src.shape[0]
    out = np.full(n, -1, dtype=np.int64)
    levels = up.shape[0]
    for i in prange(n):
        a = src[i]
        b = dst[i]
        if root[a] != root[b]:
            continue  # TRAP 1: -1 marks "no tree path"
        if depth[a] < depth[b]:
            tmp = a
            a = b
            b = tmp
        diff = depth[a] - depth[b]
        bit = 0
        while diff > 0:
            if diff & 1:
                a = up[bit, a]
            diff >>= 1
            bit += 1
        if a != b:
            for lvl in range(levels - 1, -1, -1):
                if up[lvl, a] != up[lvl, b]:
                    a = up[lvl, a]
                    b = up[lvl, b]
            a = up[0, a]
        out[i] = a
    return out


@_kernel_parallel
def _tree_terms_kernel(
    cu, cv, cand, lca, depth, sp, nq, vcon, weight, edge_p, node_q, eps
):
    """Dilation and both path-congestion norms, one candidate per iteration.

    Replaces a chain of NumPy expressions over length-|I| temporaries. Fusing
    them removes eight intermediate arrays and lets the loop parallelize; the
    arithmetic is unchanged, including the two counting traps -- eConPath
    averages over the path's ``path_len`` edges, vConPath over its
    ``path_len - 1`` internal nodes.
    """
    n = cu.shape[0]
    dil = np.empty(n, dtype=np.float64)
    econ_path = np.zeros(n, dtype=np.float64)
    vcon_path = np.zeros(n, dtype=np.float64)
    path_len = np.zeros(n, dtype=np.int64)
    inv_p = 1.0 / edge_p
    inv_q = 1.0 / node_q
    for i in prange(n):
        anc = lca[i]
        if anc < 0:
            dil[i] = np.inf  # TRAP 1: no tree path at all
            continue
        u = cu[i]
        v = cv[i]
        hops = depth[u] + depth[v] - 2 * depth[anc]
        path_len[i] = hops

        w = weight[cand[i]]
        if w < eps:
            w = eps
        dil[i] = hops / w

        se = sp[u] + sp[v] - 2.0 * sp[anc]
        if se < 0.0:
            se = 0.0
        econ_path[i] = (se / (hops + eps)) ** inv_p

        sv = (
            nq[u] + nq[v] - 2.0 * nq[anc]
            + (vcon[anc] + eps) ** node_q
            - (vcon[u] + eps) ** node_q
            - (vcon[v] + eps) ** node_q
        )
        if sv < 0.0:
            sv = 0.0
        vcon_path[i] = (sv / (hops - 1 + eps)) ** inv_q
    return dil, econ_path, vcon_path, path_len


@_kernel_parallel
def _tree_score_kernel(
    dil, econ_path, vcon_path, d_max, e_max, v_max, alpha, edge_beta, node_beta, eps
):
    """Normalize by the maxima and combine. The maxima arrive as scalars, so no
    reduction crosses iterations and the result cannot vary with thread count."""
    n = dil.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        if not np.isfinite(dil[i]):
            out[i] = np.inf
        else:
            out[i] = (
                ((dil[i] + eps) / (d_max + eps)) ** alpha
                * ((econ_path[i] + eps) / (e_max + eps)) ** edge_beta
                * ((vcon_path[i] + eps) / (v_max + eps)) ** node_beta
            )
    return out


# ----------------------------------------------------------------------
# congestion
# ----------------------------------------------------------------------
@_kernel
def _edge_congestion_kernel(src, dst, lca, parent, order, num_nodes):
    econ = np.zeros(num_nodes, dtype=np.int64)
    for i in range(src.shape[0]):
        l = lca[i]
        if l < 0:
            continue  # TRAP 1
        econ[src[i]] += 1
        econ[dst[i]] += 1
        econ[l] -= 2
    for idx in range(order.shape[0] - 1, -1, -1):  # deepest first
        x = order[idx]
        p = parent[x]
        if p != x:  # TRAP 3: never fold a root into itself
            econ[p] += econ[x]
    return econ


@_kernel
def _node_congestion_kernel(src, dst, lca, econ):
    """vCon[x] = eCon[x] + lcaCnt[x] - degI[x].

    A candidate path touches ``x`` in exactly one of two disjoint ways: it uses
    the tree edge ``(x, parent[x])`` -- counted by ``eCon`` and implying the LCA
    is strictly above ``x`` -- or ``x`` *is* the LCA. The reference counts only
    ``path[1:-1]``, so paths with ``x`` as an endpoint are subtracted.
    """
    vcon = econ.copy()
    for i in range(src.shape[0]):
        l = lca[i]
        if l < 0:
            continue  # TRAP 1
        vcon[l] += 1
        vcon[src[i]] -= 1
        vcon[dst[i]] -= 1
    return vcon


@_kernel
def _root_prefix_kernel(values, parent, order, power, eps, include_root):
    n = order.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for idx in range(n):  # ascending depth: parents precede children
        x = order[idx]
        term = (values[x] + eps) ** power
        p = parent[x]
        if p == x:  # TRAP 2
            out[x] = term if include_root else 0.0
        else:
            out[x] = out[p] + term
    return out


# ----------------------------------------------------------------------
# public helpers
# ----------------------------------------------------------------------
def spanning_forest_mask(num_nodes, src, dst, visit_order, max_edges=None):
    """Kruskal over ``visit_order``; returns a boolean mask over ``src``."""
    num_nodes = int(num_nodes)
    src = np.ascontiguousarray(src, dtype=np.int64)
    dst = np.ascontiguousarray(dst, dtype=np.int64)
    visit_order = np.ascontiguousarray(visit_order, dtype=np.int64)
    if src.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    limit = num_nodes if max_edges is None else int(max_edges)
    limit = max(0, min(limit, num_nodes))
    if limit == 0:
        return np.zeros(src.shape[0], dtype=bool)
    return _forest_mask_kernel(num_nodes, src, dst, visit_order, limit)


def random_spanning_forest_mask(num_nodes, src, dst, seed=None, max_edges=None):
    """Uniform random edge priority + Kruskal = a random spanning forest."""
    rng = np.random.default_rng(seed)
    visit = rng.permutation(int(np.asarray(src).shape[0]))
    return spanning_forest_mask(num_nodes, src, dst, visit, max_edges=max_edges)


def weighted_spanning_forest_mask(
    num_nodes, src, dst, weight=None, maximum=True, max_edges=None
):
    """Deterministic MaxST/MST forest. Ties break on edge id, so it is stable."""
    m = int(np.asarray(src).shape[0])
    if weight is None:
        visit = np.arange(m, dtype=np.int64)
    else:
        weight = np.asarray(weight, dtype=np.float64)
        keys = -weight if maximum else weight
        visit = np.argsort(keys, kind="stable").astype(np.int64, copy=False)
    return spanning_forest_mask(num_nodes, src, dst, visit, max_edges=max_edges)


def component_count(num_nodes, src, dst):
    src = np.ascontiguousarray(src, dtype=np.int64)
    dst = np.ascontiguousarray(dst, dtype=np.int64)
    if src.shape[0] == 0:
        return int(num_nodes)
    return int(_component_count_kernel(int(num_nodes), src, dst))


def build_tree_index(num_nodes, tree_src, tree_dst):
    """Return ``(depth, root, up, parent, tin)`` for a rooted spanning forest.

    ``tin`` is the DFS preorder index, used as the tree-locality sort key for
    systematic pi-ps sampling.
    """
    num_nodes = int(num_nodes)
    tree_src = np.ascontiguousarray(tree_src, dtype=np.int64)
    tree_dst = np.ascontiguousarray(tree_dst, dtype=np.int64)
    levels = max(1, int(math.ceil(math.log2(max(2, num_nodes)))) + 1)
    return _tree_index_kernel(num_nodes, tree_src, tree_dst, levels)


def depth_order(depth):
    """Node ids sorted by ascending depth (a valid topological order)."""
    depth = np.asarray(depth)
    return np.argsort(depth, kind="stable").astype(np.int64, copy=False)


def parent_edge_weights(depth, tree_src, tree_dst, tree_weight, num_nodes):
    """Weight of the tree edge joining each node to its parent (roots get 0).

    For a tree edge the deeper endpoint is the child, so the assignment is
    unambiguous. Used to build weighted root depths for weighted graphs.
    """
    tree_src = np.asarray(tree_src, dtype=np.int64)
    tree_dst = np.asarray(tree_dst, dtype=np.int64)
    tree_weight = np.asarray(tree_weight, dtype=np.float64)
    out = np.zeros(int(num_nodes), dtype=np.float64)
    if tree_src.size:
        deeper = np.where(depth[tree_src] > depth[tree_dst], tree_src, tree_dst)
        out[deeper] = tree_weight
    return out


def tree_lca(src, dst, depth, root, up):
    """LCA per pair; ``-1`` when the endpoints are in different components."""
    src = np.ascontiguousarray(src, dtype=np.int64)
    dst = np.ascontiguousarray(dst, dtype=np.int64)
    if src.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    return _lca_kernel(src, dst, depth, root, up)


# ----------------------------------------------------------------------
# the scorer
# ----------------------------------------------------------------------
def scaffold_tree_scores(
    num_nodes,
    src,
    dst,
    tree_mask,
    weight=None,
    *,
    alpha=1.0,
    edge_beta=1.0,
    node_beta=1.0,
    edge_norm_p=2.0,
    node_norm_q=2.0,
    eps=1e-8,
    tree_index=None,
    weighted_paths=False,
    candidate_mask=None,
    workers=None,
):
    """Score every non-tree edge exactly, in ``O(m log n + n)``.

    ``src`` / ``dst`` are the canonical undirected edge list and ``tree_mask``
    selects the spanning forest. ``candidate_mask`` optionally restricts the
    congestion population to a sampled batch; by default every non-tree edge
    is a candidate. All returned arrays have length ``m``; excluded entries
    are zero and omitted from every statistic, matching
    ``scaffold_greedy``, which only ever scores ``E \\ E(H)``.

    Disconnected candidates get ``dil = score = inf`` and are reported through
    the ``mandatory`` mask; they contribute nothing to any congestion counter.

    ``weight`` is always the denominator of the dilation, i.e. ``w_G(e)``.
    ``weighted_paths`` selects the *numerator*: ``False`` (the default, and what
    the tensor backend's ``_tree_distances`` uses) counts hops, ``True`` sums
    tree edge weights, matching ``scaffold_greedy`` on a weighted graph.

    ``workers`` parallelizes the LCA queries and the per-candidate term
    assembly, which together dominate the runtime. The congestion scatter and
    the root-prefix recurrences stay serial: they are order-dependent, and were
    measured at about 5% of the total. Scores are identical at any worker
    count. ``None`` resolves via ``SCAFFOLD_NUM_WORKERS`` / ``OMP_NUM_THREADS``,
    defaulting to ``min(8, cpus)``.
    """
    for order_value, name in ((edge_norm_p, "edge_norm_p"), (node_norm_q, "node_norm_q")):
        if math.isinf(order_value):
            raise NotImplementedError(
                f"{name}=inf needs a max-on-path query rather than root-prefix sums; "
                "see Brainstrom/others/notes/2026-08-17_plan_scaffold_sampling.md section 8.2.1 trap 4. "
                "Finite p/q (the 2.0 default) is supported."
            )
        if order_value <= 0:
            raise ValueError(f"{name} must be positive, got {order_value}")

    num_nodes = int(num_nodes)
    src = np.ascontiguousarray(src, dtype=np.int64)
    dst = np.ascontiguousarray(dst, dtype=np.int64)
    tree_mask = np.ascontiguousarray(tree_mask, dtype=bool)
    m = int(src.shape[0])
    if weight is None:
        weight = np.ones(m, dtype=np.float64)
    else:
        weight = np.ascontiguousarray(weight, dtype=np.float64)

    workers = auto_worker_count(0 if workers is None else workers)

    if tree_index is None:
        tree_index = build_tree_index(num_nodes, src[tree_mask], dst[tree_mask])
    depth, root, up, parent = tree_index[0], tree_index[1], tree_index[2], tree_index[3]
    order = depth_order(depth)

    if candidate_mask is None:
        cand = ~tree_mask
    else:
        candidate_mask = np.ascontiguousarray(candidate_mask, dtype=bool)
        if candidate_mask.shape != tree_mask.shape:
            raise ValueError(
                "candidate_mask must have the same shape as tree_mask"
            )
        cand = candidate_mask & ~tree_mask
    cand_idx = np.flatnonzero(cand).astype(np.int64, copy=False)
    cu = src[cand_idx]
    cv = dst[cand_idx]

    with parallel_threads(workers):
        lca = tree_lca(cu, cv, depth, root, up)
    connected = lca >= 0

    dil = np.zeros(m, dtype=np.float64)
    length = np.zeros(m, dtype=np.int64)
    econ_path = np.zeros(m, dtype=np.float64)
    vcon_path = np.zeros(m, dtype=np.float64)
    score = np.zeros(m, dtype=np.float64)
    mandatory = np.zeros(m, dtype=bool)

    if cand_idx.size == 0:
        return {
            "dil": dil, "length": length, "econ_path": econ_path,
            "vcon_path": vcon_path, "score": score, "mandatory": mandatory,
            "candidate_mask": cand, "edge_congestion": np.zeros(num_nodes, np.int64),
            "node_congestion": np.zeros(num_nodes, np.int64), "total_stretch": 0.0,
        }

    # Congestion counters see connected candidates only (TRAP 1) -- which both
    # kernels enforce by skipping ``lca < 0``, so the candidate arrays go in
    # whole rather than being compacted first.
    econ = _edge_congestion_kernel(cu, cv, lca, parent, order, num_nodes)
    vcon = _node_congestion_kernel(cu, cv, lca, econ)

    sp = _root_prefix_kernel(econ, parent, order, float(edge_norm_p), float(eps), False)
    nq = _root_prefix_kernel(vcon, parent, order, float(node_norm_q), float(eps), True)

    conn_idx = cand_idx[connected]
    disc_idx = cand_idx[~connected]

    # One fused pass: hop counts, dilation, and both path-congestion norms.
    with parallel_threads(workers):
        c_dil, c_econ, c_vcon, c_len = _tree_terms_kernel(
            np.ascontiguousarray(cu, dtype=np.int64),
            np.ascontiguousarray(cv, dtype=np.int64),
            np.ascontiguousarray(cand_idx, dtype=np.int64),
            np.ascontiguousarray(lca, dtype=np.int64),
            np.ascontiguousarray(depth, dtype=np.int64),
            np.ascontiguousarray(sp, dtype=np.float64),
            np.ascontiguousarray(nq, dtype=np.float64),
            np.ascontiguousarray(vcon, dtype=np.float64),
            np.ascontiguousarray(weight, dtype=np.float64),
            float(edge_norm_p), float(node_norm_q), float(eps),
        )

    if weighted_paths:
        # The kernel's dilation counts hops; here the numerator is a sum of
        # tree edge weights instead. The denominator, w_G(e), is unchanged.
        pw = parent_edge_weights(
            depth, src[tree_mask], dst[tree_mask], weight[tree_mask], num_nodes
        )
        wdepth = _root_prefix_kernel(pw, parent, order, 1.0, 0.0, False)
        path_dist = (
            wdepth[cu[connected]]
            + wdepth[cv[connected]]
            - 2.0 * wdepth[lca[connected]]
        )
        c_dil[connected] = path_dist / np.maximum(weight[conn_idx], eps)

    length[cand_idx] = c_len
    dil[cand_idx] = c_dil
    econ_path[cand_idx] = c_econ
    vcon_path[cand_idx] = c_vcon
    mandatory[disc_idx] = True

    # Maxima match scaffold_greedy: d_max over finite dilations only, e/v_max
    # over every candidate (disconnected ones contribute 0.0). Reduced here,
    # outside the kernels, so the summation order is fixed.
    finite = c_dil[connected]
    d_max = float(finite.max()) if finite.size else float(eps)
    e_max = float(c_econ.max()) if c_econ.size else float(eps)
    v_max = float(c_vcon.max()) if c_vcon.size else float(eps)

    with parallel_threads(workers):
        score[cand_idx] = _tree_score_kernel(
            c_dil, c_econ, c_vcon, d_max, e_max, v_max,
            float(alpha), float(edge_beta), float(node_beta), float(eps),
        )

    return {
        "dil": dil,
        "length": length,
        "econ_path": econ_path,
        "vcon_path": vcon_path,
        "score": score,
        "mandatory": mandatory,
        "candidate_mask": cand,
        "edge_congestion": econ,
        "node_congestion": vcon,
        "total_stretch": float(dil[conn_idx].sum()),
    }
