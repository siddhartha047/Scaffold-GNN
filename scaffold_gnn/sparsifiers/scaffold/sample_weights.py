"""One-time precompute of SCAFFOLD-Sample edge weights, plus artifact I/O.

The artifact is deliberately **ratio-independent**: it stores the aggregated
per-edge weight ``pi``, the tree-locality permutation, and the spanning-forest
edge ids. Inclusion probabilities and their cumulative sum are derived from
``pi`` at sparsifier construction time, which costs a few passes over ``m`` once
per run and lets a single artifact serve every target ratio.

Pipeline (see ``Brainstrom/others/notes/2026-08-17_plan_scaffold_sampling.md`` section 8.3):

1. one deterministic MaxST forest + ``R`` random spanning forests;
2. exact SCAFFOLD scores for every non-tree edge against each forest, via
   :mod:`.tree_score` -- ``O(m log n + n)`` per forest;
3. aggregate to ``pi = clip(freq + lambda * mean_normalised_score, eps, inf)``;
4. sort edges by the DFS preorder of their LCA in the deterministic forest.

Dependency-light on purpose: numpy plus :mod:`.tree_score`, no torch.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import time

import numpy as np

from . import tree_score as ts
from .parallel_utils import auto_worker_count, parallel_threads, split_workers


ARTIFACT_VERSION = 2
DEFAULT_TREE_COUNT = 8
DEFAULT_AGGREGATE_LAMBDA = 1.0
_EPS = 1e-8


# ----------------------------------------------------------------------
# fingerprints and paths
# ----------------------------------------------------------------------
def graph_fingerprint(num_nodes, src, dst):
    """Stable digest of the canonical undirected topology."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.int64(num_nodes).tobytes())
    digest.update(np.int64(src.size).tobytes())
    step = max(1, src.size // 4096)
    digest.update(np.ascontiguousarray(src[::step]).tobytes())
    digest.update(np.ascontiguousarray(dst[::step]).tobytes())
    digest.update(np.int64(int(src.sum()) if src.size else 0).tobytes())
    digest.update(np.int64(int(dst.sum()) if dst.size else 0).tobytes())
    return digest.hexdigest()


def config_tag(tree_count, aggregate_lambda, alpha, edge_beta, node_beta,
               edge_norm_p, node_norm_q, support_weight_method, weight_mode='legacy',
               det_support='maxst'):
    if weight_mode not in ('legacy', 'normalized-mixture'):
        raise ValueError('unknown Sample weight mode')
    payload = json.dumps(
        {
            "v": ARTIFACT_VERSION,
            "R": int(tree_count),
            "lam": float(aggregate_lambda),
            "a": float(alpha),
            "be": float(edge_beta),
            "bn": float(node_beta),
            "p": float(edge_norm_p),
            "q": float(node_norm_q),
            "w": str(support_weight_method),
            **({'weight_components': 1} if weight_mode == 'normalized-mixture' else {}),
            # Only emitted for non-default backbones, so existing cached
            # artifacts keep their hashes instead of all rebuilding.
            **({'det': str(det_support)} if det_support != 'maxst' else {}),
        },
        sort_keys=True,
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def default_artifact_dir(dataset, scratch_root=None):
    root = scratch_root or os.environ.get(
        "SCAFFOLD_SCRATCH_ROOT", "./results/cache"
    )
    return os.path.join(root, "cache", str(dataset).lower(), "scaffold_sample")


def default_artifact_path(dataset, tag, scratch_root=None):
    return os.path.join(default_artifact_dir(dataset, scratch_root), f"weights_{tag}.npz")


def artifact_signature(path):
    """Cheap staleness key, mirroring effective_resistance_artifact_signature."""
    if not path or not os.path.isfile(path):
        return ""
    stat = os.stat(path)
    return f"{stat.st_size}:{stat.st_mtime_ns}"


# ----------------------------------------------------------------------
# inclusion probabilities
# ----------------------------------------------------------------------
def cap_and_renormalize(pi, target):
    """Standard pi-ps: scale ``pi`` so ``sum(p) == target`` with every ``p<=1``.

    Repeatedly caps the entries that would exceed 1 and redistributes the
    remaining budget over the rest, which is the textbook fixed point.
    """
    pi = np.asarray(pi, dtype=np.float64)
    n = int(pi.size)
    target = int(target)
    if n == 0 or target <= 0:
        return np.zeros(n, dtype=np.float64)
    if target >= n:
        return np.ones(n, dtype=np.float64)

    p = np.zeros(n, dtype=np.float64)
    free = np.ones(n, dtype=bool)
    capped = 0
    guard = 0
    while True:
        guard += 1
        remaining = target - capped
        if remaining <= 0 or not free.any():
            break
        free_sum = float(pi[free].sum())
        if not np.isfinite(free_sum) or free_sum <= 0.0:
            p[free] = remaining / float(free.sum())
            break
        scale = remaining / free_sum
        newly = free & (pi * scale >= 1.0)
        if not newly.any() or guard > 64:
            p[free] = np.clip(pi[free] * scale, 0.0, 1.0)
            break
        p[newly] = 1.0
        free &= ~newly
        capped += int(newly.sum())
    return np.clip(p, 0.0, 1.0)


def validate_weight_components(artifact):
    """New opt-in caches retain the two signals without subtractive recovery."""
    m = int(artifact['num_edges'])
    if int(artifact.get('weight_components_version', 0)) != 1:
        raise ValueError('normalized-mixture requires the separate-components artifact; '
                         'run precompute_scaffold_weights.py --weight-mode normalized-mixture')
    for key in ('tree_frequency', 'support_score'):
        value = np.asarray(artifact[key], dtype=np.float64)
        if value.shape != (m,) or not np.isfinite(value).all() or (value < 0).any():
            raise ValueError(f'invalid {key} weight component')


def normalized_mixture_weights(frequency, support, pool_order, mix_alpha):
    """Normalize on the eligible pool AFTER reserving the backbone budget.

    alpha controls support-score mass, not Scaffold's dilation coefficient.
    A zero-mass component falls back to uniform on the eligible pool. A tiny
    relative floor keeps fixed-size sampling feasible even at alpha=0 or 1.
    """
    mix_alpha = float(mix_alpha)
    if not np.isfinite(mix_alpha) or not 0. <= mix_alpha <= 1.:
        raise ValueError('sample_mix_alpha must lie in [0, 1]')
    pool_order = np.asarray(pool_order, dtype=np.int64)
    if not pool_order.size:
        return np.zeros(0, dtype=np.float64)
    def normalize(values):
        values = np.asarray(values, dtype=np.float64)[pool_order]
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('sampling components must be finite and nonnegative')
        total = float(values.sum())
        return values / total if total > 0. else np.full(values.size, 1. / values.size)
    mixture = (1. - mix_alpha) * normalize(frequency) + mix_alpha * normalize(support)
    return np.maximum(mixture, _EPS / pool_order.size)


def with_weight_components(artifact, *, weight=None, alpha=1., edge_beta=1.,
                           node_beta=1., edge_norm_p=2., node_norm_q=2.,
                           workers=None):
    """Upgrade a COPY using the exact cached forests, never replace the source.

    Old caches only store freq + lambda*score. Recompute scores against their
    original trees instead of approximating them by subtracting rounded pi.
    """
    if int(artifact.get('weight_components_version', 0)) == 1:
        validate_weight_components(artifact)
        return dict(artifact)
    started = time.perf_counter()
    n, m, count = (int(artifact[k]) for k in ('num_nodes', 'num_edges', 'tree_count'))
    if count < 1:
        raise ValueError('empty cached forest bank')
    src, dst = artifact['src'], artifact['dst']
    weight = np.ones(m, dtype=np.float64) if weight is None else np.asarray(weight, dtype=np.float64)
    frequency, support = np.zeros(m), np.zeros(m)
    for r in range(count):
        mask = np.zeros(m, dtype=bool)
        mask[forest_edge_ids(artifact, r)] = True
        out = ts.scaffold_tree_scores(n, src, dst, mask, weight, workers=workers,
                                     alpha=alpha, edge_beta=edge_beta, node_beta=node_beta,
                                     edge_norm_p=edge_norm_p, node_norm_q=node_norm_q)
        finite = np.isfinite(out['score'])
        total = float(out['score'][finite].sum())
        if total > 0.:
            support[finite] += out['score'][finite] / total
        frequency += mask
    upgraded = dict(artifact, tree_frequency=frequency / count,
                    support_score=support / count, weight_components_version=np.int64(1),
                    weight_components_build_seconds=np.float64(time.perf_counter() - started))
    validate_weight_components(upgraded)
    return upgraded


# ----------------------------------------------------------------------
# precompute
# ----------------------------------------------------------------------
def build_artifact(
    num_nodes,
    src,
    dst,
    weight=None,
    *,
    tree_count=DEFAULT_TREE_COUNT,
    aggregate_lambda=DEFAULT_AGGREGATE_LAMBDA,
    alpha=1.0,
    edge_beta=1.0,
    node_beta=1.0,
    edge_norm_p=2.0,
    node_norm_q=2.0,
    weighted_paths=False,
    seed=0,
    workers=None,
    verbose=True,
    det_support="maxst",
):
    """Run the full precompute and return the artifact as a dict of arrays.

    ``det_support`` picks the deterministic forest. It is not only the backbone
    unioned into every draw: it also defines the tree-locality ordering ``perm``
    and the support scores, so changing it changes the whole artifact rather
    than just which edges are forced.
    """
    start = time.perf_counter()
    num_nodes = int(num_nodes)
    src = np.ascontiguousarray(src, dtype=np.int64)
    dst = np.ascontiguousarray(dst, dtype=np.int64)
    m = int(src.size)
    if weight is None:
        weight = np.ones(m, dtype=np.float64)
    else:
        weight = np.ascontiguousarray(weight, dtype=np.float64)

    base_components = ts.component_count(num_nodes, src, dst)
    delta_min = (num_nodes - base_components) / max(1, m)

    score_kwargs = dict(
        alpha=alpha,
        edge_beta=edge_beta,
        node_beta=node_beta,
        edge_norm_p=edge_norm_p,
        node_norm_q=node_norm_q,
        weighted_paths=weighted_paths,
    )

    # --- deterministic backbone -------------------------------------------------
    if det_support == "slst":
        # Low-stretch forest. Built through the shared NetworkX implementation
        # and mapped back onto the edge arrays, since the tensor spanning-forest
        # kernels only cover the weighted (min/max) family.
        import networkx as nx

        from scaffold_gnn.sparsifiers.scalable_low_stretch_tree import (
            ScalableLowStretchTreeSparsifier,
        )

        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(zip(src.tolist(), dst.tolist()))
        tree = ScalableLowStretchTreeSparsifier(seed=int(seed)).sparsify(G)
        keep = {frozenset(e) for e in tree.edges()}
        det_mask = np.fromiter(
            (frozenset((int(a), int(b))) in keep for a, b in zip(src, dst)),
            dtype=bool, count=len(src))
    else:
        det_mask = ts.weighted_spanning_forest_mask(
            num_nodes, src, dst, None if np.allclose(weight, weight[:1]) else weight,
            maximum=True,
        )
    det_index = ts.build_tree_index(num_nodes, src[det_mask], dst[det_mask])
    workers = auto_worker_count(0 if workers is None else workers)
    det_out = ts.scaffold_tree_scores(
        num_nodes, src, dst, det_mask, weight, tree_index=det_index,
        workers=workers, **score_kwargs
    )
    if verbose:
        print(
            f"[ScaffoldSample] backbone edges={int(det_mask.sum())} "
            f"components={base_components} delta_min={delta_min:.6f} "
            f"total_stretch={det_out['total_stretch']:.6g}",
            flush=True,
        )

    # --- R random forests -------------------------------------------------------
    acc = np.zeros(m, dtype=np.float64)
    freq = np.zeros(m, dtype=np.float64)
    mandatory = det_out["mandatory"].copy()
    forest_ids = []
    tree_count = max(1, int(tree_count))

    # The R backbones are independent, and this is the widest parallel axis in
    # SCAFFOLD -- it covers the union-find forest construction too, which is
    # serial within any single backbone. Spare workers go to the tree scorer's
    # own candidate-level threads rather than oversubscribing.
    outer, inner = split_workers(tree_count, workers)

    def score_forest(r):
        t0 = time.perf_counter()
        mask = ts.random_spanning_forest_mask(
            num_nodes, src, dst, seed=(int(seed) + 1000 * (r + 1))
        )
        out = ts.scaffold_tree_scores(
            num_nodes, src, dst, mask, weight, workers=inner, **score_kwargs
        )
        return mask, out, time.perf_counter() - t0

    # score_forest passes the inner budget explicitly; each task's scorer
    # sets its own thread-local numba mask.
    if outer > 1:
        with ThreadPoolExecutor(max_workers=outer) as pool:
            results = list(pool.map(score_forest, range(tree_count)))
    else:
        results = [score_forest(r) for r in range(tree_count)]

    # Accumulated in forest order, never completion order: these are float
    # sums, so a scheduler-dependent order would make pi depend on the worker
    # count.
    for r, (mask, out, elapsed) in enumerate(results):
        sc = out["score"]
        finite = np.isfinite(sc)
        total = float(sc[finite].sum())
        if total > 0.0:
            acc[finite] += sc[finite] / total
        freq += mask.astype(np.float64)
        mandatory |= out["mandatory"]
        forest_ids.append(np.flatnonzero(mask).astype(np.int64, copy=False))
        if verbose:
            print(
                f"[ScaffoldSample] forest {r + 1}/{tree_count} "
                f"edges={int(mask.sum())} {elapsed:.2f}s",
                flush=True,
            )

    s = acc / float(tree_count)
    freq /= float(tree_count)
    pi = freq + float(aggregate_lambda) * s
    pi = np.clip(np.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0), _EPS, None)

    # --- tree-locality ordering (DFS preorder of the LCA in the det forest) -----
    det_depth, det_root, det_up, det_parent, det_tin = det_index
    with parallel_threads(workers):
        lca = ts.tree_lca(src, dst, det_depth, det_root, det_up)
    key = np.where(lca >= 0, det_tin[np.maximum(lca, 0)], np.int64(-1))
    perm = np.lexsort((np.arange(m, dtype=np.int64), key)).astype(np.int64, copy=False)

    flat = np.concatenate(forest_ids) if forest_ids else np.zeros(0, dtype=np.int64)
    offsets = np.cumsum([0] + [int(a.size) for a in forest_ids]).astype(np.int64)

    elapsed = time.perf_counter() - start
    artifact = {
        "version": np.int64(ARTIFACT_VERSION),
        "num_nodes": np.int64(num_nodes),
        "num_edges": np.int64(m),
        "src": src,
        "dst": dst,
        "pi": pi,
        "tree_frequency": freq,
        "support_score": s,
        "weight_components_version": np.int64(1),
        "mandatory": mandatory,
        "perm": perm,
        "det_forest_edge_ids": np.flatnonzero(det_mask).astype(np.int64, copy=False),
        "forest_edge_ids_flat": flat,
        "forest_offsets": offsets,
        "tree_count": np.int64(tree_count),
        "base_components": np.int64(base_components),
        "delta_min": np.float64(delta_min),
        "total_stretch": np.float64(det_out["total_stretch"]),
        "build_seconds": np.float64(elapsed),
        "fingerprint": np.frombuffer(
            graph_fingerprint(num_nodes, src, dst).encode("ascii"), dtype=np.uint8
        ),
    }
    if verbose:
        core = int((pi >= pi.max()).sum())
        print(
            f"[ScaffoldSample] precompute done in {elapsed:.2f}s "
            f"edges={m} mandatory={int(mandatory.sum())} "
            f"pi[min={pi.min():.3e} max={pi.max():.3e} top={core}]",
            flush=True,
        )
    return artifact


def save_artifact(path, artifact):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    # np.savez appends '.npz' unless the name already ends with it, so the
    # temp name must too or os.replace looks for the wrong file.
    tmp = f"{path}.tmp{os.getpid()}.npz"
    np.savez(tmp, **artifact)
    os.replace(tmp, path)
    return path


def load_artifact(path):
    with np.load(path, allow_pickle=False) as handle:
        artifact = {key: handle[key] for key in handle.files}
    version = int(artifact.get("version", 0))
    if version != ARTIFACT_VERSION:
        raise ValueError(
            f"scaffold-sample artifact version {version} != expected "
            f"{ARTIFACT_VERSION}: rebuild {path}"
        )
    return artifact


def validate_artifact(artifact, num_nodes, src, dst, path=""):
    """Fail loudly when the artifact does not describe this exact graph."""
    expected = graph_fingerprint(num_nodes, src, dst)
    stored = bytes(np.asarray(artifact["fingerprint"], dtype=np.uint8).tolist()).decode(
        "ascii"
    )
    if stored != expected:
        raise ValueError(
            f"scaffold-sample artifact does not match the input graph ({path}): "
            f"fingerprint {stored} != {expected}. Rebuild with "
            "scripts/precompute_scaffold_weights.py"
        )
    if int(artifact["num_edges"]) != int(np.asarray(src).size):
        raise ValueError(f"scaffold-sample artifact edge-count mismatch ({path})")
    return True


def forest_edge_ids(artifact, index):
    """Edge ids of random forest ``index`` (ragged storage accessor)."""
    offsets = np.asarray(artifact["forest_offsets"], dtype=np.int64)
    count = max(1, int(offsets.size) - 1)
    index = int(index) % count
    lo = int(offsets[index])
    hi = int(offsets[index + 1])
    return np.asarray(artifact["forest_edge_ids_flat"], dtype=np.int64)[lo:hi]


def coverage_report(pi, num_edges, target_edges, epochs=(1, 10, 100, 500)):
    """Section 0.3 diagnostic: deterministic core and union-coverage curve."""
    p = cap_and_renormalize(pi, target_edges)
    always = int((p >= 1.0 - 1e-12).sum())
    with np.errstate(divide="ignore"):
        first = np.where(p > 0, 1.0 / np.maximum(p, 1e-300), np.inf)
    curve = {
        int(e): float(np.mean(1.0 - np.power(1.0 - p, e))) for e in epochs
    }
    return {
        "target_edges": int(target_edges),
        "num_edges": int(num_edges),
        "always_included": always,
        "median_epochs_to_first_inclusion": float(np.median(first[np.isfinite(first)]))
        if np.isfinite(first).any()
        else float("inf"),
        "coverage_curve": curve,
    }
