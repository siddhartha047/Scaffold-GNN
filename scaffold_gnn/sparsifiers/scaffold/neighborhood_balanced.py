"""Label-free, exact-budget compression of a saved Scaffold support union.

Soft node degree targets and first/second neighbor-feature moments replace
hard frequency priority. A forest protects connectivity; each round adds a
matching of lowest-cost edges so marginal scores within that round are exact.
The model, its predictions, and labels are never inputs to construction.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from scipy import sparse

from .consensus import (
    ConsensusGraph, canonical_original_edges, resolve_consensus_budget,
    select_top_k_distinct_supports, support_id_from_original_ids,
)
from .parallel_utils import parallel_threads
from .spanning_tree import build_spanning_forest_mask
from .tree_score import _kernel_parallel, prange


def degree_targets(union_degree, forest_degree, q):
    """Water-fill t=max(forest_degree, c*union_degree), sum(t)=2q.

    These continuous targets need not be jointly realizable by an unweighted
    subgraph; they are soft penalties, never hard endpoint caps.
    """
    upper = np.asarray(union_degree, dtype=np.float64)
    lower = np.asarray(forest_degree, dtype=np.float64)
    if upper.shape != lower.shape or np.any(lower < 0) or np.any(lower > upper):
        raise ValueError('invalid forest/union degree bounds')
    if not lower.sum() <= 2 * q <= upper.sum():
        raise ValueError('q lies outside the forest/union degree bounds')
    if 2 * q == lower.sum():
        return lower.copy()
    if 2 * q == upper.sum():
        return upper.copy()
    lo, hi = 0., 1.
    for _ in range(64):
        scale = (lo + hi) / 2
        if np.maximum(lower, scale * upper).sum() < 2 * q:
            lo = scale
        else:
            hi = scale
    return np.maximum(lower, .5 * (lo + hi) * upper)


def feature_moments(node_features, projection_dim=16, seed=42):
    """Deterministic random projection of row-normalized input features.

    Match neighbor means of z and z^2, which encourages representative
    neighbors and preserves feature spread. Uses no labels or GNN forward.
    Local RNG leaves the training/random state untouched.
    """
    x = np.asarray(torch.as_tensor(node_features).detach().cpu(), dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError('node features must be a finite [nodes, features] matrix')
    if projection_dim < 1:
        raise ValueError('projection_dim must be positive')
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    if x.shape[1] > projection_dim:
        rng = np.random.default_rng(seed)
        projection = rng.choice(np.array([-1., 1.]), size=(x.shape[1], projection_dim))
        x = x @ (projection / np.sqrt(projection_dim))
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return np.ascontiguousarray(np.concatenate((x, x * x), axis=1))


@_kernel_parallel
def addition_costs(src, dst, chosen, degree, target, residual, moments, feature_weight):
    """Exact delta of the quadratic degree + neighbor-moment objective."""
    costs = np.full(len(src), np.inf)
    for i in prange(len(src)):
        if chosen[i]:
            continue
        u, v = src[i], dst[i]
        tu, tv = max(target[u], 1.), max(target[v], 1.)
        cost = (2 * (degree[u] - target[u]) + 1) / (tu * tu)
        cost += (2 * (degree[v] - target[v]) + 1) / (tv * tv)
        for j in range(moments.shape[1]):
            cost += feature_weight * (2 * residual[u, j] * moments[v, j] + moments[v, j] ** 2) / (tu * tu)
            cost += feature_weight * (2 * residual[v, j] * moments[u, j] + moments[u, j] ** 2) / (tv * tv)
        costs[i] = cost
    return costs


def objective_components(degree, target, residual):
    denom = np.maximum(target, 1.)
    active = target > 0
    return dict(
        degree_error=float(np.sum(((degree[active] - target[active]) / denom[active]) ** 2)),
        moment_error=float(np.sum((residual[active] / denom[active, None]) ** 2)),
    )


def build_neighborhood_balanced(original_edge_index, support_records, node_features, *,
                                num_nodes, k=5, delta=None, q=None, workers=1,
                                feature_weight=1., projection_dim=16, seed=42,
                                batch_size=128, progress=None):
    """Compress top-k saved supports to an unweighted q-edge union subset.

    Score every remaining edge after each round. Commit up to batch_size
    vertex-disjoint edges per round (each endpoint updated at most once),
    then update degree and neighbor-moment residuals. Connectivity follows
    from retaining the union's spanning forest throughout.
    """
    started = time.perf_counter()
    if workers < 1 or batch_size < 1 or projection_dim < 1:
        raise ValueError('workers, batch_size and projection_dim must be positive')
    if not np.isfinite(feature_weight) or feature_weight < 0:
        raise ValueError('feature_weight must be finite and nonnegative')
    original = canonical_original_edges(original_edge_index, num_nodes)
    budget = resolve_consensus_budget(original.edge_count, delta=delta, q=q)
    selected = select_top_k_distinct_supports(support_records, original, q=budget, k=k)
    if len(selected) != k:
        raise ValueError(f'requires {k} distinct supports; found {len(selected)}')
    if any(s.source_record.get('edge_weight') is not None for s in selected):
        raise ValueError('neighborhood-balanced pilot requires unweighted supports')
    counts = np.zeros(original.edge_count, dtype=np.int64)
    for support in selected:
        counts[support.original_edge_ids] += 1
    union_ids = np.flatnonzero(counts)
    src, dst = original.pairs[:, union_ids]
    # Use the earlier frequency-first experiment's identical protected forest
    # to isolate the new completion rule. Counts have no further priority.
    order = np.lexsort((union_ids, -counts[union_ids]))
    forest_ordered = build_spanning_forest_mask(num_nodes, src[order], dst[order])
    chosen = np.zeros(len(union_ids), dtype=bool)
    chosen[order[forest_ordered]] = True
    forest_ids = union_ids[chosen].copy()
    if chosen.sum() > budget:
        raise ValueError(f'q={budget} is smaller than the union forest ({chosen.sum()} edges)')
    union_degree = np.bincount(np.concatenate((src, dst)), minlength=num_nodes).astype(float)
    degree = np.bincount(np.concatenate((src[chosen], dst[chosen])), minlength=num_nodes).astype(float)
    target = degree_targets(union_degree, degree, budget)
    moments = feature_moments(node_features, projection_dim, seed)
    if len(moments) != num_nodes:
        raise ValueError('feature node count differs from topology')
    adjacency = sparse.csr_matrix((np.ones(2 * len(src)),
                                   (np.concatenate((src, dst)), np.concatenate((dst, src)))),
                                  shape=(num_nodes, num_nodes))
    target_sum = (adjacency @ moments) * (target / np.maximum(union_degree, 1.))[:, None]
    residual = -target_sum
    np.add.at(residual, src[chosen], moments[dst[chosen]])
    np.add.at(residual, dst[chosen], moments[src[chosen]])
    initial_objective = objective_components(degree, target, residual)
    rounds, retained = 0, int(chosen.sum())
    with parallel_threads(workers):
        while retained < budget:
            costs = addition_costs(src, dst, chosen, degree, target, residual, moments, feature_weight)
            order = np.lexsort((union_ids, costs))
            used = np.zeros(num_nodes, dtype=bool)
            batch = []
            for pos in order:
                if chosen[pos]:
                    continue
                u, v = src[pos], dst[pos]
                if used[u] or used[v]:
                    continue
                batch.append(pos)
                used[u] = used[v] = True
                if len(batch) == min(batch_size, budget - retained):
                    break
            if not batch:
                raise RuntimeError('no candidate available to complete the exact edge budget')
            batch = np.asarray(batch, dtype=np.int64)
            chosen[batch] = True
            np.add.at(degree, src[batch], 1)
            np.add.at(degree, dst[batch], 1)
            np.add.at(residual, src[batch], moments[dst[batch]])
            np.add.at(residual, dst[batch], moments[src[batch]])
            retained += len(batch)
            rounds += 1
            if progress and (rounds == 1 or rounds % 25 == 0 or retained == budget):
                progress(f'round={rounds} edges={retained}/{budget} elapsed={time.perf_counter() - started:.1f}s')
    output_ids = union_ids[chosen]
    if len(output_ids) != budget or not np.isin(forest_ids, output_ids).all():
        raise AssertionError('construction violated budget or protected forest')
    metadata = dict(
        schema_version=1, algorithm='scaffold-neighborhood-balanced-v1',
        m=original.edge_count, q=budget, num_nodes=int(num_nodes),
        union_edges=len(union_ids), forest_edges=len(forest_ids),
        final_canonical_edges=len(output_ids), k_requested=k, k_effective=len(selected),
        target_ratio=budget / original.edge_count if original.edge_count else 0.,
        feature_weight=feature_weight, projection_dim=projection_dim, projection_seed=seed,
        moment_dimension=moments.shape[1], feature_source='row-normalized-input-features',
        degree_target='max(forest_degree, scale*union_degree); sum=2q',
        target_degree_sum=float(target.sum()), degree_penalty_weight=1.,
        selection='recomputed-quadratic-marginal-cost; disjoint-edge-batches',
        frequency_policy='forest-only; no hard frequency preference in completion',
        node_congestion=False, uses_labels=False, uses_model_predictions=False,
        batch_size=batch_size, rounds=rounds, workers=workers,
        initial_objective=initial_objective, final_objective=objective_components(degree, target, residual),
        construction_time_sec=time.perf_counter() - started,
        occurrence_histogram={str(i): int(np.count_nonzero(counts[union_ids] == i)) for i in range(1, k + 1)},
        retained_occurrence_histogram={str(i): int(np.count_nonzero(counts[output_ids] == i)) for i in range(1, k + 1)},
        selected_supports=[dict(epoch=s.epoch, validation_score=s.validation_score, support_id=s.support_id) for s in selected],
        output_support_id=support_id_from_original_ids(output_ids))
    return ConsensusGraph(torch.from_numpy(original.pairs[:, output_ids].copy()).long(),
                          output_ids, forest_ids, metadata)
