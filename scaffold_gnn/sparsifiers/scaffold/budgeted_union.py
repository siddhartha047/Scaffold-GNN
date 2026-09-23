"""Final-only, frequency-first pruning of a support union to exactly q edges.

Unlike Consensus's soft frequency/score mixture, repeated edges take strict
priority over singletons (except for a protected spanning forest). Scores use
the ORIGINAL graph as routing demand, not just the union. This is the one-pass
Scaffold-Fast tree approximation, not adaptive Greedy reverse deletion.
"""

from collections import Counter

import numpy as np
import torch

from .consensus import (
    ConsensusGraph, canonical_original_edges, resolve_consensus_budget,
    select_top_k_distinct_supports, support_id_from_original_ids,
)
from .spanning_tree import build_spanning_forest_mask
from .tree_score import scaffold_tree_scores


def build_budgeted_union(original_edge_index, support_records, *, num_nodes,
                         k=5, delta=None, q=None, alpha=1., edge_beta=1.,
                         node_beta=0., edge_norm_p=2., workers=1,
                         expected_dataset=None, expected_split_fingerprint=None,
                         expected_run=None, expected_target_ratio=None):
    """Return an unweighted q-edge subset of k distinct validation-ranked graphs.

    Votes are per distinct canonical support, not per directed edge or epoch.
    A frequency-prioritized spanning forest protects union connectivity. Keep
    all remaining repeated edges if they fit; otherwise highest frequency
    wins and original-graph support scores break ties. Singletons fill only
    leftover slots, highest support score first. Infeasible forest budgets
    fail rather than silently disconnecting the graph or exceeding q.
    """
    if node_beta != 0:
        raise ValueError('budgeted union disables node congestion: node_beta must be 0')
    if workers < 1 or alpha < 0 or edge_beta < 0 or edge_norm_p < 1:
        raise ValueError('workers must be positive and score parameters nonnegative (p >= 1)')
    if not all(np.isfinite(v) for v in (alpha, edge_beta, edge_norm_p)):
        raise ValueError('score parameters must be finite')
    original = canonical_original_edges(original_edge_index, num_nodes)
    budget = resolve_consensus_budget(original.edge_count, delta=delta, q=q)
    selected = select_top_k_distinct_supports(
        support_records, original, q=budget, k=k,
        expected_dataset=expected_dataset,
        expected_split_fingerprint=expected_split_fingerprint,
        expected_run=expected_run, expected_target_ratio=expected_target_ratio)
    if len(selected) != k:
        raise ValueError(f'budgeted union requires {k} distinct supports; got {len(selected)}')
    counts = np.zeros(original.edge_count, dtype=np.int64)
    for support in selected:
        counts[support.original_edge_ids] += 1
    union = np.flatnonzero(counts)
    # A frequency-maximal forest retains bridges, including unique bridges.
    ordered = union[np.lexsort((union, -counts[union]))]
    pairs = original.pairs[:, ordered]
    forest = np.sort(ordered[build_spanning_forest_mask(num_nodes, pairs[0], pairs[1])])
    if len(forest) > budget:
        raise ValueError(f'q={budget} cannot preserve the union spanning forest ({len(forest)} edges)')
    forest_mask = np.zeros(original.edge_count, dtype=bool)
    forest_mask[forest] = True
    remaining = budget - len(forest)
    candidates = union[~forest_mask[union]]
    repeated = candidates[counts[candidates] > 1]
    unique = candidates[counts[candidates] == 1]
    scores = np.zeros(original.edge_count, dtype=np.float64)
    if len(union) > budget and remaining:
        # All omitted original edges contribute to tree-path congestion.
        scores = np.asarray(scaffold_tree_scores(
            num_nodes, original.pairs[0], original.pairs[1], forest_mask, None,
            alpha=alpha, edge_beta=edge_beta, node_beta=0.,
            edge_norm_p=edge_norm_p, node_norm_q=2., workers=workers)['score'])
        if np.isnan(scores).any():
            raise ValueError('non-finite (NaN) Scaffold support score')
    repeat_order = np.lexsort((repeated, -scores[repeated], -counts[repeated]))
    keep_repeated = repeated[repeat_order[:remaining]]
    slots = remaining - len(keep_repeated)
    unique_order = np.lexsort((unique, -scores[unique]))
    output = np.sort(np.concatenate((forest, keep_repeated, unique[unique_order[:slots]])))
    if len(output) != budget or len(np.unique(output)) != budget:
        raise AssertionError('budgeted union did not produce exactly q canonical edges')
    if not np.isin(forest, output).all() or not np.isin(output, union).all():
        raise AssertionError('budgeted union lost its forest or escaped its source union')
    histogram = Counter(int(c) for c in counts[union])
    all_repeated = union[counts[union] > 1]
    kept_repeated = int(np.count_nonzero(counts[output] > 1))
    metadata = dict(
        schema_version=1, algorithm='scaffold-budgeted-union-frequency-first',
        score_reference='original-full-graph', score_approximation='fixed-forest-scaffold-fast',
        num_nodes=int(num_nodes), m=original.edge_count, q=budget,
        target_ratio=budget / original.edge_count if original.edge_count else 0.,
        k_requested=int(k), k_effective=len(selected), union_edges=len(union),
        repeated_core_edges=len(all_repeated), repeated_core_retained=kept_repeated,
        repeated_core_dropped=len(all_repeated) - kept_repeated,
        repeated_core_and_forest_fit=bool(len(np.union1d(all_repeated, forest)) <= budget),
        singleton_retained=int(np.count_nonzero(counts[output] == 1)),
        singleton_forest_edges=int(np.count_nonzero(counts[forest] == 1)),
        forest_edges=len(forest), node_congestion=False,
        occurrence_histogram={str(c): histogram[c] for c in sorted(histogram)},
        selected_supports=[dict(support_id=s.support_id, epoch=s.epoch,
                                validation_score=s.validation_score) for s in selected],
        final_canonical_edges=len(output), output_support_id=support_id_from_original_ids(output))
    return ConsensusGraph(torch.from_numpy(original.pairs[:, output].copy()).long(),
                          output, forest, metadata)
