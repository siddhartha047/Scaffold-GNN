"""Exact-budget Scaffold-Consensus graph construction.

The union of validation-ranked training supports is a *construction* graph
only.  This module always returns one canonical, undirected, non-self-loop
support with exactly ``q`` edges.  GNN evaluation and TunedGNN self-loop
handling remain the responsibility of :mod:`main`.

Scaffold's existing tree scorer and spanning-forest implementation are reused
directly; occurrence counts influence edge selection but never become message
passing weights.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .spanning_tree import build_spanning_forest_mask, resolve_edge_budget
from .tree_score import scaffold_tree_scores


@dataclass(frozen=True)
class CanonicalOriginalEdges:
    """Original undirected edge index with stable first-occurrence IDs."""

    num_nodes: int
    pairs: np.ndarray
    keys: np.ndarray
    sorted_keys: np.ndarray
    sorted_original_ids: np.ndarray

    @property
    def edge_count(self) -> int:
        return int(self.pairs.shape[1])


@dataclass(frozen=True)
class SelectedSupport:
    support_id: str
    epoch: int
    validation_score: float
    original_edge_ids: np.ndarray
    source_record: Mapping[str, Any]


@dataclass(frozen=True)
class ConsensusGraph:
    """One exact-budget graph and auditable construction metadata."""

    undirected_pairs: torch.Tensor
    original_edge_ids: np.ndarray
    forest_original_edge_ids: np.ndarray
    metadata: Mapping[str, Any]


def _as_numpy_edge_index(edge_index: Any) -> np.ndarray:
    edges = torch.as_tensor(edge_index, dtype=torch.long).detach().cpu()
    if edges.ndim != 2 or edges.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    return edges.numpy().astype(np.int64, copy=False)


def canonical_original_edges(
    edge_index: Any,
    num_nodes: int,
) -> CanonicalOriginalEdges:
    """Canonicalize edges while preserving stable original edge IDs.

    Reverse directions count as one edge, self-loops are excluded, and the ID
    of an undirected edge is its first occurrence in ``edge_index``.  This is
    deterministic for both the repository's undirected-unique and symmetric
    storage conventions.
    """

    num_nodes = int(num_nodes)
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    edges = _as_numpy_edge_index(edge_index)
    if edges.size:
        if int(edges.min()) < 0 or int(edges.max()) >= num_nodes:
            raise ValueError(f"node ID lies outside [0, {num_nodes - 1}]")
    keep = edges[0] != edges[1]
    lo = np.minimum(edges[0, keep], edges[1, keep])
    hi = np.maximum(edges[0, keep], edges[1, keep])
    keys = lo * np.int64(max(1, num_nodes)) + hi
    if keys.size:
        _, first = np.unique(keys, return_index=True)
        first = np.sort(first.astype(np.int64, copy=False))
        pairs = np.ascontiguousarray(np.stack((lo[first], hi[first]), axis=0))
        stable_keys = np.ascontiguousarray(keys[first], dtype=np.int64)
    else:
        pairs = np.empty((2, 0), dtype=np.int64)
        stable_keys = np.empty(0, dtype=np.int64)
    key_order = np.argsort(stable_keys, kind="stable")
    return CanonicalOriginalEdges(
        num_nodes=num_nodes,
        pairs=pairs,
        keys=stable_keys,
        sorted_keys=np.ascontiguousarray(stable_keys[key_order]),
        sorted_original_ids=np.ascontiguousarray(key_order, dtype=np.int64),
    )


def support_original_edge_ids(
    edge_index: Any,
    original: CanonicalOriginalEdges,
) -> np.ndarray:
    """Map one support to sorted, unique original undirected edge IDs."""

    support = canonical_original_edges(edge_index, original.num_nodes)
    if support.edge_count == 0:
        return np.empty(0, dtype=np.int64)
    if original.sorted_keys.size == 0:
        raise ValueError("non-empty support supplied for an empty original graph")
    positions = np.searchsorted(original.sorted_keys, support.keys)
    valid = positions < original.sorted_keys.size
    valid &= original.sorted_keys[
        np.minimum(positions, original.sorted_keys.size - 1)
    ] == support.keys
    if not bool(valid.all()):
        missing = support.pairs[:, ~valid]
        preview = missing[:, :5].T.tolist()
        raise ValueError(f"support contains edges outside the original graph: {preview}")
    ids = original.sorted_original_ids[positions]
    return np.sort(np.unique(ids).astype(np.int64, copy=False))


def support_id_from_original_ids(original_edge_ids: Any) -> str:
    ids = np.ascontiguousarray(original_edge_ids, dtype=np.int64)
    return hashlib.sha256(ids.tobytes()).hexdigest()


def resolve_consensus_budget(
    original_edge_count: int,
    *,
    delta: float | None,
    q: int | None,
) -> int:
    """Resolve ``q`` using the repository's ceil-based budget convention."""

    m = int(original_edge_count)
    if m < 0:
        raise ValueError("original_edge_count must be non-negative")
    ratio_budget = None
    if delta is not None:
        delta = float(delta)
        if not 0.0 <= delta <= 1.0:
            raise ValueError("delta must lie in [0, 1]")
        ratio_budget = int(resolve_edge_budget(m, target_ratio=delta))
    if q is None:
        if ratio_budget is None:
            raise ValueError("one of delta or q must be supplied")
        return ratio_budget
    q = int(q)
    if not 0 <= q <= m:
        raise ValueError(f"q must lie in [0, {m}], got {q}")
    if ratio_budget is not None and q != ratio_budget:
        raise ValueError(
            f"inconsistent q/delta: q={q}, but delta={delta:g} gives {ratio_budget}"
        )
    return q


def _record_value(record: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key in record:
        return record[key]
    return default


def select_top_k_distinct_supports(
    records: Iterable[Mapping[str, Any]],
    original: CanonicalOriginalEdges,
    *,
    q: int,
    k: int,
    expected_dataset: str | None = None,
    expected_split_fingerprint: str | None = None,
    expected_run: int | None = None,
    expected_target_ratio: float | None = None,
) -> list[SelectedSupport]:
    """Select validation-ranked distinct supports after strict provenance checks."""

    k = int(k)
    if k < 1:
        raise ValueError("K must be at least 1")
    prepared: list[SelectedSupport] = []
    seen: set[str] = set()
    ordered = sorted(
        list(records),
        key=lambda item: (
            float(_record_value(item, "validation_score", float("-inf"))),
            int(_record_value(item, "epoch", -1)),
        ),
        reverse=True,
    )
    for record in ordered:
        for key, expected in (
            ("dataset", expected_dataset),
            ("split_fingerprint", expected_split_fingerprint),
            ("run", expected_run),
        ):
            actual = _record_value(record, key)
            if expected is not None and actual is not None and str(actual) != str(expected):
                raise ValueError(
                    f"support {key} mismatch: expected {expected!r}, got {actual!r}"
                )
        actual_ratio = _record_value(record, "target_ratio")
        if (
            expected_target_ratio is not None
            and actual_ratio is not None
            and not math.isclose(
                float(actual_ratio), float(expected_target_ratio), rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise ValueError(
                "support target ratio mismatch: "
                f"expected {expected_target_ratio}, got {actual_ratio}"
            )
        edge_index = _record_value(record, "edge_index")
        if edge_index is None:
            view = _record_value(record, "view")
            if view is None:
                raise ValueError("support record requires edge_index or view")
            edge_index = view[0]
        ids = support_original_edge_ids(edge_index, original)
        if int(ids.size) != int(q):
            raise ValueError(
                f"support epoch={_record_value(record, 'epoch', -1)} has "
                f"{ids.size} canonical edges, expected exactly q={q}"
            )
        identity = support_id_from_original_ids(ids)
        if identity in seen:
            continue
        seen.add(identity)
        prepared.append(
            SelectedSupport(
                support_id=str(_record_value(record, "support_id", identity)),
                epoch=int(_record_value(record, "epoch", -1)),
                validation_score=float(_record_value(record, "validation_score")),
                original_edge_ids=ids,
                source_record=record,
            )
        )
        if len(prepared) >= k:
            break
    if not prepared:
        raise ValueError("no valid distinct support records were supplied")
    return prepared


def average_percentile_ranks(scores: Any) -> np.ndarray:
    """Ascending average ranks normalized to [0, 1], including tied scores."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if bool(np.isnan(values).any()):
        raise ValueError("Fast scores must not contain NaN")
    size = int(values.size)
    if size == 0:
        return np.empty(0, dtype=np.float64)
    if size == 1:
        return np.asarray([0.5], dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(size, dtype=np.float64)
    start = 0
    while start < size:
        stop = start + 1
        while stop < size and values[order[stop]] == values[order[start]]:
            stop += 1
        # Average of one-based ranks [start + 1, ..., stop].
        average_one_based = 0.5 * ((start + 1) + stop)
        ranks[order[start:stop]] = (average_one_based - 1.0) / (size - 1.0)
        start = stop
    return ranks


def _fast_scores_for_union(
    original: CanonicalOriginalEdges,
    union_ids: np.ndarray,
    forest_union_mask: np.ndarray,
    *,
    edge_weights: np.ndarray | None,
    alpha: float,
    edge_beta: float,
    node_beta: float,
    edge_norm_p: float,
    node_norm_q: float,
    workers: int,
) -> np.ndarray:
    pairs = original.pairs[:, union_ids]
    weights = None if edge_weights is None else np.asarray(edge_weights, dtype=np.float64)[union_ids]
    return scaffold_tree_scores(
        original.num_nodes,
        pairs[0],
        pairs[1],
        forest_union_mask,
        weights,
        alpha=float(alpha),
        edge_beta=float(edge_beta),
        node_beta=float(node_beta),
        edge_norm_p=float(edge_norm_p),
        node_norm_q=float(node_norm_q),
        workers=int(workers),
    )["score"]


def build_scaffold_consensus(
    original_edge_index: Any,
    support_records: Sequence[Mapping[str, Any]],
    *,
    num_nodes: int,
    k: int = 5,
    delta: float | None = None,
    q: int | None = None,
    mixing_lambda: float = 0.5,
    edge_weights: Any | None = None,
    alpha: float = 1.0,
    edge_beta: float = 1.0,
    node_beta: float = 0.0,
    edge_norm_p: float = 2.0,
    node_norm_q: float = 2.0,
    workers: int = 1,
    expected_dataset: str | None = None,
    expected_split_fingerprint: str | None = None,
    expected_run: int | None = None,
    expected_target_ratio: float | None = None,
) -> ConsensusGraph:
    """Construct one validation-ranked, exact-budget consensus support."""

    mixing_lambda = float(mixing_lambda)
    if not 0.0 <= mixing_lambda <= 1.0:
        raise ValueError("consensus lambda must lie in [0, 1]")
    original = canonical_original_edges(original_edge_index, num_nodes)
    budget = resolve_consensus_budget(original.edge_count, delta=delta, q=q)
    selected = select_top_k_distinct_supports(
        support_records,
        original,
        q=budget,
        k=k,
        expected_dataset=expected_dataset,
        expected_split_fingerprint=expected_split_fingerprint,
        expected_run=expected_run,
        expected_target_ratio=expected_target_ratio,
    )
    k_effective = len(selected)

    counts = np.zeros(original.edge_count, dtype=np.int64)
    for support in selected:
        counts[support.original_edge_ids] += 1
    union_ids = np.flatnonzero(counts).astype(np.int64, copy=False)
    forest_priority = np.lexsort((union_ids, -counts[union_ids]))
    ordered_union_ids = union_ids[forest_priority]
    ordered_pairs = original.pairs[:, ordered_union_ids]
    ordered_forest_mask = build_spanning_forest_mask(
        original.num_nodes,
        ordered_pairs[0],
        ordered_pairs[1],
    )
    forest_ids = np.sort(ordered_union_ids[ordered_forest_mask])
    if int(forest_ids.size) > budget:
        raise ValueError(
            "q is too small for the consensus spanning forest: "
            f"q={budget}, forest_edges={forest_ids.size}"
        )

    forest_union_mask = np.isin(union_ids, forest_ids, assume_unique=True)
    candidate_positions = np.flatnonzero(~forest_union_mask).astype(np.int64, copy=False)
    candidate_ids = union_ids[candidate_positions]
    remaining = int(budget - forest_ids.size)
    if remaining > int(candidate_ids.size):
        raise ValueError(
            "union does not contain enough edges to meet q: "
            f"remaining={remaining}, candidates={candidate_ids.size}"
        )

    all_fast_scores = _fast_scores_for_union(
        original,
        union_ids,
        forest_union_mask,
        edge_weights=(
            None
            if edge_weights is None
            else np.asarray(torch.as_tensor(edge_weights).detach().cpu(), dtype=np.float64)
        ),
        alpha=alpha,
        edge_beta=edge_beta,
        node_beta=node_beta,
        edge_norm_p=edge_norm_p,
        node_norm_q=node_norm_q,
        workers=workers,
    )
    raw_scores = all_fast_scores[candidate_positions]
    percentile = average_percentile_ranks(raw_scores)
    frequency = counts[candidate_ids].astype(np.float64) / float(k_effective)
    combined = (1.0 - mixing_lambda) * percentile + mixing_lambda * frequency
    completion_order = np.lexsort((candidate_ids, -raw_scores, -combined))
    completion_ids = candidate_ids[completion_order[:remaining]]
    output_ids = np.sort(np.concatenate((forest_ids, completion_ids))).astype(
        np.int64, copy=False
    )

    if int(output_ids.size) != budget:
        raise AssertionError(f"consensus produced {output_ids.size} edges, expected {budget}")
    if not set(forest_ids.tolist()).issubset(output_ids.tolist()):
        raise AssertionError("consensus output does not contain its spanning forest")
    if not set(output_ids.tolist()).issubset(union_ids.tolist()):
        raise AssertionError("consensus output contains an edge outside the support union")

    histogram = Counter(int(value) for value in counts[union_ids].tolist())
    overlap = {
        support.support_id: int(
            np.intersect1d(output_ids, support.original_edge_ids, assume_unique=True).size
        )
        for support in selected
    }
    pairs = torch.from_numpy(original.pairs[:, output_ids].copy()).long().contiguous()
    metadata = {
        "schema_version": 1,
        "algorithm": "scaffold-consensus",
        "num_nodes": int(num_nodes),
        "m": original.edge_count,
        "q": budget,
        "target_ratio": (float(budget) / original.edge_count if original.edge_count else 0.0),
        "lambda": mixing_lambda,
        "k_requested": int(k),
        "k_effective": k_effective,
        "union_edges": int(union_ids.size),
        "union_ratio": (float(union_ids.size) / original.edge_count if original.edge_count else 0.0),
        "forest_edges": int(forest_ids.size),
        "occurrence_histogram": {str(key): histogram[key] for key in sorted(histogram)},
        "selected_supports": [
            {
                "support_id": support.support_id,
                "epoch": support.epoch,
                "validation_score": support.validation_score,
                "overlap_edges": overlap[support.support_id],
            }
            for support in selected
        ],
        "final_canonical_edges": int(output_ids.size),
        "output_support_id": support_id_from_original_ids(output_ids),
    }
    return ConsensusGraph(
        undirected_pairs=pairs,
        original_edge_ids=output_ids,
        forest_original_edge_ids=forest_ids,
        metadata=metadata,
    )


def select_validation_candidate(
    validation_metrics: Mapping[str, float],
    *,
    fallback_name: str = "scaffold-1",
    consensus_order: Sequence[str] = (
        "consensus-lambda-0.5",
        "consensus-lambda-0",
        "consensus-lambda-1",
    ),
) -> str:
    """Select by validation metric only; exact ties prefer Scaffold-1."""

    if not validation_metrics:
        raise ValueError("at least one validation candidate is required")
    values = {str(name): float(value) for name, value in validation_metrics.items()}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("validation metrics must be finite")
    best = max(values.values())
    tied = {name for name, value in values.items() if value == best}
    if fallback_name in tied:
        return fallback_name
    for name in consensus_order:
        if name in tied:
            return name
    return sorted(tied)[0]


def save_consensus_artifact(
    graph: ConsensusGraph,
    destination: str | os.PathLike[str],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save a directly reloadable fixed-support artifact."""

    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = dict(graph.metadata)
    if metadata:
        merged.update(dict(metadata))
    payload = {
        "undirected_pairs": graph.undirected_pairs.detach().cpu().long().contiguous(),
        "num_nodes": int(graph.metadata.get("num_nodes", 0)),
        "metadata": merged,
    }
    if payload["num_nodes"] <= 0 and graph.undirected_pairs.numel():
        payload["num_nodes"] = int(graph.undirected_pairs.max()) + 1
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_consensus_artifact(
    source: str | os.PathLike[str],
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    path = Path(source).expanduser().resolve()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "undirected_pairs" not in payload:
        raise ValueError(f"{path}: invalid Scaffold-Consensus artifact")
    pairs = torch.as_tensor(payload["undirected_pairs"], dtype=torch.long).contiguous()
    if pairs.ndim != 2 or pairs.size(0) != 2:
        raise ValueError(f"{path}: undirected_pairs must have shape [2, q]")
    if pairs.numel() and bool((pairs[0] >= pairs[1]).any()):
        raise ValueError(f"{path}: undirected_pairs are not canonical")
    if int(torch.unique(pairs, dim=1).size(1)) != int(pairs.size(1)):
        raise ValueError(f"{path}: duplicate undirected edges")
    return pairs, dict(payload.get("metadata", {}))
