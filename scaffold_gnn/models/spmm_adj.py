"""Route PyG convolutions through fused SpMM instead of COO message passing.

A COO forward materializes one ``[num_edges, hidden]`` message tensor. Under
k-view the evaluation graph is the union of K supports, which on the densest
datasets approaches the whole graph: that single allocation reached 80.8 GiB on
reddit and 72.0 GiB on ogbn-products, past an 80 GB card. Handing the conv a
``SparseTensor`` routes it through ``message_and_aggregate`` (fused SpMM),
which never builds that tensor.

This is a cost change, not a results change: verified identical to the
``edge_index`` path to 1.8e-7 through ``ProductsGNN``.

**Coalescing matters.** Duplicate ``(row, col)`` pairs are summed by SpMM but
normalized differently by ``gcn_norm`` on an edge list, and the two then
disagree by ~0.17. The loader coalesces every graph it produces, so model
forwards pass ``coalesce_first=False``; the k-view union, which overlaps by
construction, passes ``True``.

``ProteinGNN`` is excluded: it is DGL, not PyG, and builds its own ``dgl.graph``
from the edge list.

Set ``SCAFFOLD_DISABLE_SPMM=1`` to fall back to COO everywhere.
"""

from __future__ import annotations

import os
import weakref

import torch

__all__ = ["to_spmm_adj", "spmm_enabled", "clear_spmm_cache"]

# Built adjacencies, keyed by id(edge_index). Rebuilding costs a sort -- 0.43 s
# for 8M directed edges, so ~5 s per forward on the ogbn-products k-view union
# -- and the graph is unchanged across the forwards of one epoch. Entries are
# evicted by a weakref.finalize on the edge_index itself, so a cached adjacency
# can never outlive its tensor and an id can never be reused while live.
#
# NOTE: SparseTensor(row=..., col=...) is COO-backed at construction and only
# builds CSR (rowptr) on first use; that conversion is cheap (3 ms) because the
# constructor has already sorted. is_sorted=True is NOT a valid shortcut here:
# after PyG's coalesce the destination array is not non-decreasing, and passing
# it silently yields a different, wrong rowptr.
_ADJ_CACHE: dict = {}
_ANNOUNCED = False


def clear_spmm_cache():
    _ADJ_CACHE.clear()


def spmm_enabled() -> bool:
    return os.environ.get("SCAFFOLD_DISABLE_SPMM", "") not in ("1", "true", "True")


def to_spmm_adj(edge_index, edge_weight=None, num_nodes=None, *,
                coalesce_first=False, drop_values=False):
    """Return (adj, None) for the fused path, or (edge_index, edge_weight).

    Returning the pair lets a caller splice the result straight into a conv
    call without branching on which path it got.
    """

    if not spmm_enabled():
        return edge_index, edge_weight
    if edge_index is None or not isinstance(edge_index, torch.Tensor):
        return edge_index, edge_weight          # already a SparseTensor, or absent
    if edge_index.dim() != 2 or edge_index.size(0) != 2 or edge_index.numel() == 0:
        return edge_index, edge_weight
    try:
        from torch_sparse import SparseTensor
    except ImportError:
        return edge_index, edge_weight

    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1
    # GAT, GIN and SAGE take no edge_weight, so the COO path silently ignores
    # it. SpMM would instead fold it into the aggregation, which would make the
    # two paths disagree. Drop the values so both ignore it alike.
    if drop_values:
        edge_weight = None
    if coalesce_first:
        from torch_geometric.utils import coalesce as _coalesce
        if edge_weight is None:
            edge_index = _coalesce(edge_index, num_nodes=num_nodes)
        else:
            edge_index, edge_weight = _coalesce(
                edge_index, edge_weight, num_nodes=num_nodes, reduce="max")

    key = (id(edge_index), num_nodes, bool(drop_values),
           None if edge_weight is None else id(edge_weight))
    hit = _ADJ_CACHE.get(key)
    if hit is not None:
        return hit, None

    # PyG's adj_t convention is transposed: row is the destination.
    adj = SparseTensor(
        row=edge_index[1], col=edge_index[0], value=edge_weight,
        sparse_sizes=(num_nodes, num_nodes),
    )
    adj.storage.rowptr()          # materialize CSR once, not per conv layer
    global _ANNOUNCED
    if not _ANNOUNCED:
        _ANNOUNCED = True
        print(
            "[SpMM] fused CSR aggregation active: "
            f"nodes={num_nodes} directed_edges={edge_index.size(1)} "
            f"values={'dropped' if drop_values else ('none' if edge_weight is None else 'kept')}"
            " (COO message tensor never materialized)",
            flush=True,
        )
    _ADJ_CACHE[key] = adj
    weakref.finalize(edge_index, _ADJ_CACHE.pop, key, None)
    return adj, None
