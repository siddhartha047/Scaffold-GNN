"""Small helpers for consuming SpectralSparsification artifacts in Benchmark."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def directed_edge_fingerprint(edge_index, num_nodes: int) -> str:
    """Return the exporter fingerprint for the exact PyG edge-column order."""

    import torch

    edges = torch.as_tensor(edge_index).detach().cpu().long().contiguous().numpy()
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {edges.shape}")
    digest = hashlib.sha256()
    digest.update(b"benchmark-directed-edge-order-v1\0")
    digest.update(np.asarray([int(num_nodes)], dtype="<i8").tobytes())
    # Hash by row in bounded chunks.  This is byte-for-byte equivalent to the
    # exporter's C-order hash without materializing another complete edge list.
    chunk_size = 1_048_576
    for row in range(2):
        for first in range(0, edges.shape[1], chunk_size):
            chunk = np.asarray(edges[row, first : first + chunk_size], dtype="<i8")
            digest.update(memoryview(chunk).cast("B"))
    return digest.hexdigest()


def _decode_ascii(value: np.ndarray) -> str:
    return bytes(np.asarray(value, dtype=np.uint8).tolist()).decode("ascii")


def effective_resistance_artifact(
    data_root: str | Path,
    dataset: str,
    *,
    directed: bool = True,
) -> Path:
    """Resolve an ER artifact for any supported Benchmark dataset name."""

    from export_benchmark_graph import DATASET_DIRECTORIES

    key = str(dataset).strip().lower().replace("_", "-")
    try:
        relative = DATASET_DIRECTORIES[key]
    except KeyError as error:
        raise ValueError(f"no effective-resistance artifact for {dataset!r}") from error
    filename = (
        "effective_resistance_directed.npz"
        if directed
        else "effective_resistance.npz"
    )
    return Path(data_root) / relative / "spectral_sparsification" / filename


def load_sparsifier(path: str | Path, *, make_undirected: bool = True):
    """Return `(edge_index, edge_weight)` CPU tensors from a Julia artifact."""

    import torch

    with np.load(Path(path), allow_pickle=False) as values:
        src = np.asarray(values["src"], dtype=np.int64)
        dst = np.asarray(values["dst"], dtype=np.int64)
        weight = np.asarray(values["edge_weight"], dtype=np.float64)
    if make_undirected:
        edge_index = np.vstack((np.r_[src, dst], np.r_[dst, src]))
        edge_weight = np.r_[weight, weight]
    else:
        edge_index = np.vstack((src, dst))
        edge_weight = weight
    return torch.from_numpy(edge_index).long(), torch.from_numpy(edge_weight).float()


def apply_sparsifier(data, path: str | Path, *, clone: bool = True):
    """Replace a PyG Data topology while preserving its features and labels."""

    output = data.clone() if clone else data
    edge_index, edge_weight = load_sparsifier(path)
    output.edge_index = edge_index
    output.edge_weight = edge_weight
    return output


def load_effective_resistance(path: str | Path):
    """Return canonical endpoints, conductance, resistance, and leverage arrays."""

    with np.load(Path(path), allow_pickle=False) as values:
        return {
            key: np.asarray(values[key]).copy()
            for key in ("src", "dst", "conductance", "resistance", "leverage_score")
        }


def load_directed_effective_resistance(
    path: str | Path,
    *,
    edge_index=None,
    num_nodes: int | None = None,
    expected_edges: int | None = None,
):
    """Load ER values aligned exactly with Benchmark ``data.edge_index``.

    Passing both ``edge_index`` and ``num_nodes`` verifies a SHA-256 over the
    exact edge-column order.  This is the safe mode for pipeline use.  The
    legacy ``expected_edges`` argument checks length only and cannot detect a
    reordered topology.
    """

    with np.load(Path(path), allow_pickle=False) as values:
        # Artifacts are stored as float32.  Preserve that dtype so loading
        # hundreds of millions of edge values does not briefly double RAM.
        resistance = np.asarray(values["resistance"], dtype=np.float32).copy()
        artifact_fingerprint = (
            _decode_ascii(values["directed_fingerprint"])
            if "directed_fingerprint" in values.files
            else None
        )
    if (edge_index is None) != (num_nodes is None):
        raise ValueError("edge_index and num_nodes must be provided together")
    if edge_index is not None:
        actual_edges = int(edge_index.shape[1])
        if expected_edges is not None and actual_edges != int(expected_edges):
            raise ValueError(
                f"edge_index has {actual_edges} columns; expected {expected_edges}"
            )
        expected_edges = actual_edges
        if artifact_fingerprint is None:
            raise ValueError("directed ER artifact has no edge-order fingerprint")
        actual_fingerprint = directed_edge_fingerprint(edge_index, int(num_nodes))
        if actual_fingerprint != artifact_fingerprint:
            raise ValueError(
                "directed ER artifact does not match this data.edge_index order; "
                "use the canonical artifact and align_effective_resistance if "
                "the graph was normalized or reordered"
            )
    if expected_edges is not None and len(resistance) != int(expected_edges):
        raise ValueError(
            f"directed ER artifact has {len(resistance)} values; expected {expected_edges}"
        )
    return resistance


def attach_directed_effective_resistance(
    data,
    path: str | Path,
    *,
    attribute: str = "edge_effective_resistance",
    clone: bool = False,
):
    """Attach strictly edge-order-checked ER values to a PyG ``Data`` object."""

    import torch

    output = data.clone() if clone else data
    values = load_directed_effective_resistance(
        path,
        edge_index=output.edge_index,
        num_nodes=int(output.num_nodes),
    )
    setattr(output, attribute, torch.from_numpy(values).float())
    return output


def align_effective_resistance(edge_index, path: str | Path):
    """Return resistance in the original PyG directed-edge order.

    Reciprocal edges receive the same value. Self-loop resistance is zero; an
    edge absent from the cached canonical graph raises `ValueError`.
    """

    import torch

    query = torch.as_tensor(edge_index).detach().cpu().numpy().astype(np.int64, copy=False)
    if query.ndim != 2 or query.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {query.shape}")
    cached = load_effective_resistance(path)
    src = np.asarray(cached["src"], dtype=np.int64)
    dst = np.asarray(cached["dst"], dtype=np.int64)
    stride = int(max(dst.max(initial=0), query.max(initial=0)) + 1)
    cached_key = src * stride + dst

    low = np.minimum(query[0], query[1])
    high = np.maximum(query[0], query[1])
    query_key = low * stride + high
    positions = np.searchsorted(cached_key, query_key)
    loops = low == high
    safe_positions = np.minimum(positions, len(cached_key) - 1)
    missing = (~loops) & (
        (positions == len(cached_key)) | (cached_key[safe_positions] != query_key)
    )
    if missing.any():
        first = int(np.flatnonzero(missing)[0])
        raise ValueError(
            f"edge ({int(query[0, first])}, {int(query[1, first])}) is absent from "
            "the effective-resistance artifact"
        )

    resistance = np.zeros(query.shape[1], dtype=np.float64)
    non_loop = ~loops
    resistance[non_loop] = cached["resistance"][positions[non_loop]]
    return torch.from_numpy(resistance).float()
