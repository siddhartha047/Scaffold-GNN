#!/usr/bin/env python3
"""Export Benchmark's PyG topology to the canonical Julia interchange format."""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_DATA_ROOT = Path("./data")
DEFAULT_SCAFFOLD_ROOT = Path(__file__).resolve().parents[3]

ALL_DATASETS = (
    "cora",
    "citeseer",
    "pubmed",
    "amazon-computer",
    "amazon-photo",
    "coauthor-cs",
    "coauthor-physics",
    "wikics",
    "squirrel",
    "chameleon",
    "roman-empire",
    "amazon-ratings",
    "minesweeper",
    "questions",
    "reddit",
    "ogbn-products",
    "ogbn-arxiv",
    "ogbn-proteins",
    "pokec",
)

DATASET_DIRECTORIES = {
    "cora": Path("Planetoid/Cora"),
    "citeseer": Path("Planetoid/CiteSeer"),
    "pubmed": Path("Planetoid/PubMed"),
    "amazon-computer": Path("Amazon/Computers"),
    "amazon-photo": Path("Amazon/Photo"),
    "coauthor-cs": Path("Coauthor/CS"),
    "coauthor-physics": Path("Coauthor/Physics"),
    "wikics": Path("wikics"),
    "squirrel": Path("geom-gcn/squirrel"),
    "chameleon": Path("geom-gcn/chameleon"),
    "roman-empire": Path("Heterophilous/roman_empire"),
    "amazon-ratings": Path("Heterophilous/amazon_ratings"),
    "minesweeper": Path("Heterophilous/minesweeper"),
    "questions": Path("Heterophilous/questions"),
    "reddit": Path("Reddit"),
    "ogbn-products": Path("ogb/ogbn_products"),
    "ogbn-arxiv": Path("ogb/ogbn_arxiv"),
    "ogbn-proteins": Path("ogb/ogbn_proteins"),
    "pokec": Path("pokec"),
    # KarateClub ships with PyG itself, so this is the only synthetic directory.
    "karate": Path("Karate"),
}


@dataclass(frozen=True)
class ExportResult:
    dataset: str
    graph_path: Path
    artifact_dir: Path
    num_nodes: int
    num_directed_input_edges: int
    num_undirected_edges: int
    fingerprint: str
    export_seconds: float
    cached: bool


def artifact_directory(data_root: Path, dataset: str) -> Path:
    try:
        relative = DATASET_DIRECTORIES[dataset]
    except KeyError as error:
        raise ValueError(f"No artifact-directory mapping for {dataset!r}") from error
    return Path(os.environ.get("SCAFFOLD_SPECTRAL_CACHE", "./results/cache/spectral")) / dataset


def _canonical_undirected_edges(
    edge_index: np.ndarray,
    num_nodes: int,
    edge_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetrize a PyG edge list and return one sorted entry per node pair.

    Reciprocal PyG entries represent one undirected edge, not two parallel
    conductances. Their conductances are averaged. This agrees with taking one
    triangle of a symmetric adjacency matrix, as the original notebook did.
    """

    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {edges.shape}")
    if edge_weight is None:
        weight = np.ones(edges.shape[1], dtype=np.float64)
    else:
        weight = np.asarray(edge_weight, dtype=np.float64).reshape(-1)
        if len(weight) != edges.shape[1]:
            raise ValueError("edge_weight is not aligned with edge_index")

    source, target = edges
    valid = source != target
    source = source[valid]
    target = target[valid]
    weight = weight[valid]
    if source.size == 0:
        raise ValueError("graph contains no non-loop edges")
    if source.min() < 0 or target.min() < 0:
        raise ValueError("edge_index contains negative node ids")
    if source.max() >= num_nodes or target.max() >= num_nodes:
        raise ValueError("edge_index contains a node outside num_nodes")
    if not np.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError("spectral conductances must be finite and strictly positive")

    low = np.minimum(source, target)
    high = np.maximum(source, target)
    key = low * np.int64(num_nodes) + high
    order = np.argsort(key, kind="stable")
    key = key[order]
    low = low[order]
    high = high[order]
    weight = weight[order]

    starts = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(key)])
    undirected_weight = np.add.reduceat(weight, starts) / counts
    return (
        np.ascontiguousarray(low[starts], dtype=np.int64),
        np.ascontiguousarray(high[starts], dtype=np.int64),
        np.ascontiguousarray(undirected_weight, dtype=np.float64),
    )


def _fingerprint(
    num_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"benchmark-spectral-graph-v1\0")
    digest.update(np.asarray([num_nodes], dtype="<i8").tobytes())
    digest.update(np.asarray(src, dtype="<i8").tobytes())
    digest.update(np.asarray(dst, dtype="<i8").tobytes())
    digest.update(np.asarray(weight, dtype="<f8").tobytes())
    return digest.hexdigest()


def _directed_fingerprint(num_nodes: int, edge_index: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"benchmark-directed-edge-order-v1\0")
    digest.update(np.asarray([num_nodes], dtype="<i8").tobytes())
    digest.update(np.asarray(edge_index, dtype="<i8", order="C").tobytes())
    return digest.hexdigest()


def _directed_to_undirected(
    edge_index: np.ndarray,
    num_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
) -> np.ndarray:
    """Map every original directed edge position to a canonical edge position."""

    source, target = np.asarray(edge_index, dtype=np.int64)
    low = np.minimum(source, target)
    high = np.maximum(source, target)
    canonical_key = src * np.int64(num_nodes) + dst
    directed_key = low * np.int64(num_nodes) + high
    positions = np.searchsorted(canonical_key, directed_key)
    loops = source == target
    safe = np.minimum(positions, len(canonical_key) - 1)
    missing = (~loops) & (
        (positions == len(canonical_key)) | (canonical_key[safe] != directed_key)
    )
    if missing.any():
        first = int(np.flatnonzero(missing)[0])
        raise ValueError(
            f"directed edge ({int(source[first])}, {int(target[first])}) "
            "was not found in the canonical graph"
        )
    if len(canonical_key) > np.iinfo(np.int32).max:
        raise ValueError("canonical graph is too large for the Int32 edge mapping")
    mapping = positions.astype(np.int32)
    mapping[loops] = -1
    return mapping


def _bytes(value: str) -> np.ndarray:
    return np.frombuffer(value.encode("utf-8"), dtype=np.uint8).copy()


def _existing_graph_matches(
    path: Path,
    fingerprint: str,
    directed_fingerprint: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as values:
            if "directed_to_undirected" not in values.files:
                return False
            actual = bytes(values["graph_fingerprint"].tolist()).decode("ascii")
            actual_directed = bytes(values["directed_fingerprint"].tolist()).decode("ascii")
        return actual == fingerprint and actual_directed == directed_fingerprint
    except (KeyError, OSError, ValueError, UnicodeDecodeError):
        return False


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export_dataset(
    dataset: str,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    benchmark_root: Path = DEFAULT_SCAFFOLD_ROOT,
    seed: int = 42,
    split_protocol: str = "tunedgnn",
) -> ExportResult:
    started = time.perf_counter()
    resolved_benchmark = benchmark_root.expanduser().resolve()
    if str(resolved_benchmark) not in sys.path:
        sys.path.insert(0, str(resolved_benchmark))
    from scaffold_gnn.data import canonicalize_dataset_name, load_dataset

    canonical = canonicalize_dataset_name(dataset)
    bundle = load_dataset(
        data_root.expanduser().resolve(),
        canonical,
        seed=seed,
        split_protocol=split_protocol,
    )
    data = bundle.data.cpu()
    node_count = int(data.num_nodes)
    edge_index = data.edge_index.detach().numpy()
    raw_weight = getattr(data, "edge_weight", None)
    if raw_weight is not None:
        raw_weight = raw_weight.detach().cpu().numpy()
    src, dst, weight = _canonical_undirected_edges(
        edge_index,
        node_count,
        raw_weight,
    )
    fingerprint = _fingerprint(node_count, src, dst, weight)
    directed_fingerprint = _directed_fingerprint(node_count, edge_index)
    directed_edge_count = int(edge_index.shape[1])
    directed_mapping = _directed_to_undirected(
        edge_index,
        node_count,
        src,
        dst,
    )
    output_dir = artifact_directory(data_root.expanduser().resolve(), canonical)
    graph_path = output_dir / "input_graph.npz"
    cached = _existing_graph_matches(graph_path, fingerprint, directed_fingerprint)
    if not cached:
        _atomic_savez(
            graph_path,
            num_nodes=np.asarray([node_count], dtype=np.int64),
            src=src,
            dst=dst,
            weight=weight,
            graph_fingerprint=_bytes(fingerprint),
            directed_fingerprint=_bytes(directed_fingerprint),
            directed_to_undirected=directed_mapping,
            num_directed_edges=np.asarray([directed_edge_count], dtype=np.int64),
            dataset=_bytes(canonical),
            schema_version=np.asarray([2], dtype=np.int64),
        )
    del bundle, data, edge_index, raw_weight, directed_mapping
    gc.collect()
    return ExportResult(
        dataset=canonical,
        graph_path=graph_path,
        artifact_dir=output_dir,
        num_nodes=node_count,
        num_directed_input_edges=directed_edge_count,
        num_undirected_edges=len(src),
        fingerprint=fingerprint,
        export_seconds=time.perf_counter() - started,
        cached=cached,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=(*ALL_DATASETS, "karate"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_SCAFFOLD_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-protocol", default="tunedgnn")
    args = parser.parse_args()
    result = export_dataset(
        args.dataset,
        data_root=args.data_root,
        benchmark_root=args.benchmark_root,
        seed=args.seed,
        split_protocol=args.split_protocol,
    )
    print(
        f"dataset={result.dataset} nodes={result.num_nodes} "
        f"directed_input_edges={result.num_directed_input_edges} "
        f"undirected_edges={result.num_undirected_edges} cached={result.cached} "
        f"export_seconds={result.export_seconds:.6f} path={result.graph_path}"
    )


if __name__ == "__main__":
    main()
