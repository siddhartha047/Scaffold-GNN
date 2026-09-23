"""Adapters that expose Benchmark datasets to external baseline projects."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from ..config.paths import DATA_ROOT, SCRATCH_ROOT
from .datasets import (
    DatasetBundle,
    canonicalize_dataset_name,
    load_dataset,
    split_fingerprint,
)


DEFAULT_SPLIT_PROTOCOL = "tunedgnn"


def _resolve_data_root(data_root: str | Path | None) -> Path:
    return Path(
        data_root
        or os.environ.get("SCAFFOLD_DATA_ROOT")
        or os.environ.get("SUPPORT_GRAPH_DATA_DIR")
        or DATA_ROOT
    ).expanduser().resolve()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else int(default)


def _indices_to_mask(indices: torch.Tensor, num_nodes: int) -> torch.Tensor:
    mask = torch.zeros(int(num_nodes), dtype=torch.bool)
    mask[torch.as_tensor(indices).long().cpu()] = True
    return mask


def method_scratch_dir(dataset_name: str, method_name: str | None = None) -> Path:
    """Return the method-specific scratch directory for a dataset."""

    configured = os.environ.get("SCAFFOLD_METHOD_SCRATCH")
    if configured:
        destination = Path(configured).expanduser().resolve()
    else:
        method = method_name or os.environ.get("BASELINE_METHOD_NAME", "shared")
        destination = (
            SCRATCH_ROOT
            / "BatchExperiments"
            / canonicalize_dataset_name(dataset_name)
            / str(method).strip().lower().replace(" ", "_")
        )
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _write_dataset_manifest(
    destination: Path,
    *,
    bundle: DatasetBundle,
    split: dict[str, torch.Tensor],
    data_root: Path,
    run_index: int,
) -> None:
    payload: dict[str, Any] = {
        "dataset": bundle.name,
        "data_root": str(data_root),
        "split_protocol": bundle.split_protocol,
        "split_index": int(run_index) % len(bundle.splits),
        "split_fingerprint": split_fingerprint(split),
        "num_nodes": int(bundle.data.num_nodes),
        "num_directed_edges": int(bundle.data.edge_index.size(1)),
        "num_features": int(bundle.data.x.size(1)),
        "num_classes_or_tasks": int(bundle.num_classes),
        "is_multilabel": bool(bundle.is_multilabel),
    }
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".dataset_manifest.",
        suffix=".json",
        dir=destination,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, destination / "dataset_manifest.json")


def load_benchmark_bundle(
    data_root: str | Path | None,
    dataset_name: str,
    *,
    seed: int | None = None,
    split_protocol: str | None = None,
) -> DatasetBundle:
    """Load the exact dataset/split implementation used by Benchmark."""

    resolved_root = _resolve_data_root(data_root)
    resolved_seed = (
        int(seed)
        if seed is not None
        else _env_int("SCAFFOLD_DATASET_SEED", _env_int("BASELINE_SEED", 42))
    )
    protocol = (
        split_protocol
        or os.environ.get("SCAFFOLD_SPLIT_PROTOCOL")
        or DEFAULT_SPLIT_PROTOCOL
    )
    return load_dataset(
        resolved_root,
        dataset_name,
        seed=resolved_seed,
        split_protocol=protocol,
    )


def load_pyg_data(
    data_root: str | Path | None,
    dataset_name: str,
    *,
    seed: int | None = None,
    run_index: int | None = None,
    split_protocol: str | None = None,
) -> tuple[Data, DatasetBundle]:
    """Return a PyG graph carrying Benchmark's train/valid/test masks."""

    resolved_root = _resolve_data_root(data_root)
    bundle = load_benchmark_bundle(
        resolved_root,
        dataset_name,
        seed=seed,
        split_protocol=split_protocol,
    )
    resolved_run = (
        int(run_index)
        if run_index is not None
        else _env_int("SCAFFOLD_SPLIT_RUN", 0)
    )
    split = bundle.split_for_run(resolved_run)
    data = bundle.data.clone().cpu()
    data.benchmark_splits = [
        {
            key: torch.as_tensor(value).long().cpu()
            for key, value in candidate.items()
        }
        for candidate in bundle.splits
    ]
    select_pyg_split(data, resolved_run, announce=False)
    fingerprint = split_fingerprint(split)
    data.benchmark_dataset_name = bundle.name
    data.benchmark_split_protocol = bundle.split_protocol
    data.benchmark_split_fingerprint = fingerprint
    scratch = method_scratch_dir(bundle.name)
    data.benchmark_method_scratch = str(scratch)
    _write_dataset_manifest(
        scratch,
        bundle=bundle,
        split=split,
        data_root=resolved_root,
        run_index=resolved_run,
    )
    print(
        "[BenchmarkDataset] "
        f"dataset={bundle.name} split_protocol={bundle.split_protocol} "
        f"split_index={resolved_run % len(bundle.splits)} "
        f"split_fingerprint={fingerprint} "
        f"train={int(data.train_mask.sum())} "
        f"valid={int(data.val_mask.sum())} "
        f"test={int(data.test_mask.sum())} "
        f"data_root={resolved_root} "
        f"method_scratch={scratch}",
        flush=True,
    )
    return data, bundle


def select_pyg_split(
    data: Data,
    run_index: int,
    *,
    announce: bool = True,
) -> Data:
    """Select the same fixed split tunedGNN uses for one internal run."""

    splits = getattr(data, "benchmark_splits", None)
    if not splits:
        return data
    resolved_index = int(run_index) % len(splits)
    split = splits[resolved_index]
    device = data.x.device
    num_nodes = int(data.num_nodes)
    data.train_mask = _indices_to_mask(
        torch.as_tensor(split["train"]).cpu(), num_nodes
    ).to(device)
    data.val_mask = _indices_to_mask(
        torch.as_tensor(split["valid"]).cpu(), num_nodes
    ).to(device)
    data.test_mask = _indices_to_mask(
        torch.as_tensor(split["test"]).cpu(), num_nodes
    ).to(device)
    fingerprint = split_fingerprint(
        {
            key: torch.as_tensor(value).cpu()
            for key, value in split.items()
        }
    )
    data.benchmark_split_fingerprint = fingerprint
    data.benchmark_split_index = resolved_index
    if announce:
        print(
            "[BenchmarkSplit] "
            f"run={int(run_index)} split_index={resolved_index} "
            f"split_fingerprint={fingerprint}",
            flush=True,
        )
    return data


def load_dense_tensors(
    data_root: str | Path | None,
    dataset_name: str,
):
    """Convert an Benchmark graph for legacy dense full-graph baselines."""

    data, _bundle = load_pyg_data(data_root, dataset_name)
    dense_limit = _env_int("SCAFFOLD_DENSE_MAX_NODES", 30_000)
    if int(data.num_nodes) > dense_limit:
        raise RuntimeError(
            f"{dataset_name} has {data.num_nodes} nodes, but this baseline requires "
            f"a dense adjacency and SCAFFOLD_DENSE_MAX_NODES={dense_limit}. "
            "Use a sparse/minibatch implementation for this method/dataset."
        )
    adjacency = to_dense_adj(
        data.edge_index,
        max_num_nodes=int(data.num_nodes),
    )[0]
    labels = data.y
    if labels.ndim > 1 and labels.size(-1) > 1:
        raise RuntimeError(
            f"{dataset_name} is multi-label, but this legacy dense baseline uses "
            "single-label cross-entropy."
        )
    return (
        adjacency,
        data.x.float(),
        labels.reshape(-1).long(),
        torch.where(data.train_mask)[0],
        torch.where(data.val_mask)[0],
        torch.where(data.test_mask)[0],
    )


def infer_embedding_dim(
    data_root: str | Path | None,
    dataset_name: str,
    hidden_dim: int = 512,
) -> list[int]:
    """Infer input/output dimensions without changing the Benchmark split."""

    bundle = load_benchmark_bundle(data_root, dataset_name)
    return [
        int(bundle.data.x.size(1)),
        int(hidden_dim),
        int(bundle.num_classes),
    ]
