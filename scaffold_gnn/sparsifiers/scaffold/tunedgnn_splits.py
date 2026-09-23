"""Fixed-split resolution matching Benchmark/tunedGNN for Scaffold/full runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .tunedgnn_presets import canonical_dataset


PLANETOID_DATASETS = {"cora", "citeseer", "pubmed"}
NPZ_SPLIT_DATASETS = {
    "amazon-computer",
    "amazon-photo",
    "coauthor-cs",
    "coauthor-physics",
}
HETEROPHILOUS_DATASETS = {
    "roman-empire",
    "amazon-ratings",
    "minesweeper",
    "questions",
}
GEOM_GCN_DATASETS = {"chameleon", "squirrel"}
OGB_DATASETS = {"ogbn-arxiv", "ogbn-products", "ogbn-proteins"}
TUNEDGNN_SPLIT_DATASETS = (
    PLANETOID_DATASETS
    | NPZ_SPLIT_DATASETS
    | HETEROPHILOUS_DATASETS
    | GEOM_GCN_DATASETS
    | OGB_DATASETS
    | {"wikics", "pokec", "reddit"}
)


def _indices(mask):
    return torch.where(torch.as_tensor(mask))[0].long()


def _npz_split(path):
    values = np.load(path)
    return [{key: torch.from_numpy(values[key]).long() for key in ("train", "valid", "test")}]


def tunedgnn_fixed_splits(data_dir, dataset, dataset_name):
    """Return the same split list loaded by tunedGNN's medium/large mains.

    Planetoid datasets return ``None`` because their class-balanced split is
    created by the main parser path.  Reddit uses its native PyG masks because
    tunedGNN-org publishes no Reddit entrypoint.
    """

    name = canonical_dataset(dataset_name)
    root = Path(data_dir)

    if name in PLANETOID_DATASETS:
        return None
    if name in NPZ_SPLIT_DATASETS:
        return _npz_split(root / f"{name}_split.npz")
    if name in HETEROPHILOUS_DATASETS:
        from torch_geometric.datasets import HeterophilousGraphDataset

        pyg_dataset = HeterophilousGraphDataset(
            # PyG accepts the tunedGNN spelling produced by ``capitalize``
            # (for example Roman-empire), not CamelCase RomanEmpire.
            name=name.capitalize(),
            root=str(root),
        )
        data = pyg_dataset[0]
        return [
            {
                "train": _indices(data.train_mask[:, column]),
                "valid": _indices(data.val_mask[:, column]),
                "test": _indices(data.test_mask[:, column]),
            }
            for column in range(data.train_mask.shape[1])
        ]
    if name == "wikics":
        from torch_geometric.datasets import WikiCS

        data = WikiCS(root=str(root / "wikics"))[0]
        valid_mask = torch.logical_or(data.val_mask, data.stopping_mask)
        return [
            {
                "train": _indices(data.train_mask[:, column]),
                "valid": _indices(valid_mask[:, column]),
                "test": _indices(data.test_mask),
            }
            for column in range(data.train_mask.shape[1])
        ]
    if name in GEOM_GCN_DATASETS:
        values = np.load(root / "geom-gcn" / name / f"{name}_filtered.npz")
        nodes = np.arange(values["train_masks"].shape[1])
        return [
            {
                "train": torch.as_tensor(nodes[values["train_masks"][column]]).long(),
                "valid": torch.as_tensor(nodes[values["val_masks"][column]]).long(),
                "test": torch.as_tensor(nodes[values["test_masks"][column]]).long(),
            }
            for column in range(values["train_masks"].shape[0])
        ]
    if name == "pokec":
        values = np.load(root / "pokec" / "pokec-splits.npy", allow_pickle=True)
        splits = []
        for raw in values:
            item = raw.item() if hasattr(raw, "item") else raw
            splits.append(
                {
                    key: torch.from_numpy(np.asarray(item[key])).long()
                    for key in ("train", "valid", "test")
                }
            )
        return splits
    if name in OGB_DATASETS and hasattr(dataset, "load_fixed_splits"):
        value = dataset.load_fixed_splits()
        return value if isinstance(value, list) else [value]
    if name in OGB_DATASETS | {"reddit"} and all(
        hasattr(dataset, key) for key in ("train_idx", "valid_idx", "test_idx")
    ):
        if dataset.train_idx is not None:
            return [{
                "train": torch.as_tensor(dataset.train_idx).long(),
                "valid": torch.as_tensor(dataset.valid_idx).long(),
                "test": torch.as_tensor(dataset.test_idx).long(),
            }]
    if name in TUNEDGNN_SPLIT_DATASETS:
        raise RuntimeError(
            f"Dataset '{name}' did not expose its required tunedGNN split; "
            "refusing a random split fallback for a Scaffold comparison run."
        )
    return None
