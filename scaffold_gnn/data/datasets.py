"""Standalone node-classification dataset loaders.

The supported names mirror ``SupportGraphJuly23/utils/dataset.py`` while
returning a compact, uniform PyG ``Data`` + split contract.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures

from ..config.tunedgnn_presets import TUNEDGNN_REVISION
from .split_utils import class_rand_splits

CANONICAL_DATASETS = (
    "amazon-computer",
    "amazon-photo",
    "amazon-ratings",
    "chameleon",
    "citeseer",
    "coauthor-cs",
    "coauthor-physics",
    "cora",
    "cora-full",
    "cornell",
    "karate",
    "karate-balance",
    "minesweeper",
    "ogbn-arxiv",
    "ogbn-products",
    "ogbn-proteins",
    "pokec",
    "pubmed",
    "reddit",
    "reddit2",
    "roman-empire",
    "squirrel",
    "texas",
    "tolokers",
    "questions",
    "wikics",
    "wisconsin",
    "reed98",
    "amherst41",
    "penn94",
    "johnshopkins55",
    "cornell5",
)


ALIASES = {
    "amazon-computers": "amazon-computer",
    "arxiv": "ogbn-arxiv",
    "corafull": "cora-full",
    "karate-balanced": "karate-balance",
    "karate_balance": "karate-balance",
    "karateclub": "karate",
    "ogb-arxiv": "ogbn-arxiv",
    "ogb-product": "ogbn-products",
    "ogb-products": "ogbn-products",
    "ogb-protein": "ogbn-proteins",
    "ogb-proteins": "ogbn-proteins",
    "product": "ogbn-products",
    "products": "ogbn-products",
    "protein": "ogbn-proteins",
    "proteins": "ogbn-proteins",
    "pokec-regions": "pokec",
    "graphland-pokec": "pokec",
    "amherest41": "amherst41",
    "johnhopkings55": "johnshopkins55",
}

SPLIT_PROTOCOLS = ("tunedgnn", "native", "random")


@dataclass(frozen=True)
class VerifiedAsset:
    relative_path: str
    url: str
    sha256: str


_TUNEDGNN_RAW = (
    "https://raw.githubusercontent.com/LUOyk1999/tunedGNN/"
    f"{TUNEDGNN_REVISION}"
)
TUNEDGNN_ASSETS = {
    "amazon-computer": VerifiedAsset(
        "amazon-computer_split.npz",
        f"{_TUNEDGNN_RAW}/medium_graph/data/amazon-computer_split.npz",
        "dff698aba1dd30b8bd7d1dca65c60b1f8d9027d950e07a23e86dcbf16c539f24",
    ),
    "amazon-photo": VerifiedAsset(
        "amazon-photo_split.npz",
        f"{_TUNEDGNN_RAW}/medium_graph/data/amazon-photo_split.npz",
        "0609f6ab15ddabfe33a34f10d960d09499e634472f72104abb1f3be858127432",
    ),
    "coauthor-cs": VerifiedAsset(
        "coauthor-cs_split.npz",
        f"{_TUNEDGNN_RAW}/medium_graph/data/coauthor-cs_split.npz",
        "b8c46a38d93b1f9d6b8b8ab944d870e8d44ccdde0639188681e0d38212480ec5",
    ),
    "coauthor-physics": VerifiedAsset(
        "coauthor-physics_split.npz",
        f"{_TUNEDGNN_RAW}/medium_graph/data/coauthor-physics_split.npz",
        "2034eb90f5ff80eee0e371cc05b20690a16786fb085924b9166c83361fbbcb39",
    ),
    "pokec": VerifiedAsset(
        "pokec/pokec-splits.npy",
        f"{_TUNEDGNN_RAW}/large_graph/data/pokec/pokec-splits.npy",
        "2a6ceb92307ee911daa3428dc2491d83b5f7deefd7faa2dfe502b8f7c2947d1b",
    ),
}

_HETERO_REVISION = "a431395582e929d88271309716bea4fe24ce6318"
_HETERO_RAW = (
    "https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/"
    f"{_HETERO_REVISION}/data"
)
TUNEDGNN_ASSETS.update(
    {
        "chameleon": VerifiedAsset(
            "geom-gcn/chameleon/chameleon_filtered.npz",
            f"{_HETERO_RAW}/chameleon_filtered.npz",
            "bf46f07e1fb5249280447e5fe3100f3e82fc4b93ad1e13ffcfdec924b6ac0bb5",
        ),
        "squirrel": VerifiedAsset(
            "geom-gcn/squirrel/squirrel_filtered.npz",
            f"{_HETERO_RAW}/squirrel_filtered.npz",
            "f06370ffb3116c5de8b7a600005400b2293c025e0ce2f6916d8201a825505ac7",
        ),
    }
)


def canonicalize_dataset_name(name: str) -> str:
    normalized = "-".join(str(name).strip().lower().replace("_", "-").split())
    return ALIASES.get(normalized, normalized)


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _ensure_verified_asset(root: Path, asset: VerifiedAsset) -> Path:
    destination = root / asset.relative_path
    if destination.exists():
        actual = _file_sha256(destination)
        if actual != asset.sha256:
            raise ValueError(
                f"Existing tunedGNN asset failed SHA-256 verification: {destination} "
                f"({actual} != {asset.sha256})"
            )
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".download",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        urllib.request.urlretrieve(asset.url, temporary)
        actual = _file_sha256(temporary)
        if actual != asset.sha256:
            raise ValueError(
                f"Downloaded tunedGNN asset failed SHA-256 verification: "
                f"{actual} != {asset.sha256}"
            )
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def _tunedgnn_class_split(labels: torch.Tensor, seed: int) -> dict[str, torch.Tensor]:
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        return class_rand_splits(
            labels,
            label_num_per_class=20,
            valid_num=500,
            test_num=1000,
        )
    finally:
        torch.random.set_rng_state(rng_state)


def _split_from_npz(path: Path) -> list[dict[str, torch.Tensor]]:
    values = np.load(path)
    return [
        {
            "train": torch.from_numpy(values["train"]),
            "valid": torch.from_numpy(values["valid"]),
            "test": torch.from_numpy(values["test"]),
        }
    ]


def _split_from_object_array(path: Path) -> list[dict[str, torch.Tensor]]:
    values = np.load(path, allow_pickle=True)
    splits: list[dict[str, torch.Tensor]] = []
    for raw in values:
        item = raw.item() if hasattr(raw, "item") else raw
        splits.append(
            {
                "train": torch.from_numpy(np.asarray(item["train"])),
                "valid": torch.from_numpy(np.asarray(item["valid"])),
                "test": torch.from_numpy(np.asarray(item["test"])),
            }
        )
    return splits


def _wikics_tunedgnn_splits(data: Data) -> list[dict[str, torch.Tensor]]:
    train_mask = torch.as_tensor(data.train_mask)
    valid_mask = torch.logical_or(
        torch.as_tensor(data.val_mask),
        torch.as_tensor(data.stopping_mask),
    )
    test_mask = torch.as_tensor(data.test_mask)
    return [
        {
            "train": torch.where(train_mask[:, index])[0],
            "valid": torch.where(valid_mask[:, index])[0],
            "test": torch.where(test_mask)[0],
        }
        for index in range(train_mask.shape[1])
    ]


@dataclass
class DatasetBundle:
    name: str
    data: Data
    splits: list[dict[str, torch.Tensor]]
    num_classes: int
    is_multilabel: bool
    split_protocol: str = "tunedgnn"

    def split_for_run(self, run: int = 0) -> dict[str, torch.Tensor]:
        if not self.splits:
            raise RuntimeError(f"dataset '{self.name}' has no train/valid/test split")
        split = self.splits[int(run) % len(self.splits)]
        return {key: value.detach().cpu().long() for key, value in split.items()}


def _indices(mask_or_indices: torch.Tensor, column: int | None = None) -> torch.Tensor:
    value = torch.as_tensor(mask_or_indices).detach().cpu()
    if value.ndim == 2:
        value = value[:, 0 if column is None else int(column)]
    if value.dtype == torch.bool:
        return torch.where(value)[0].long()
    return value.reshape(-1).long()


def _splits_from_data(data: Data) -> list[dict[str, torch.Tensor]]:
    if not all(hasattr(data, key) for key in ("train_mask", "val_mask", "test_mask")):
        return []
    train_mask = torch.as_tensor(data.train_mask)
    valid_mask = torch.as_tensor(data.val_mask)
    test_mask = torch.as_tensor(data.test_mask)
    width = max(
        value.size(1) if value.ndim == 2 else 1 for value in (train_mask, valid_mask, test_mask)
    )
    return [
        {
            "train": _indices(train_mask, index),
            "valid": _indices(valid_mask, index),
            "test": _indices(test_mask, index),
        }
        for index in range(width)
    ]


def split_fingerprint(split: dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for key in ("train", "valid", "test"):
        values = torch.as_tensor(split[key]).long().sort().values
        hasher.update(key.encode("utf-8"))
        hasher.update(values.numpy().tobytes())
    return hasher.hexdigest()


def _random_split(
    labels: torch.Tensor,
    *,
    seed: int,
    train_ratio: float,
    valid_ratio: float,
) -> dict[str, torch.Tensor]:
    labels = torch.as_tensor(labels).detach().cpu()
    if labels.ndim > 1 and labels.size(1) > 1:
        labeled = torch.where(torch.isfinite(labels).any(dim=1))[0]
    else:
        flat = labels.reshape(-1)
        labeled = torch.where(torch.isfinite(flat.float()) & (flat >= 0))[0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = labeled[torch.randperm(labeled.numel(), generator=generator)]
    train_count = int(order.numel() * float(train_ratio))
    valid_count = int(order.numel() * float(valid_ratio))
    return {
        "train": order[:train_count],
        "valid": order[train_count : train_count + valid_count],
        "test": order[train_count + valid_count :],
    }


def _balanced_karate_split(labels: torch.Tensor, seed: int) -> dict[str, torch.Tensor]:
    labels = labels.reshape(-1).long()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    train_parts: list[torch.Tensor] = []
    remaining_by_class: list[torch.Tensor] = []
    for class_id in torch.unique(labels, sorted=True):
        nodes = torch.where(labels == class_id)[0]
        nodes = nodes[torch.randperm(nodes.numel(), generator=generator)]
        train_parts.append(nodes[:2])
        remaining_by_class.append(nodes[2:])

    valid_total = 8
    remaining_counts = torch.tensor(
        [nodes.numel() for nodes in remaining_by_class],
        dtype=torch.long,
    )
    ideal = remaining_counts.double() * (valid_total / int(remaining_counts.sum()))
    allocation = torch.floor(ideal).long()
    capacity = torch.clamp(remaining_counts - 1, min=0)
    allocation = torch.minimum(allocation, capacity)
    while int(allocation.sum()) < valid_total:
        candidates = [
            (float(ideal[index] - allocation[index]), -index, index)
            for index in range(len(remaining_by_class))
            if int(allocation[index]) < int(capacity[index])
        ]
        if not candidates:
            raise RuntimeError("unable to construct the balanced Karate split")
        allocation[max(candidates)[-1]] += 1

    valid_parts: list[torch.Tensor] = []
    test_parts: list[torch.Tensor] = []
    for index, nodes in enumerate(remaining_by_class):
        count = int(allocation[index])
        valid_parts.append(nodes[:count])
        test_parts.append(nodes[count:])

    def shuffled(parts: list[torch.Tensor]) -> torch.Tensor:
        values = torch.cat(parts)
        return values[torch.randperm(values.numel(), generator=generator)]

    return {
        "train": shuffled(train_parts),
        "valid": shuffled(valid_parts),
        "test": shuffled(test_parts),
    }


def _ensure_features(data: Data) -> None:
    if getattr(data, "x", None) is not None:
        data.x = torch.as_tensor(data.x).float()
        if data.x.ndim == 1:
            data.x = data.x.unsqueeze(1)
        return
    degree = torch.bincount(data.edge_index.reshape(-1), minlength=int(data.num_nodes)).float()
    data.x = torch.log1p(degree).unsqueeze(1)


def _finalize(
    name: str,
    data: Data,
    *,
    root: Path,
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    split_protocol: str,
    explicit_splits: list[dict[str, torch.Tensor]] | None = None,
) -> DatasetBundle:
    data = data.cpu()
    data.edge_index = torch.as_tensor(data.edge_index).long().cpu()
    data.num_nodes = int(data.num_nodes)
    _ensure_features(data)
    data.y = torch.as_tensor(data.y).cpu()
    is_multilabel = data.y.ndim == 2 and data.y.size(1) > 1
    if not is_multilabel:
        data.y = data.y.reshape(-1).long()
        valid_labels = data.y[data.y >= 0]
        num_classes = int(valid_labels.max()) + 1 if valid_labels.numel() else 0
    else:
        data.y = data.y.float()
        num_classes = int(data.y.size(1))

    splits = explicit_splits if explicit_splits is not None else _splits_from_data(data)
    if split_protocol == "random":
        splits = []
    if not splits:
        split_dir = root / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        ratio_tag = f"{train_ratio:.6f}_{valid_ratio:.6f}".replace(".", "p")
        cache = split_dir / f"{name}_{ratio_tag}_seed{int(seed)}.pt"
        if cache.exists():
            split = torch.load(cache, map_location="cpu", weights_only=True)
        else:
            split = _random_split(
                data.y,
                seed=seed,
                train_ratio=train_ratio,
                valid_ratio=valid_ratio,
            )
            torch.save(split, cache)
        splits = [split]
    for split in splits:
        if set(split) != {"train", "valid", "test"}:
            raise ValueError(f"invalid split contract for dataset '{name}'")
        split["train"] = _indices(split["train"])
        split["valid"] = _indices(split["valid"])
        split["test"] = _indices(split["test"])
    return DatasetBundle(
        name=name,
        data=data,
        splits=splits,
        num_classes=num_classes,
        is_multilabel=is_multilabel,
        split_protocol=split_protocol,
    )


def _load_planetoid(root: Path, name: str, *, normalize_features: bool) -> Data:
    from torch_geometric.datasets import Planetoid

    pyg_name = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}[name]
    return Planetoid(
        root=str(root / "Planetoid"),
        name=pyg_name,
        transform=NormalizeFeatures() if normalize_features else None,
    )[0]


def _load_heterophilous(root: Path, name: str) -> Data:
    from torch_geometric.datasets import HeterophilousGraphDataset

    return HeterophilousGraphDataset(root=str(root / "Heterophilous"), name=name)[0]


def _load_wikipedia(root: Path, name: str) -> Data:
    local_npz = root / "geom-gcn" / name / f"{name}_filtered.npz"
    if local_npz.exists():
        values = np.load(local_npz)
        data = Data(
            x=torch.as_tensor(values["node_features"]).float(),
            y=torch.as_tensor(values["node_labels"]).long(),
            edge_index=torch.as_tensor(values["edges"]).long().t().contiguous(),
        )
        if "train_masks" in values:
            data.train_mask = torch.as_tensor(values["train_masks"]).bool().t()
            data.val_mask = torch.as_tensor(values["val_masks"]).bool().t()
            data.test_mask = torch.as_tensor(values["test_masks"]).bool().t()
        return data
    from torch_geometric.datasets import WikipediaNetwork

    return WikipediaNetwork(
        root=str(root / "WikipediaNetwork"),
        name=name,
        geom_gcn_preprocess=True,
        transform=NormalizeFeatures(),
    )[0]


def _load_facebook100(root: Path, name: str) -> Data:
    directory = root / "facebook100"
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / f"{name}.mat"
    if not source.exists():
        url = f"https://github.com/sisaman/Facebook100/raw/master/data/{name}.mat"
        urllib.request.urlretrieve(url, source)
    values = scipy.io.loadmat(source)
    adjacency = values["A"].tocoo()
    info = np.asarray(values["local_info"])
    edge_index = torch.from_numpy(np.vstack((adjacency.row, adjacency.col))).long()
    feature_parts: list[torch.Tensor] = []
    for column in (0, 1, 2, 3, 4, 6):
        raw = torch.as_tensor(info[:, column]).long()
        _, encoded = torch.unique(raw, sorted=True, return_inverse=True)
        feature_parts.append(F.one_hot(encoded).float())
    raw_labels = torch.as_tensor(info[:, 5]).long()
    _, labels = torch.unique(raw_labels, sorted=True, return_inverse=True)
    return Data(
        x=torch.cat(feature_parts, dim=1),
        y=labels.long(),
        edge_index=edge_index,
        num_nodes=info.shape[0],
    )


def _load_pokec(root: Path) -> tuple[Data, list[dict[str, torch.Tensor]] | None]:
    directory = root / "pokec"
    mat_path = directory / "pokec.mat"
    if mat_path.exists():
        values = scipy.io.loadmat(mat_path)
        data = Data(
            x=torch.as_tensor(values["node_feat"]).float(),
            y=torch.as_tensor(values["label"]).reshape(-1).long(),
            edge_index=torch.as_tensor(values["edge_index"]).long(),
            num_nodes=int(np.asarray(values["num_nodes"]).reshape(-1)[0]),
        )
        split_path = directory / "pokec-splits.npy"
        if split_path.exists():
            raw_splits = np.load(split_path, allow_pickle=True)
            splits = []
            for raw in raw_splits:
                item = raw.item() if hasattr(raw, "item") else raw
                splits.append(
                    {
                        "train": torch.as_tensor(item["train"]).long(),
                        "valid": torch.as_tensor(item["valid"]).long(),
                        "test": torch.as_tensor(item["test"]).long(),
                    }
                )
            return data, splits
        return data, None

    try:
        from torch_geometric.datasets import LINKXDataset

        data = LINKXDataset(root=str(root / "LINKX"), name="pokec")[0]
        return data, _splits_from_data(data)
    except Exception as exc:
        raise FileNotFoundError(
            "Pokec was not available through PyG. Place the LINKX pokec.mat and "
            "optional pokec-splits.npy under <data-root>/pokec/."
        ) from exc


def _load_trusted_ogb_cache(path: Path) -> Any:
    """Load an OGB-owned processed cache under the user-selected data root.

    OGB 1.3.6 does not pass ``weights_only=False`` when reading its processed
    graph dictionaries. PyTorch 2.6 changed that omitted argument's default to
    ``True``, which rejects OGB's protocol-4 NumPy payloads. This opt-out is
    intentionally limited to files created and addressed by OGB itself.
    """

    return torch.load(path, map_location="cpu", weights_only=False)


def _load_ogb(root: Path, name: str) -> tuple[Data, list[dict[str, torch.Tensor]]]:
    try:
        from ogb.nodeproppred import NodePropPredDataset
    except ImportError as exc:
        raise ImportError("OGB datasets require the 'ogb' package") from exc

    class _CompatibleNodePropPredDataset(NodePropPredDataset):
        """Scope PyTorch 2.6 compatibility to OGB's own cache reads."""

        def pre_process(self) -> None:
            processed = Path(self.root) / "processed" / "data_processed"
            # OGB 1.3.6 assumes this directory was created while extracting a
            # fresh download.  A valid raw-only cache (for example, one copied
            # into scratch) otherwise fails when torch.save opens this path.
            processed.parent.mkdir(parents=True, exist_ok=True)
            if processed.exists():
                loaded = _load_trusted_ogb_cache(processed)
                self.graph = loaded["graph"]
                self.labels = loaded["labels"]
                return
            super().pre_process()

        def get_idx_split(self, split_type=None):
            resolved_type = split_type or self.meta_info["split"]
            cached_split = (
                Path(self.root) / "split" / str(resolved_type) / "split_dict.pt"
            )
            if cached_split.is_file():
                return _load_trusted_ogb_cache(cached_split)
            return super().get_idx_split(split_type)

    dataset = _CompatibleNodePropPredDataset(name=name, root=str(root / "ogb"))
    graph, labels = dataset[0]
    x = graph.get("node_feat")
    if x is None:
        edge_features = torch.as_tensor(graph["edge_feat"]).float()
        destinations = torch.as_tensor(graph["edge_index"])[1].long()
        x = torch.zeros((int(graph["num_nodes"]), edge_features.size(1)), dtype=torch.float)
        x.index_add_(0, destinations, edge_features)
    data = Data(
        x=torch.as_tensor(x).float(),
        y=torch.as_tensor(labels),
        edge_index=torch.as_tensor(graph["edge_index"]).long(),
        num_nodes=int(graph["num_nodes"]),
    )
    # The native ogbn-proteins MoG learner scores edges from their eight input
    # attributes. Preserve them in the shared PyG object instead of retaining
    # only the node features aggregated from those attributes.
    if graph.get("edge_feat") is not None:
        data.edge_attr = torch.as_tensor(graph["edge_feat"]).float()
    raw = dataset.get_idx_split()
    split = {
        "train": torch.as_tensor(raw["train"]).long(),
        "valid": torch.as_tensor(raw["valid"]).long(),
        "test": torch.as_tensor(raw["test"]).long(),
    }
    return data, [split]


def load_dataset(
    data_root: str | Path,
    name: str,
    *,
    seed: int = 42,
    train_ratio: float = 0.5,
    valid_ratio: float = 0.25,
    split_protocol: str = "tunedgnn",
) -> DatasetBundle:
    """Load a dataset with tunedGNN splits by default.

    Datasets absent from tunedGNN retain their native PyG/OGB mask when one is
    available and otherwise use Benchmark's deterministic random fallback.
    """

    root = Path(data_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    name = canonicalize_dataset_name(name)
    split_protocol = str(split_protocol).lower()
    if split_protocol not in SPLIT_PROTOCOLS:
        raise ValueError(f"split_protocol must be one of: {', '.join(SPLIT_PROTOCOLS)}")
    if name not in CANONICAL_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {', '.join(CANONICAL_DATASETS)}")
    explicit_splits: list[dict[str, torch.Tensor]] | None = None

    if name in {"cora", "citeseer", "pubmed"}:
        data = _load_planetoid(
            root,
            name,
            normalize_features=split_protocol != "tunedgnn",
        )
        if split_protocol == "tunedgnn":
            explicit_splits = [_tunedgnn_class_split(data.y, seed)]
    elif name == "karate" or name == "karate-balance":
        from torch_geometric.datasets import KarateClub

        data = KarateClub()[0]
        if name == "karate-balance":
            explicit_splits = [_balanced_karate_split(data.y, seed)]
    elif name in {"roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"}:
        data = _load_heterophilous(root, name)
    elif name in {"chameleon", "squirrel"}:
        if split_protocol == "tunedgnn":
            _ensure_verified_asset(root, TUNEDGNN_ASSETS[name])
        data = _load_wikipedia(root, name)
    elif name in {"cornell", "texas", "wisconsin"}:
        from torch_geometric.datasets import WebKB

        data = WebKB(root=str(root / "WebKB"), name=name.capitalize())[0]
    elif name in {"amazon-photo", "amazon-computer"}:
        from torch_geometric.datasets import Amazon

        pyg_name = "Photo" if name == "amazon-photo" else "Computers"
        data = Amazon(
            root=str(root / "Amazon"),
            name=pyg_name,
            transform=NormalizeFeatures(),
        )[0]
        if split_protocol == "tunedgnn":
            split_path = _ensure_verified_asset(root, TUNEDGNN_ASSETS[name])
            explicit_splits = _split_from_npz(split_path)
    elif name in {"coauthor-cs", "coauthor-physics"}:
        from torch_geometric.datasets import Coauthor

        pyg_name = "CS" if name == "coauthor-cs" else "Physics"
        data = Coauthor(
            root=str(root / "Coauthor"),
            name=pyg_name,
            transform=NormalizeFeatures(),
        )[0]
        if split_protocol == "tunedgnn":
            split_path = _ensure_verified_asset(root, TUNEDGNN_ASSETS[name])
            explicit_splits = _split_from_npz(split_path)
    elif name == "wikics":
        from torch_geometric.datasets import WikiCS

        data = WikiCS(
            root=str(root / "wikics"),
            transform=None if split_protocol == "tunedgnn" else NormalizeFeatures(),
        )[0]
        if split_protocol == "tunedgnn":
            explicit_splits = _wikics_tunedgnn_splits(data)
    elif name in {"ogbn-arxiv", "ogbn-products", "ogbn-proteins"}:
        data, explicit_splits = _load_ogb(root, name)
    elif name == "pokec":
        data, explicit_splits = _load_pokec(root)
        if split_protocol == "tunedgnn":
            split_path = _ensure_verified_asset(root, TUNEDGNN_ASSETS[name])
            explicit_splits = _split_from_object_array(split_path)
    elif name in {"reed98", "amherst41", "penn94", "johnshopkins55", "cornell5"}:
        data = _load_facebook100(root, name)
    elif name == "reddit":
        from torch_geometric.datasets import Reddit

        data = Reddit(root=str(root / "Reddit"))[0]
    elif name == "reddit2":
        from torch_geometric.datasets import Reddit2

        data = Reddit2(root=str(root / "Reddit2"))[0]
    elif name == "cora-full":
        from torch_geometric.datasets import CoraFull

        data = CoraFull(root=str(root / "CoraFull"), transform=NormalizeFeatures())[0]
    else:  # pragma: no cover - guarded by CANONICAL_DATASETS.
        raise AssertionError(name)

    bundle = _finalize(
        name,
        data,
        root=root,
        seed=seed,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        split_protocol=split_protocol,
        explicit_splits=explicit_splits,
    )
    return bundle


def dataset_summary(bundle: DatasetBundle) -> dict[str, Any]:
    split = bundle.split_for_run(0)
    return {
        "name": bundle.name,
        "split_protocol": bundle.split_protocol,
        "num_nodes": int(bundle.data.num_nodes),
        "num_directed_edges": int(bundle.data.edge_index.size(1)),
        "num_features": int(bundle.data.x.size(1)),
        "num_classes_or_tasks": int(bundle.num_classes),
        "is_multilabel": bool(bundle.is_multilabel),
        "num_splits": len(bundle.splits),
        "split_fingerprint": split_fingerprint(split),
        "train_nodes": int(split["train"].numel()),
        "valid_nodes": int(split["valid"].numel()),
        "test_nodes": int(split["test"].numel()),
    }
