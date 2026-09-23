"""Download/load the canonical datasets and audit graph/split identity.

This is the headless companion to ``dataset_audit.ipynb``. Every load goes
through the same ``utils.dataset`` path used by ``main.py`` and the baseline
bridge, so standard PyG/OGB processed files are persisted below one data root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Iterable

_SUPPORT_GRAPH_ROOT = Path(__file__).resolve().parents[2]
if str(_SUPPORT_GRAPH_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUPPORT_GRAPH_ROOT))

import torch  # noqa: E402

from scripts.common.baseline_dataset_bridge import load_pyg_data  # noqa: E402
from scaffold_gnn.utils.dataset import canonicalize_dataset_name  # noqa: E402
from scaffold_gnn.utils.defaults import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR  # noqa: E402


DATASETS = [
    "karate",
    "cora",
    "citeseer",
    "pubmed",
    "Reddit",
    "ogbn-arxiv",
    "ogbn-products",
    "ogbn-proteins",
    "pokec",
]


def _hash_tensor_sample(hasher, tensor: torch.Tensor | None, max_items: int = 8192) -> None:
    if tensor is None:
        hasher.update(b"none")
        return
    tensor = torch.as_tensor(tensor).detach().cpu().contiguous()
    hasher.update(str(tuple(tensor.shape)).encode("ascii"))
    hasher.update(str(tensor.dtype).encode("ascii"))
    flat = tensor.reshape(-1)
    if flat.numel() > max_items:
        # Integer arithmetic avoids float32 rounding to ``flat.numel()`` on
        # large feature/edge tensors.
        positions = (
            torch.arange(max_items, dtype=torch.int64)
            * (int(flat.numel()) - 1)
            // (max_items - 1)
        )
        flat = flat[positions]
    hasher.update(flat.numpy().tobytes())


def dataset_fingerprint(data) -> str:
    """Cheap stable identity for graph tensors and the exact canonical split."""
    hasher = hashlib.sha256()
    hasher.update(f"n={data.num_nodes};e={data.edge_index.size(1)}".encode("ascii"))
    _hash_tensor_sample(hasher, data.edge_index)
    _hash_tensor_sample(hasher, data.x)
    _hash_tensor_sample(hasher, data.y)
    for key in ("train_mask", "val_mask", "test_mask"):
        idx = torch.where(getattr(data, key).cpu().bool())[0]
        _hash_tensor_sample(hasher, idx, max_items=max(8192, int(idx.numel())))
    return hasher.hexdigest()


def _homophily_labels(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, str]:
    y = torch.as_tensor(y).detach().cpu()
    if y.dim() == 1 or (y.dim() == 2 and y.size(1) == 1):
        labels = y.reshape(-1).long()
        valid = torch.isfinite(y.reshape(-1).float()) & (labels >= 0)
        return labels, valid, "single-label"

    finite = torch.isfinite(y.float())
    valid = finite.any(dim=1)
    safe_y = torch.where(finite, y.float(), torch.full_like(y.float(), -math.inf))
    labels = safe_y.argmax(dim=1).long()
    return labels, valid, "dominant-label proxy for multi-label targets"


def exact_homophily(data, chunk_edges: int = 2_000_000) -> tuple[float, float, str]:
    """Compute edge and node homophily in bounded-memory edge chunks."""
    labels, valid_node, definition = _homophily_labels(data.y)
    num_nodes = int(data.num_nodes)
    same_per_dst = torch.zeros(num_nodes, dtype=torch.float64)
    degree_per_dst = torch.zeros(num_nodes, dtype=torch.float64)
    same_edges = 0
    valid_edges = 0
    edge_index = data.edge_index.detach().cpu()

    for start in range(0, int(edge_index.size(1)), max(1, int(chunk_edges))):
        src, dst = edge_index[:, start : start + chunk_edges]
        keep = valid_node[src] & valid_node[dst]
        if not bool(keep.any()):
            continue
        dst = dst[keep]
        same = labels[src[keep]].eq(labels[dst])
        same_edges += int(same.sum())
        valid_edges += int(same.numel())
        degree_per_dst.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float64))
        same_per_dst.index_add_(0, dst, same.to(torch.float64))

    if valid_edges == 0:
        return float("nan"), float("nan"), definition
    active = degree_per_dst > 0
    edge_h = same_edges / valid_edges
    node_h = float((same_per_dst[active] / degree_per_dst[active]).mean())
    return node_h, edge_h, definition


def audit_dataset(name: str, data_dir: str, *, homophily: bool, chunk_edges: int) -> dict:
    data, _dataset = load_pyg_data(data_dir, name)
    canonical = canonicalize_dataset_name(name)
    y = torch.as_tensor(data.y)
    if y.dim() <= 1 or y.size(-1) == 1:
        flat_y = y.reshape(-1)
        labeled = flat_y[torch.isfinite(flat_y.float()) & (flat_y >= 0)]
        num_classes = int(labeled.max()) + 1 if labeled.numel() else 0
    else:
        num_classes = int(y.size(1))

    result = {
        "dataset": canonical,
        "data_dir": str(Path(data_dir).resolve()),
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "num_features": int(data.x.size(1)),
        "num_classes_or_tasks": num_classes,
        "train_nodes": int(data.train_mask.sum()),
        "valid_nodes": int(data.val_mask.sum()),
        "test_nodes": int(data.test_mask.sum()),
        "fingerprint": dataset_fingerprint(data),
    }
    if homophily:
        node_h, edge_h, definition = exact_homophily(data, chunk_edges=chunk_edges)
        result.update(
            node_homophily=node_h,
            edge_homophily=edge_h,
            homophily_definition=definition,
        )
    return result


def _print_result(result: dict) -> None:
    print(f"\n=== {result['dataset']} ===")
    print(
        f"  N={result['num_nodes']:,}  E={result['num_edges']:,}  "
        f"F={result['num_features']}  C/tasks={result['num_classes_or_tasks']}"
    )
    print(
        f"  train={result['train_nodes']:,}  valid={result['valid_nodes']:,}  "
        f"test={result['test_nodes']:,}"
    )
    if "node_homophily" in result:
        print(
            f"  node-homophily={result['node_homophily']:.6f}  "
            f"edge-homophily={result['edge_homophily']:.6f}  "
            f"({result['homophily_definition']})"
        )
    print(f"  fingerprint={result['fingerprint']}")


def run_audit(
    datasets: Iterable[str] = DATASETS,
    data_dir: str = DEFAULT_DATA_DIR,
    manifest_path: str | None = None,
    *,
    homophily: bool = True,
    chunk_edges: int = 2_000_000,
) -> list[dict]:
    data_root = Path(data_dir).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)
    if manifest_path is None:
        manifest_path = str(Path(DEFAULT_CACHE_DIR) / "dataset_audit" / "dataset_manifest.json")

    print(f"Canonical data directory: {data_root.resolve()}")
    results = []
    failures = []
    for name in datasets:
        try:
            result = audit_dataset(
                name,
                str(data_root),
                homophily=homophily,
                chunk_edges=chunk_edges,
            )
            results.append(result)
            _print_result(result)
        except Exception as exc:
            print(f"\n[FAIL] {name}: {type(exc).__name__}: {exc}")
            failures.append({"dataset": name, "error": f"{type(exc).__name__}: {exc}"})

    manifest = {
        "schema": 1,
        "data_dir": str(data_root.resolve()),
        "datasets": results,
        "failures": failures,
    }
    target = Path(manifest_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=True) + "\n")
    temp.replace(target)
    print(f"\nManifest: {target}")
    if failures:
        raise RuntimeError(f"{len(failures)} dataset audit(s) failed")
    return results


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=DATASETS)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--skip-homophily", action="store_true")
    parser.add_argument("--homophily-chunk-edges", type=int, default=2_000_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_audit(
            args.datasets,
            args.data_dir,
            args.manifest,
            homophily=not args.skip_homophily,
            chunk_edges=args.homophily_chunk_edges,
        )
    except RuntimeError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
