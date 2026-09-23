"""Persistent cache helpers for one-shot graph sparsification experiments."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data


CACHE_SCHEMA_VERSION = 1


def _ratio_tag(value: float) -> str:
    text = f"{float(value):.10f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return str(value)


def sampled_edge_positions(edge_count: int, sample_count: int = 256) -> torch.Tensor:
    """Return deterministic, in-range, legacy-compatible sample positions.

    ``torch.linspace`` defaults to float32, whose rounded endpoint can equal
    ``edge_count`` once an edge list is larger than 2**24.  That caused the
    cache fingerprint to fail before sparsification on Products, Proteins,
    and Pokec. Clamping fixes that failure while retaining every old in-range
    position, so existing Reddit and Arxiv cache keys remain reusable.
    """

    edge_count = max(0, int(edge_count))
    count = min(max(0, int(sample_count)), edge_count)
    if count == 0:
        return torch.empty(0, dtype=torch.long)
    if count == 1:
        return torch.zeros(1, dtype=torch.long)
    positions = torch.linspace(
        0, edge_count - 1, steps=count, dtype=torch.float32
    ).long()
    return positions.clamp_(min=0, max=edge_count - 1)


def sampled_edge_fingerprint(edge_index: torch.Tensor, num_nodes: int) -> str:
    """Cheaply distinguish source tensors without hashing every large-graph edge."""

    edges = edge_index.detach().cpu()
    count = int(edges.size(1))
    if count:
        positions = sampled_edge_positions(count)
        sample = edges[:, positions].contiguous().numpy().tobytes()
    else:
        sample = b""
    digest = hashlib.sha256()
    digest.update(f"{int(num_nodes)}:{count}:".encode("utf-8"))
    digest.update(sample)
    return digest.hexdigest()[:20]


def cache_entry(
    cache_root: str | os.PathLike[str] | None,
    *,
    dataset: str,
    sparsifier: str,
    target_ratio: float,
    seed: int,
    split_fingerprint: str,
    source_edge_index: torch.Tensor,
    num_nodes: int,
    num_undirected_edges: int,
    sparsifier_params: dict[str, Any],
) -> dict[str, Any] | None:
    if not cache_root or sparsifier == "full":
        return None

    # Explicit legacy still denotes the original distribution/cache identity.
    # Never strip the normalized mode or its alpha: those graphs must be isolated.
    sparsifier_params = dict(sparsifier_params)
    if sparsifier == "scaffold_sample" and sparsifier_params.get("sample_weight_mode") == "legacy":
        sparsifier_params.pop("sample_weight_mode")
        sparsifier_params.pop("sample_mix_alpha", None)

    identity = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset": str(dataset),
        "sparsifier": str(sparsifier),
        "target_ratio": float(target_ratio),
        "seed": int(seed),
        "split_fingerprint": str(split_fingerprint),
        "num_nodes": int(num_nodes),
        "num_undirected_edges": int(num_undirected_edges),
        "source_edge_fingerprint": sampled_edge_fingerprint(
            source_edge_index, num_nodes
        ),
        "sparsifier_params": _json_value(sparsifier_params),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    parameter_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    directory = (
        Path(cache_root).expanduser()
        / str(dataset)
        / str(sparsifier)
        / f"ratio_{_ratio_tag(target_ratio)}"
        / f"seed_{int(seed)}"
        / f"split_{str(split_fingerprint)[:16]}"
        / parameter_hash
    )
    networkit_dir = Path(cache_root).expanduser() / "networkit" / str(dataset)
    source_tag = (
        f"n{int(num_nodes)}_m{int(num_undirected_edges)}_"
        f"{identity['source_edge_fingerprint']}"
    )
    return {
        "identity": identity,
        "directory": directory,
        "graph_path": directory / "graph.pt",
        "metadata_path": directory / "metadata.json",
        "networkit_path": networkit_dir / f"{source_tag}.nkbg",
    }


def load_cached_graph(entry: dict[str, Any] | None, original_data: Data):
    if entry is None:
        return None
    graph_path = Path(entry["graph_path"])
    metadata_path = Path(entry["metadata_path"])
    if not graph_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("identity") != entry["identity"]:
            return None
        try:
            payload = torch.load(graph_path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch before weights_only support.
            payload = torch.load(graph_path, map_location="cpu")
        edge_index = payload["edge_index"].long().contiguous()
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            return None
        if edge_index.numel() and (
            int(edge_index.min()) < 0
            or int(edge_index.max()) >= int(original_data.num_nodes)
        ):
            return None
        edge_weight = payload.get("edge_weight")
        if (
            entry["identity"].get("sparsifier")
            in {"effective_resistance", "effective_resistance_fixed"}
            and edge_weight is None
        ):
            # ER training is weighted. Silently recompute old topology-only
            # cache entries produced before weighted sampling was supported.
            return None
        if edge_weight is not None:
            edge_weight = edge_weight.float().contiguous()
            if edge_weight.dim() != 1 or edge_weight.numel() != edge_index.size(1):
                return None
            if not bool(torch.isfinite(edge_weight).all()) or bool(
                (edge_weight <= 0).any()
            ):
                return None
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return None

    data = Data(
        x=original_data.x,
        edge_index=edge_index,
        y=original_data.y,
        num_nodes=original_data.num_nodes,
    )
    if edge_weight is not None:
        data.edge_weight = edge_weight
    for name in (
        "edge_index_is_undirected_unique",
        "edge_index_is_symmetric_unique",
        "num_undirected_edges",
        "num_self_loops",
    ):
        if name in payload:
            setattr(data, name, payload[name])
    return data, metadata


def save_cached_graph(
    entry: dict[str, Any] | None,
    sparsified_data: Data,
    *,
    sparsification_time_sec: float,
    before_stats: tuple[int, int, int],
    after_stats: tuple[int, int, int],
) -> dict[str, Any] | None:
    if entry is None:
        return None
    directory = Path(entry["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    graph_path = Path(entry["graph_path"])
    metadata_path = Path(entry["metadata_path"])
    payload: dict[str, Any] = {
        "edge_index": sparsified_data.edge_index.detach().cpu().long().contiguous()
    }
    source_edge_weight = getattr(sparsified_data, "edge_weight", None)
    if source_edge_weight is not None:
        edge_weight = source_edge_weight.detach().cpu().float().contiguous()
        if edge_weight.dim() != 1 or edge_weight.numel() != payload["edge_index"].size(1):
            raise ValueError("edge_weight must align with sparsified edge_index")
        if not bool(torch.isfinite(edge_weight).all()) or bool(
            (edge_weight <= 0).any()
        ):
            raise ValueError("edge_weight must be finite and strictly positive")
        payload["edge_weight"] = edge_weight
    for name in (
        "edge_index_is_undirected_unique",
        "edge_index_is_symmetric_unique",
        "num_undirected_edges",
        "num_self_loops",
    ):
        if hasattr(sparsified_data, name):
            payload[name] = getattr(sparsified_data, name)

    graph_tmp = directory / f".graph.{os.getpid()}.tmp"
    torch.save(payload, graph_tmp)
    os.replace(graph_tmp, graph_path)

    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity": entry["identity"],
        "sparsification_time_sec": float(sparsification_time_sec),
        "before": {
            "directed_edges": int(before_stats[0]),
            "undirected_edges": int(before_stats[1]),
            "self_loops": int(before_stats[2]),
        },
        "after": {
            "directed_edges": int(after_stats[0]),
            "undirected_edges": int(after_stats[1]),
            "self_loops": int(after_stats[2]),
        },
        "graph_path": str(graph_path.resolve()),
        "weighted": "edge_weight" in payload,
        "networkit_conversion_path": str(Path(entry["networkit_path"]).resolve()),
    }
    metadata_tmp = directory / f".metadata.{os.getpid()}.tmp"
    metadata_tmp.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(metadata_tmp, metadata_path)
    return metadata
