"""Publish restartable TGT binary output as Benchmark-compatible NPZ artifacts."""

from __future__ import annotations

import os
import shutil
import struct
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tgt_binary import UINT32_MAX, TGTBinaryInfo, _chunks


_ORIENTED = struct.Struct("<8sIIQQQ64sddiIIIQQQQ")


def _bytes(value: str) -> np.ndarray:
    return np.frombuffer(value.encode("utf-8"), dtype=np.uint8).copy()


def _write_array(path: Path, array: np.ndarray) -> None:
    with path.open("wb") as output:
        np.lib.format.write_array(output, array, allow_pickle=False)


def _copy_member(
    source_zip: zipfile.ZipFile,
    output_zip: zipfile.ZipFile,
    source_name: str,
    output_name: str | None = None,
) -> None:
    name = source_name if source_name.endswith(".npy") else f"{source_name}.npy"
    target = output_name or name
    info = source_zip.getinfo(name)
    target_info = zipfile.ZipInfo(target)
    target_info.compress_type = zipfile.ZIP_STORED
    target_info.external_attr = 0o600 << 16
    with source_zip.open(info, "r") as source, output_zip.open(
        target_info, "w", force_zip64=True
    ) as output:
        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)


def _add_file(zip_file: zipfile.ZipFile, path: Path, name: str) -> None:
    info = zipfile.ZipInfo(name)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o600 << 16
    with path.open("rb") as source, zip_file.open(info, "w", force_zip64=True) as output:
        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)


def _atomic_toml(path: Path, values: dict[str, object]) -> None:
    lines: list[str] = []
    for key in sorted(values):
        value = values[key]
        if isinstance(value, bool):
            encoded = "true" if value else "false"
        elif isinstance(value, str):
            encoded = '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
        elif isinstance(value, float):
            encoded = repr(value)
        else:
            encoded = str(value)
        lines.append(f"{key} = {encoded}\n")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.writelines(lines)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _load_oriented(path: Path, binary: TGTBinaryInfo):
    with path.open("rb") as source:
        payload = source.read(_ORIENTED.size)
    if len(payload) != _ORIENTED.size:
        raise ValueError(f"{path} has a truncated oriented-result header")
    header = _ORIENTED.unpack(payload)
    (
        magic,
        version,
        omega,
        num_nodes,
        num_edges,
        completed,
        fingerprint,
        epsilon,
        delta,
        gamma,
        threads,
        edge_components,
        primary_nodes,
        seed,
        low_offset,
        high_offset,
        file_bytes,
    ) = header
    if not (
        magic == b"SSORI001"
        and version == 2
        and num_nodes == binary.graph.num_nodes
        and num_edges == binary.graph.num_undirected_edges
        and completed == num_nodes
        and fingerprint.rstrip(b"\0").decode("ascii")
        == binary.graph.graph_fingerprint
        and file_bytes == path.stat().st_size
    ):
        raise ValueError(f"{path} is incomplete or belongs to another graph")
    low = np.memmap(path, dtype="<f4", mode="r", offset=low_offset, shape=(num_edges,))
    high = np.memmap(path, dtype="<f4", mode="r", offset=high_offset, shape=(num_edges,))
    parameters = {
        "threads": threads,
        "edge_components": edge_components,
        "primary_nodes": primary_nodes,
        "seed": seed,
    }
    if gamma < 0:
        encoded = -gamma
        if encoded >= 1_000_000:
            projection_code = encoded - 1_000_000
            projections_completed = projection_code // 2
            finalized_buffer = projection_code % 2
        else:
            # Compatibility with the first JL-PCG artifact format, which
            # finalized into low[] and cleared high[].
            projections_completed = encoded
            finalized_buffer = -1
        parameters.update(
            {
                "projection_target": omega,
                "projections_completed": projections_completed,
                "_jl_finalized_buffer": finalized_buffer,
            }
        )
    else:
        parameters.update(
            {
                "omega": omega,
                "epsilon": epsilon,
                "delta": delta,
                "gamma": gamma,
            }
        )
    return low, high, parameters


def _active_nodes(input_npz: Path, num_nodes: int) -> int:
    active = np.zeros(num_nodes, dtype=np.bool_)
    with zipfile.ZipFile(input_npz, "r") as values:
        for name in ("src", "dst"):
            for chunk in _chunks(values, name):
                active[np.asarray(chunk, dtype=np.int64)] = True
    return int(active.sum())


def finalize_tgt(
    binary: TGTBinaryInfo,
    oriented_path: Path,
    output_dir: Path,
    *,
    compute_seconds: float,
    backend: str | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    low, high, parameters = _load_oriented(oriented_path, binary)
    jl_finalized_buffer = int(parameters.pop("_jl_finalized_buffer", -1))
    canonical_path = output_dir / "effective_resistance.npz"
    directed_path = output_dir / "effective_resistance_directed.npz"

    with tempfile.TemporaryDirectory(prefix=".tgt-finalize.", dir=output_dir) as name:
        work = Path(name)
        resistance_path = work / "resistance.npy"
        canonical = np.lib.format.open_memmap(
            resistance_path,
            mode="w+",
            dtype="<f4",
            shape=(binary.graph.num_undirected_edges,),
        )
        clipped_low = 0
        clipped_high = 0
        leverage_sum = 0.0
        chunk_size = 2_000_000
        for first in range(0, len(canonical), chunk_size):
            last = min(len(canonical), first + chunk_size)
            if jl_finalized_buffer == 0:
                values = np.asarray(low[first:last], dtype=np.float64)
            elif jl_finalized_buffer == 1:
                values = np.asarray(high[first:last], dtype=np.float64)
            else:
                values = np.asarray(low[first:last], dtype=np.float64) + np.asarray(
                    high[first:last], dtype=np.float64
                )
            if not np.isfinite(values).all():
                raise ValueError("TGT result contains NaN or Inf")
            clipped_low += int((values < 0.0).sum())
            clipped_high += int((values > 1.0).sum())
            values = np.clip(values, 0.0, 1.0)
            canonical[first:last] = values
            leverage_sum += float(values.sum(dtype=np.float64))
        canonical.flush()
        del canonical

        scalar_files: dict[str, Path] = {}
        for key, value in {
            "graph_fingerprint": _bytes(binary.graph.graph_fingerprint),
            "dataset": _bytes(binary.graph.dataset),
        }.items():
            target = work / f"{key}.npy"
            _write_array(target, value)
            scalar_files[key] = target

        canonical_temporary = work / "effective_resistance.npz"
        with zipfile.ZipFile(
            canonical_temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as output, zipfile.ZipFile(binary.graph.path, "r") as input_values:
            _copy_member(input_values, output, "src")
            _copy_member(input_values, output, "dst")
            _copy_member(input_values, output, "weight", "conductance.npy")
            _add_file(output, resistance_path, "resistance.npy")
            _add_file(output, resistance_path, "leverage_score.npy")
            for key, path in scalar_files.items():
                _add_file(output, path, f"{key}.npy")

        directed_values_path = work / "directed_resistance.npy"
        directed = np.lib.format.open_memmap(
            directed_values_path,
            mode="w+",
            dtype="<f4",
            shape=(binary.graph.num_directed_edges,),
        )
        canonical_read = np.load(resistance_path, mmap_mode="r", allow_pickle=False)
        mapping = np.memmap(
            binary.path,
            dtype="<u4",
            mode="r",
            offset=binary.mapping_offset,
            shape=(binary.graph.num_directed_edges,),
        )
        for first in range(0, len(directed), chunk_size):
            last = min(len(directed), first + chunk_size)
            indices = np.asarray(mapping[first:last], dtype=np.uint32)
            loops = indices == UINT32_MAX
            safe = indices.copy()
            safe[loops] = 0
            values = np.asarray(canonical_read[safe], dtype=np.float32)
            values[loops] = 0.0
            directed[first:last] = values
        directed.flush()
        del directed, canonical_read, mapping

        directed_scalars = {
            "num_directed_edges": np.asarray(
                [binary.graph.num_directed_edges], dtype=np.int64
            ),
            "graph_fingerprint": _bytes(binary.graph.graph_fingerprint),
            "directed_fingerprint": _bytes(binary.graph.directed_fingerprint),
            "dataset": _bytes(binary.graph.dataset),
        }
        directed_temporary = work / "effective_resistance_directed.npz"
        with zipfile.ZipFile(
            directed_temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as output:
            _add_file(output, directed_values_path, "resistance.npy")
            for key, value in directed_scalars.items():
                target = work / f"directed_{key}.npy"
                _write_array(target, value)
                _add_file(output, target, f"{key}.npy")

        os.replace(canonical_temporary, canonical_path)
        os.replace(directed_temporary, directed_path)

    active_nodes = _active_nodes(binary.graph.path, binary.graph.num_nodes)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    inferred_backend = (
        "jl_pcg_cpp" if "projections_completed" in parameters else "tgt_plus_cpp"
    )
    common: dict[str, object] = {
        "schema_version": 6,
        "dataset": binary.graph.dataset,
        "backend": backend or inferred_backend,
        "graph_fingerprint": binary.graph.graph_fingerprint,
        "created_at_utc": created,
        "num_nodes": binary.graph.num_nodes,
        "num_active_nodes": active_nodes,
        "num_undirected_edges": binary.graph.num_undirected_edges,
        "compute_seconds": float(compute_seconds),
        "finalize_seconds": time.perf_counter() - started,
        "leverage_sum": leverage_sum,
        "kirchhoff_target": active_nodes - int(parameters["edge_components"]),
        "clipped_below_zero": clipped_low,
        "clipped_above_one": clipped_high,
        **parameters,
        **(extra_metadata or {}),
    }
    _atomic_toml(output_dir / "effective_resistance.toml", common)
    directed_metadata = {
        **common,
        "directed_fingerprint": binary.graph.directed_fingerprint,
        "num_directed_edges": binary.graph.num_directed_edges,
        "edge_order": "Benchmark data.edge_index columns",
        "self_loop_resistance": 0.0,
    }
    _atomic_toml(output_dir / "effective_resistance_directed.toml", directed_metadata)
    return {
        **common,
        "artifact": str(directed_path),
        "canonical_artifact": str(canonical_path),
        "num_directed_edges": binary.graph.num_directed_edges,
    }


__all__ = ["finalize_tgt"]
