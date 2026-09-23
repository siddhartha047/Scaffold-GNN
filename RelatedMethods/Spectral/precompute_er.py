#!/usr/bin/env python3
"""Precompute edge-aligned ER artifacts with a memory-safe backend dispatcher."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import threading
import time
import tomllib
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from export_benchmark_graph import (  # noqa: E402
    ALL_DATASETS,
    DEFAULT_DATA_ROOT,
    DEFAULT_SCAFFOLD_ROOT,
    ExportResult,
    artifact_directory,
    export_dataset,
)
from finalize_tgt import finalize_tgt  # noqa: E402
from tgt_binary import (  # noqa: E402
    InputGraphInfo,
    _shape,
    convert_to_tgt_binary,
    inspect_input_graph,
)


DEFAULT_DATASET_TABLE = Path(
    "./results/accuracy.csv"
)
DEFAULT_TGT_EXECUTABLE = PROJECT_ROOT / "build/large_graph/tgt_effective_resistance"


@dataclass(frozen=True)
class PreparedGraph:
    info: InputGraphInfo
    artifact_dir: Path
    export_seconds: float
    cached: bool


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _datasets_from_table(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = csv.DictReader(source)
        if rows.fieldnames is None or "dataset" not in rows.fieldnames:
            raise ValueError(f"{path} does not contain a dataset column")
        datasets: list[str] = []
        seen: set[str] = set()
        for row in rows:
            dataset = row["dataset"].strip()
            if dataset and dataset not in seen:
                datasets.append(dataset)
                seen.add(dataset)
    unknown = [dataset for dataset in datasets if dataset not in ALL_DATASETS]
    if unknown:
        raise ValueError(f"Unsupported datasets in {path}: {', '.join(unknown)}")
    return datasets


@contextmanager
def _heartbeat(dataset: str, stage: str, seconds: int):
    stopped = threading.Event()
    started = time.monotonic()

    def report() -> None:
        while not stopped.wait(seconds):
            print(
                f"[{_timestamp()}] HEARTBEAT dataset={dataset} stage={stage} "
                f"elapsed_seconds={time.monotonic() - started:.0f}",
                flush=True,
            )

    worker = threading.Thread(target=report, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join()


def _write_status(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _prepare_graph(args: argparse.Namespace, dataset: str) -> PreparedGraph:
    artifact_dir = artifact_directory(args.data_root.expanduser().resolve(), dataset)
    graph_path = artifact_dir / "input_graph.npz"
    if graph_path.is_file() and not args.force_export:
        info = inspect_input_graph(graph_path)
        if info.dataset != dataset:
            raise ValueError(f"{graph_path} contains {info.dataset}, expected {dataset}")
        return PreparedGraph(info, artifact_dir, 0.0, True)
    exported: ExportResult = export_dataset(
        dataset,
        data_root=args.data_root,
        benchmark_root=args.benchmark_root,
        seed=args.dataset_seed,
        split_protocol=args.split_protocol,
    )
    return PreparedGraph(
        inspect_input_graph(exported.graph_path),
        exported.artifact_dir,
        exported.export_seconds,
        exported.cached,
    )


def _load_metadata(path: Path) -> dict[str, object] | None:
    try:
        with path.open("rb") as source:
            return tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError):
        return None


def _complete_cache(graph: PreparedGraph) -> dict[str, object] | None:
    path = graph.artifact_dir / "effective_resistance_directed.npz"
    metadata = _load_metadata(graph.artifact_dir / "effective_resistance_directed.toml")
    if not path.is_file() or metadata is None:
        return None
    if (
        metadata.get("graph_fingerprint") != graph.info.graph_fingerprint
        or metadata.get("directed_fingerprint") != graph.info.directed_fingerprint
        or int(metadata.get("num_directed_edges", -1)) != graph.info.num_directed_edges
    ):
        return None
    try:
        with zipfile.ZipFile(path, "r") as values:
            if _shape(values, "resistance") != (graph.info.num_directed_edges,):
                return None
    except (KeyError, OSError, ValueError, zipfile.BadZipFile):
        return None
    return metadata


def _select_backend(args: argparse.Namespace, graph: PreparedGraph) -> tuple[str, str]:
    if args.backend != "auto":
        return args.backend, "explicit --backend"
    small = (
        graph.info.num_nodes <= args.laplacians_max_nodes
        and graph.info.num_undirected_edges <= args.laplacians_max_edges
    )
    if small:
        return "laplacians", (
            f"nodes<={args.laplacians_max_nodes} and "
            f"edges<={args.laplacians_max_edges}"
        )
    return "jlpcg", (
        f"graph exceeds Laplacians.jl limits "
        f"({args.laplacians_max_nodes} nodes, {args.laplacians_max_edges} edges)"
    )


def _run_laplacians(
    args: argparse.Namespace,
    graph: PreparedGraph,
    dataset_started: float,
) -> dict[str, object]:
    command = [
        args.julia,
        "--startup-file=no",
        "--history-file=no",
        "--compiled-modules=existing",
        f"--project={PROJECT_ROOT}",
        f"--threads={args.threads}",
        str(PROJECT_ROOT / "bin/precompute_er.jl"),
        "--input",
        str(graph.info.path),
        "--output-dir",
        str(graph.artifact_dir),
        "--method",
        "approx",
        "--jl-factor",
        str(args.jl_factor),
        "--solver-tolerance",
        str(args.solver_tolerance),
        "--er-seed",
        str(args.er_seed),
        "--exact-max-nodes",
        str(args.exact_max_nodes),
        "--progress-every",
        str(args.progress_every),
    ]
    if args.force_er:
        command.append("--force-er")
    environment = os.environ.copy()
    # The Miniconda module prepends its own GLib to LD_LIBRARY_PATH.  Julia's
    # Glib_jll must load the matching artifact libraries instead; otherwise
    # compute nodes fail during module initialization with g_dir_unref missing.
    environment.pop("LD_LIBRARY_PATH", None)
    environment.pop("LD_PRELOAD", None)
    environment.update(
        {
            "JULIA_DEPOT_PATH": (
                str(Path.home() / ".julia")
            ),
            "JULIA_NUM_THREADS": str(args.threads),
            "JULIA_PKG_PRECOMPILE_AUTO": "0",
            "JULIA_NUM_PRECOMPILE_TASKS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
    )
    started = time.monotonic()
    remaining = args.max_dataset_seconds - elapsed(dataset_started)
    if remaining < 1.0:
        raise TimeoutError("dataset deadline expired before Julia started")
    subprocess.run(command, check=True, env=environment, timeout=remaining)
    metadata = _complete_cache(graph)
    if metadata is None:
        raise RuntimeError("Laplacians.jl exited without a valid directed ER artifact")
    return {**metadata, "compute_seconds": metadata.get("compute_seconds", elapsed(started))}


def elapsed(started: float) -> float:
    return time.monotonic() - started


def _run_tgt(
    args: argparse.Namespace,
    graph: PreparedGraph,
    dataset_started: float,
) -> dict[str, object]:
    executable = args.tgt_executable.expanduser().resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(
            f"TGT executable is missing: {executable}; run scripts/build_tgt.sh"
        )
    conversion_started = time.monotonic()
    binary = convert_to_tgt_binary(
        graph.info.path,
        graph.artifact_dir / "input_graph.tgtbin",
        force=args.force_binary,
    )
    print(
        f"[{_timestamp()}] TGT_BINARY dataset={graph.info.dataset} "
        f"cached={binary.cached} bytes={binary.file_bytes} "
        f"seconds={elapsed(conversion_started):.3f}",
        flush=True,
    )
    work = graph.artifact_dir / "tgt_work"
    work.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable),
        "--input",
        str(binary.path),
        "--eigen-cache",
        str(work / f"eigen_omega_{args.omega}.bin"),
        "--oriented-output",
        str(work / "oriented_er.bin"),
        "--threads",
        str(args.threads),
        "--epsilon",
        str(args.epsilon),
        "--delta",
        str(args.delta),
        "--omega",
        str(args.omega),
        "--gamma",
        str(args.gamma),
        "--seed",
        str(args.er_seed),
        "--eigen-tolerance",
        str(args.eigen_tolerance),
        "--max-iterations",
        str(args.max_iterations),
        "--block-sources",
        str(args.block_sources),
        "--memory-gb",
        str(args.memory_gb),
    ]
    if args.force_er:
        command.append("--force")
    started = time.monotonic()
    remaining = args.max_dataset_seconds - elapsed(dataset_started)
    if remaining < 1.0:
        raise TimeoutError("dataset deadline expired before TGT+ started")
    subprocess.run(command, check=True, timeout=remaining)
    compute_seconds = elapsed(started)
    return finalize_tgt(
        binary,
        work / "oriented_er.bin",
        graph.artifact_dir,
        compute_seconds=compute_seconds,
    )


def _run_jlpcg(
    args: argparse.Namespace,
    graph: PreparedGraph,
    dataset_started: float,
) -> dict[str, object]:
    executable = args.tgt_executable.expanduser().resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(
            f"large-graph executable is missing: {executable}; run scripts/build_tgt.sh"
        )
    conversion_started = time.monotonic()
    binary = convert_to_tgt_binary(
        graph.info.path,
        graph.artifact_dir / "input_graph.tgtbin",
        force=args.force_binary,
    )
    print(
        f"[{_timestamp()}] JLPCG_BINARY dataset={graph.info.dataset} "
        f"cached={binary.cached} bytes={binary.file_bytes} "
        f"seconds={elapsed(conversion_started):.3f}",
        flush=True,
    )
    remaining = (
        args.max_dataset_seconds
        - elapsed(dataset_started)
        - args.finalize_reserve_seconds
    )
    if remaining < 60.0:
        raise TimeoutError(
            "less than 60 seconds remain after reserving artifact-finalization time"
        )
    work = graph.artifact_dir / "jl_pcg_work"
    work.mkdir(parents=True, exist_ok=True)
    tolerance_tag = format(args.pcg_tolerance, ".3g").replace(".", "p")
    oriented = work / (
        f"oriented_p{args.large_projections}_i{args.pcg_iterations}_"
        f"t{tolerance_tag}_s{args.er_seed}.bin"
    )
    command = [
        str(executable),
        "--algorithm",
        "jl-pcg",
        "--input",
        str(binary.path),
        "--eigen-cache",
        str(work / "unused_eigen_cache.bin"),
        "--oriented-output",
        str(oriented),
        "--threads",
        str(args.threads),
        "--omega",
        str(args.large_projections),
        "--gamma",
        "-1",
        "--seed",
        str(args.er_seed),
        "--pcg-tolerance",
        str(args.pcg_tolerance),
        "--pcg-iterations",
        str(args.pcg_iterations),
        "--min-projections",
        str(args.min_projections),
        "--max-seconds",
        str(remaining),
        "--memory-gb",
        str(args.memory_gb),
    ]
    if args.force_er:
        command.append("--force")
    started = time.monotonic()
    subprocess.run(command, check=True, timeout=remaining + 60.0)
    compute_seconds = elapsed(started)
    return finalize_tgt(
        binary,
        oriented,
        graph.artifact_dir,
        compute_seconds=compute_seconds,
        backend="jl_pcg_cpp",
        extra_metadata={
            "projection_target": args.large_projections,
            "pcg_tolerance": args.pcg_tolerance,
            "pcg_max_iterations": args.pcg_iterations,
            "dataset_time_limit_seconds": args.max_dataset_seconds,
            "finalize_reserve_seconds": args.finalize_reserve_seconds,
        },
    )


def _run_dataset(
    args: argparse.Namespace, dataset: str, index: int, total: int
) -> dict[str, object]:
    print(f"\n[{_timestamp()}] DATASET_START index={index}/{total} dataset={dataset}", flush=True)
    wall_started = time.monotonic()
    with _heartbeat(dataset, "graph_prepare", args.heartbeat_seconds):
        graph = _prepare_graph(args, dataset)
    print(
        f"[{_timestamp()}] GRAPH_READY dataset={dataset} cached={graph.cached} "
        f"nodes={graph.info.num_nodes} directed_edges={graph.info.num_directed_edges} "
        f"undirected_edges={graph.info.num_undirected_edges} "
        f"seconds={graph.export_seconds:.3f}",
        flush=True,
    )

    cached_metadata = None if args.force_er else _complete_cache(graph)
    if cached_metadata is not None:
        backend = str(cached_metadata.get("backend", cached_metadata.get("method", "cached")))
        metadata = cached_metadata
        status = "cached"
        print(
            f"[{_timestamp()}] ER_CACHE_HIT dataset={dataset} backend={backend} "
            f"directed_er_values={graph.info.num_directed_edges}",
            flush=True,
        )
    else:
        backend, reason = _select_backend(args, graph)
        print(
            f"[{_timestamp()}] BACKEND_SELECTED dataset={dataset} backend={backend} "
            f"reason={reason}",
            flush=True,
        )
        with _heartbeat(dataset, f"{backend}_effective_resistance", args.heartbeat_seconds):
            metadata = (
                _run_laplacians(args, graph, wall_started)
                if backend == "laplacians"
                else (
                    _run_tgt(args, graph, wall_started)
                    if backend == "tgt"
                    else _run_jlpcg(args, graph, wall_started)
                )
            )
        status = "complete"

    wall_seconds = elapsed(wall_started)
    print(
        f"[{_timestamp()}] DATASET_DONE index={index}/{total} dataset={dataset} "
        f"backend={backend} directed_er_values={graph.info.num_directed_edges} "
        f"wall_seconds={wall_seconds:.3f}",
        flush=True,
    )
    return {
        "dataset": dataset,
        "status": status,
        "backend": backend,
        "num_nodes": graph.info.num_nodes,
        "num_directed_edges": graph.info.num_directed_edges,
        "num_undirected_edges": graph.info.num_undirected_edges,
        "er_compute_seconds": metadata.get("compute_seconds", ""),
        "finalize_seconds": metadata.get(
            "finalize_seconds", metadata.get("write_seconds", "")
        ),
        "wall_seconds": wall_seconds,
        "artifact": str(graph.artifact_dir / "effective_resistance_directed.npz"),
        "error": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--dataset", nargs="+", choices=ALL_DATASETS)
    parser.add_argument("--dataset-table", type=Path, default=DEFAULT_DATASET_TABLE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_SCAFFOLD_ROOT)
    parser.add_argument(
        "--backend",
        choices=("auto", "laplacians", "jlpcg", "tgt"),
        default="auto",
    )
    parser.add_argument("--laplacians-max-nodes", type=int, default=200_000)
    parser.add_argument("--laplacians-max-edges", type=int, default=2_000_000)
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--tgt-executable", type=Path, default=DEFAULT_TGT_EXECUTABLE)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-gb", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--omega", type=int, default=128)
    parser.add_argument("--gamma", type=int, default=10)
    parser.add_argument("--eigen-tolerance", type=float, default=1e-6)
    parser.add_argument("--max-iterations", type=int, default=10_000)
    parser.add_argument("--block-sources", type=int, default=1_024)
    parser.add_argument("--large-projections", type=int, default=64)
    parser.add_argument("--min-projections", type=int, default=16)
    parser.add_argument("--pcg-tolerance", type=float, default=1e-2)
    parser.add_argument("--pcg-iterations", type=int, default=100)
    parser.add_argument("--max-dataset-seconds", type=float, default=21_600.0)
    parser.add_argument("--finalize-reserve-seconds", type=float, default=900.0)
    parser.add_argument("--jl-factor", type=float, default=4.0)
    parser.add_argument("--solver-tolerance", type=float, default=1e-2)
    parser.add_argument("--er-seed", type=int, default=42)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--split-protocol", default="tunedgnn")
    parser.add_argument("--exact-max-nodes", type=int, default=5_000)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("--force-er", action="store_true")
    parser.add_argument("--force-binary", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--status-file",
        type=Path,
        default=PROJECT_ROOT / "results/er_precompute_status.csv",
    )
    args = parser.parse_args()
    positive = (
        "threads",
        "laplacians_max_nodes",
        "laplacians_max_edges",
        "omega",
        "block_sources",
        "large_projections",
        "min_projections",
        "pcg_iterations",
        "heartbeat_seconds",
        "progress_every",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.memory_gb < 0:
        parser.error("--memory-gb cannot be negative")
    if args.min_projections > args.large_projections:
        parser.error("--min-projections cannot exceed --large-projections")
    if not 0.0 < args.pcg_tolerance < 1.0:
        parser.error("--pcg-tolerance must be between zero and one")
    if args.max_dataset_seconds <= 0 or args.finalize_reserve_seconds < 0:
        parser.error("dataset/finalization time limits are invalid")
    if args.finalize_reserve_seconds + 60 >= args.max_dataset_seconds:
        parser.error("--finalize-reserve-seconds leaves no compute time")

    datasets = _datasets_from_table(args.dataset_table) if args.all else args.dataset
    print(
        f"[{_timestamp()}] BATCH_START datasets={len(datasets)} backend={args.backend} "
        f"threads={args.threads} memory_gb={args.memory_gb} table={args.dataset_table}",
        flush=True,
    )
    rows: list[dict[str, object]] = []
    failures = 0
    for index, dataset in enumerate(datasets, start=1):
        try:
            row = _run_dataset(args, dataset, index, len(datasets))
        except Exception as error:
            failures += 1
            print(
                f"[{_timestamp()}] DATASET_FAILED index={index}/{len(datasets)} "
                f"dataset={dataset} error={error!r}",
                flush=True,
            )
            row = {
                "dataset": dataset,
                "status": "failed",
                "backend": "",
                "num_nodes": "",
                "num_directed_edges": "",
                "num_undirected_edges": "",
                "er_compute_seconds": "",
                "finalize_seconds": "",
                "wall_seconds": "",
                "artifact": "",
                "error": repr(error),
            }
            if not args.keep_going:
                rows.append(row)
                _write_status(rows, args.status_file)
                raise
        rows.append(row)
        _write_status(rows, args.status_file)

    print(
        f"[{_timestamp()}] BATCH_DONE complete={len(datasets) - failures} "
        f"failed={failures} status_file={args.status_file}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
