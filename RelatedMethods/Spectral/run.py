#!/usr/bin/env python3
"""Benchmark-aware launcher for Julia effective resistance and sparsification."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from export_benchmark_graph import (  # noqa: E402
    ALL_DATASETS,
    DEFAULT_DATA_ROOT,
    DEFAULT_SCAFFOLD_ROOT,
    export_dataset,
)


def _read_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as source:
        return tomllib.load(source)


def _run_one(args: argparse.Namespace, dataset: str) -> dict[str, object]:
    exported = export_dataset(
        dataset,
        data_root=args.data_root,
        benchmark_root=args.benchmark_root,
        seed=args.dataset_seed,
        split_protocol=args.split_protocol,
    )
    budget = args.budget
    if budget is None:
        budget = max(1, round(args.budget_ratio * exported.num_undirected_edges))

    print(
        f"\n[{exported.dataset}] nodes={exported.num_nodes} "
        f"directed_input_edges={exported.num_directed_input_edges} "
        f"undirected_edges={exported.num_undirected_edges} budget={budget}",
        flush=True,
    )
    print(
        f"[{exported.dataset}] graph_export_cached={exported.cached} "
        f"export_seconds={exported.export_seconds:.6f}",
        flush=True,
    )

    command = [
        args.julia,
        f"--project={PROJECT_ROOT}",
        f"--threads={args.threads}",
        str(PROJECT_ROOT / "bin" / "spectral_sparsify.jl"),
        "--input",
        str(exported.graph_path),
        "--output-dir",
        str(exported.artifact_dir),
        "--budget",
        str(budget),
        "--method",
        args.method,
        "--jl-factor",
        str(args.jl_factor),
        "--solver-tolerance",
        str(args.solver_tolerance),
        "--er-seed",
        str(args.er_seed),
        "--sample-seed",
        str(args.sample_seed),
        "--exact-max-nodes",
        str(args.exact_max_nodes),
    ]
    if args.force_er:
        command.append("--force-er")
    if args.force_sparsifier:
        command.append("--force-sparsifier")
    subprocess.run(command, check=True)

    er_metadata = _read_toml(exported.artifact_dir / "effective_resistance.toml")
    sparse_metadata = _read_toml(
        exported.artifact_dir / f"sparsified_budget_{budget}_seed_{args.sample_seed}.toml"
    )
    return {
        "dataset": exported.dataset,
        "num_nodes": exported.num_nodes,
        "num_input_directed_edges": exported.num_directed_input_edges,
        "num_input_undirected_edges": exported.num_undirected_edges,
        "budget_draws": budget,
        "num_output_undirected_edges": sparse_metadata["num_output_undirected_edges"],
        "graph_export_seconds": exported.export_seconds,
        "effective_resistance_seconds": er_metadata["compute_seconds"],
        "er_factor_seconds": er_metadata["factor_seconds"],
        "er_solve_seconds": er_metadata["solve_seconds"],
        "sparsification_seconds": sparse_metadata["compute_seconds"],
        "effective_resistance_path": str(exported.artifact_dir / "effective_resistance.npz"),
        "sparsifier_path": str(
            exported.artifact_dir / f"sparsified_budget_{budget}_seed_{args.sample_seed}.npz"
        ),
    }


def _write_report(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--dataset", nargs="+", choices=(*ALL_DATASETS, "karate"))
    selection.add_argument("--all", action="store_true", help="Run Benchmark's 19 experiment datasets")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--budget", type=int, help="Number of edge draws per dataset")
    budget.add_argument(
        "--budget-ratio",
        type=float,
        default=None,
        help="Edge draws as a fraction of canonical undirected edges (default: 0.5)",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_SCAFFOLD_ROOT)
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--threads", default="auto")
    parser.add_argument("--method", choices=("approx", "exact"), default="approx")
    parser.add_argument("--jl-factor", type=float, default=4.0)
    parser.add_argument("--solver-tolerance", type=float, default=1e-2)
    parser.add_argument("--er-seed", type=int, default=42)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--split-protocol", default="tunedgnn")
    parser.add_argument("--exact-max-nodes", type=int, default=5000)
    parser.add_argument("--force-er", action="store_true")
    parser.add_argument("--force-sparsifier", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "results" / "timings.csv",
    )
    args = parser.parse_args()
    if args.budget_ratio is None and args.budget is None:
        args.budget_ratio = 0.5
    if args.budget is not None and args.budget <= 0:
        parser.error("--budget must be positive")
    if args.budget_ratio is not None and not 0 < args.budget_ratio:
        parser.error("--budget-ratio must be positive")

    datasets = ALL_DATASETS if args.all else args.dataset
    rows = [_run_one(args, dataset) for dataset in datasets]
    _write_report(rows, args.report)
    print("\nTiming summary")
    for row in rows:
        print(
            f"  {row['dataset']}: ER={float(row['effective_resistance_seconds']):.6f}s "
            f"sparsify={float(row['sparsification_seconds']):.6f}s "
            f"unique_edges={row['num_output_undirected_edges']}/{row['num_input_undirected_edges']}"
        )
    print(f"Report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
