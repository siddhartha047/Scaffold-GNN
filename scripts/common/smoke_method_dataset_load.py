"""Load one dataset through the shared method adapter and record its identity."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scaffold_gnn.utils.dataset_smoke import audit_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--homophily", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    method = args.method.strip().lower()
    os.environ["BASELINE_METHOD_NAME"] = method
    os.environ["BASELINE_PARTITION_MODE"] = "none"

    result = audit_dataset(
        args.dataset,
        args.data_dir,
        homophily=args.homophily,
        chunk_edges=2_000_000,
    )
    result.update(method=method, status="PASS")

    output = Path(args.cache_dir) / "dataset_load_smoke" / method / f"{result['dataset']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)

    print(
        "[DatasetLoadOnly] "
        f"status=PASS method={method} dataset={result['dataset']} "
        f"data_root={result['data_dir']} nodes={result['num_nodes']} "
        f"edges={result['num_edges']} features={result['num_features']} "
        f"train={result['train_nodes']} valid={result['valid_nodes']} test={result['test_nodes']} "
        f"fingerprint={result['fingerprint']} manifest={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
