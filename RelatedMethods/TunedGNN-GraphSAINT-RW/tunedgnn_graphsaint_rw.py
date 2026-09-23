"""Train the tunedGNN GCN architecture on GraphSAINT random-walk samples."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parent
SUPPORT_GRAPH_ROOT = Path(
    os.environ.get("SUPPORT_GRAPH_ROOT", METHOD_ROOT.parents[1])
).resolve()
GRAPHSAINT_ENTRYPOINT = (
    SUPPORT_GRAPH_ROOT / "RelatedMethods" / "GraphSAINT-RW" / "graphsaint_rw.py"
)

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from model import TunedGNNGCN  # noqa: E402


def _load_graphsaint_backend():
    spec = importlib.util.spec_from_file_location(
        "_tunedgnn_graphsaint_backend", GRAPHSAINT_ENTRYPOINT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load GraphSAINT backend: {GRAPHSAINT_ENTRYPOINT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    backend = _load_graphsaint_backend()
    backend.main(
        model_factory=TunedGNNGCN,
        result_method="tunedgnn-graphsaint-rw",
        description=(
            "tunedGNN GCN trained with the GraphSAINT random-walk sampler"
        ),
        data_prep=TunedGNNGCN.prepare_data,
    )


if __name__ == "__main__":
    main()
