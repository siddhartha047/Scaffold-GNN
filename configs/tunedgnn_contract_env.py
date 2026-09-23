"""Export the shared TunedGNN preset as shell environment assignments."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import shlex
import sys
from pathlib import Path

# Load the sibling table by path so this works both as ``python -m
# config.tunedgnn_contract_env`` and as a direct script invocation, and is not
# affected by another ``config`` package on PYTHONPATH.
_PRESETS_PATH = Path(__file__).resolve().parent / "tunedgnn_presets.py"
_PRESETS_SPEC = importlib.util.spec_from_file_location(
    "_tunedgnn_contract_presets", _PRESETS_PATH
)
if _PRESETS_SPEC is None or _PRESETS_SPEC.loader is None:
    raise ImportError(f"Cannot load tunedGNN presets from {_PRESETS_PATH}")
_PRESETS = importlib.util.module_from_spec(_PRESETS_SPEC)
sys.modules[_PRESETS_SPEC.name] = _PRESETS
_PRESETS_SPEC.loader.exec_module(_PRESETS)

TUNEDGNN_PRESET_PROFILES = _PRESETS.TUNEDGNN_PRESET_PROFILES
get_tunedgnn_preset = _PRESETS.get_tunedgnn_preset


# Mirrors TUNEDGNN_COMPARISON_MODELS in Benchmark/scripts/Configuration.py.
# Everything else is a GCN-family comparison, matching that file's
# ``.get(method, "gcn")`` default.
METHOD_MODELS = {
    "tunedgnn": "gcn",
    "tunedgnn-graphsage": "sage",
}

# Same alias set the SCAFFOLD translator uses, kept local so this module does
# not have to import the sparsifiers package.
DATASET_ALIASES = {
    "amazon-computers": "amazon-computer",
    "ogb-products": "ogbn-products",
    "ogb-product": "ogbn-products",
    "ogbn-product": "ogbn-products",
    "product": "ogbn-products",
    "products": "ogbn-products",
    "ogb-arxiv": "ogbn-arxiv",
    "arxiv": "ogbn-arxiv",
    "ogb-protein": "ogbn-proteins",
    "ogb-proteins": "ogbn-proteins",
    "ogbn-protein": "ogbn-proteins",
    "protein": "ogbn-proteins",
    "proteins": "ogbn-proteins",
}

# preset field -> wrapper environment variable
CONTRACT_FIELDS = (
    ("hidden_channels", "TUNEDGNN_HIDDEN_CHANNELS"),
    ("num_layers", "TUNEDGNN_NUM_LAYERS"),
    ("heads", "TUNEDGNN_HEADS"),
    ("learning_rate", "TUNEDGNN_LEARNING_RATE"),
    ("weight_decay", "TUNEDGNN_WEIGHT_DECAY"),
    ("dropout", "TUNEDGNN_DROPOUT"),
    ("input_dropout", "TUNEDGNN_INPUT_DROPOUT"),
    ("metric", "TUNEDGNN_METRIC"),
    ("pre_linear", "TUNEDGNN_PRE_LINEAR"),
    ("residual_connections", "TUNEDGNN_RESIDUAL_CONNECTIONS"),
    ("layer_norm", "TUNEDGNN_LAYER_NORM"),
    ("batch_norm", "TUNEDGNN_BATCH_NORM"),
    ("jumping_knowledge", "TUNEDGNN_JUMPING_KNOWLEDGE"),
    ("eval_every", "TUNEDGNN_EVAL_EVERY"),
    ("eval_start_epoch", "TUNEDGNN_EVAL_START_EPOCH"),
    ("log_every", "TUNEDGNN_LOG_EVERY"),
)


def canonical_dataset(name: str) -> str:
    key = "-".join(str(name).strip().lower().replace("_", "-").split())
    return DATASET_ALIASES.get(key, key)


def model_for_method(method: str) -> str:
    key = "-".join(str(method).strip().lower().replace("_", "-").split())
    return METHOD_MODELS.get(key, "gcn")


def _render(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return ""
    return str(value)


def contract_env(dataset: str, model: str, profile: str = "original") -> dict[str, str]:
    """Return the wrapper environment for one dataset/model/profile."""

    preset = get_tunedgnn_preset(canonical_dataset(dataset), model, profile)
    if preset is None:
        return {}
    values = asdict(preset)
    return {
        variable: _render(values[field])
        for field, variable in CONTRACT_FIELDS
        if field in values
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default=None, help="gcn or sage")
    parser.add_argument(
        "--method",
        default=None,
        help="Resolve the model family from a comparison method name instead.",
    )
    parser.add_argument(
        "--profile",
        default="original",
        choices=TUNEDGNN_PRESET_PROFILES,
    )
    args = parser.parse_args(argv)

    model = args.model or (
        model_for_method(args.method) if args.method else "gcn"
    )
    environment = contract_env(args.dataset, model, args.profile)
    if not environment:
        print(
            f"No tunedGNN {args.profile} preset for dataset "
            f"'{args.dataset}' model '{model}'",
            file=sys.stderr,
        )
        return 1
    for variable, value in environment.items():
        print(f"{variable}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
