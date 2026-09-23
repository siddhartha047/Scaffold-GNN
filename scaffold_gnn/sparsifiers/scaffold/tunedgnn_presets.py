"""Original tunedGNN GCN/GraphSAGE contracts used by SCAFFOLD runs.

The versioned ``original`` tunedGNN table in :mod:`config.tunedgnn_presets` is
this repository's single source of truth.  SCAFFOLD translates that table to
this repository's argument names and adds only implementation-specific pipeline
details.  This keeps SCAFFOLD-GCN/GraphSAGE synchronized with
TunedGNN-GCN/GraphSAGE instead of maintaining a second hand-copied
hyperparameter matrix.

The table used to be loaded by path out of a sibling Benchmark checkout, which
made SCAFFOLD runs fail whenever that project moved the file between branches.
It is now vendored in-repo, so no Benchmark checkout is required.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import os
from pathlib import Path
import sys


SHARED_DATA_ROOT = os.environ.get("SCAFFOLD_DATA_ROOT", "./data")
DEFAULT_SCAFFOLD_SCRATCH_ROOT = os.environ.get("SCAFFOLD_CACHE_ROOT", "./results/cache")
SUPPORT_GRAPH_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TUNEDGNN_PRESETS_PATH = PROJECT_ROOT.parent / "configs" / "tunedgnn_presets.py"


def _load_reference_tunedgnn_presets():
    """Load the in-repo tunedGNN table from its exact path.

    Loading by path rather than by module name keeps this immune to a differently
    named ``config`` package appearing earlier on ``sys.path``/``PYTHONPATH``,
    which the wrappers set to several roots.
    """

    if not TUNEDGNN_PRESETS_PATH.is_file():
        raise RuntimeError(
            f"Missing the in-repo tunedGNN contract: {TUNEDGNN_PRESETS_PATH}"
        )
    module_name = "_scaffold_reference_tunedgnn_presets"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, TUNEDGNN_PRESETS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load the tunedGNN contract: {TUNEDGNN_PRESETS_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_REFERENCE_TUNEDGNN = _load_reference_tunedgnn_presets()
TUNEDGNN_REVISION = _REFERENCE_TUNEDGNN.TUNEDGNN_REVISION


@dataclass(frozen=True)
class TunedGNNPreset:
    lr: float
    hidden_channels: int
    local_layers: int
    weight_decay: float
    dropout: float
    epochs: int = 500
    runs: int = 3
    seed: int = 42
    pre_linear: bool = False
    res: bool = False
    ln: bool = False
    bn: bool = False
    jk: bool = False
    pre_ln: bool = False
    in_dropout: float = 0.0
    metric: str = "acc"
    model_profile: str = "medium"
    num_heads: int = 1
    train_mode: str = "full"
    large_graph_pipeline: str = "full"
    large_batch_size: int | str = "auto"
    large_train_parts: int | str = "auto"
    large_eval_parts: int | str = "auto"
    large_loader_workers: int = 0
    large_eval_cpu: bool = False
    neighbor_batch_size: int = 1024
    neighbor_eval_batch_size: int = 2048
    neighbor_num_neighbors: str = ""
    neighbor_eval_num_neighbors: str = ""
    neighbor_workers: int = 0
    optimizer: str = "adam"
    lr_scheduler: str = "none"
    lr_scheduler_factor: float = 0.75
    lr_scheduler_patience: int = 50
    display_step: int = 100
    eval_step: int = 1
    eval_start_epoch: int = 1
    log_every: int = 10


def _csv(values) -> str:
    if values is None:
        return ""
    return ",".join(str(value) for value in values)


def _translate_reference_preset(dataset: str, preset) -> TunedGNNPreset:
    """Translate one Benchmark original-profile preset to SCAFFOLD arguments."""

    profile = str(preset.model_profile)
    if profile == "products":
        large_graph_pipeline = "random_node_loader"
    elif profile == "proteins":
        large_graph_pipeline = "neighbor"
    elif dataset == "pokec":
        large_graph_pipeline = "node_batch"
    else:
        large_graph_pipeline = "full"

    # Benchmark records 1001 as the zero-based Pokec warm-up boundary.  This
    # parser exposes a one-based first eligible epoch, hence the +1.  The first
    # actual evaluation remains upstream's epoch index 1008 because eval_step=9.
    eval_start_epoch = int(preset.eval_start_epoch)
    if dataset == "pokec":
        eval_start_epoch += 1

    if profile == "proteins":
        display_step = int(preset.log_every)
    elif dataset == "pokec":
        display_step = int(preset.eval_every)
    elif profile in {"large", "products"}:
        display_step = 1
    else:
        display_step = 100

    neighbor_training = str(preset.train_mode) == "neighbor"
    return TunedGNNPreset(
        lr=float(preset.learning_rate),
        hidden_channels=int(preset.hidden_channels),
        local_layers=int(preset.num_layers),
        weight_decay=float(preset.weight_decay),
        dropout=float(preset.dropout),
        epochs=int(preset.epochs),
        runs=int(preset.runs),
        seed=int(preset.seed),
        pre_linear=bool(preset.pre_linear),
        res=bool(preset.residual_connections),
        ln=bool(preset.layer_norm),
        bn=bool(preset.batch_norm),
        jk=bool(preset.jumping_knowledge),
        in_dropout=(
            0.0 if profile == "products" else float(preset.input_dropout)
        ),
        metric=str(preset.metric),
        model_profile=profile,
        num_heads=int(preset.heads),
        train_mode="neighbor" if neighbor_training else "full",
        large_graph_pipeline=large_graph_pipeline,
        large_batch_size=(
            int(preset.batch_size) if preset.batch_size is not None else "auto"
        ),
        large_train_parts=(
            int(preset.partition_parts)
            if preset.partition_parts is not None
            else "auto"
        ),
        large_eval_parts=1 if profile == "products" else "auto",
        large_loader_workers=int(preset.loader_workers),
        large_eval_cpu=dataset == "pokec",
        neighbor_batch_size=(
            int(preset.batch_size)
            if neighbor_training and preset.batch_size is not None
            else 1024
        ),
        neighbor_eval_batch_size=(
            int(preset.eval_batch_size)
            if neighbor_training and preset.eval_batch_size is not None
            else 2048
        ),
        neighbor_num_neighbors=(
            _csv(preset.neighbor_fanouts) if neighbor_training else ""
        ),
        neighbor_eval_num_neighbors=(
            _csv(preset.eval_neighbor_fanouts) if neighbor_training else ""
        ),
        neighbor_workers=int(preset.loader_workers) if neighbor_training else 0,
        optimizer=str(preset.optimizer),
        lr_scheduler=str(preset.lr_scheduler),
        lr_scheduler_factor=float(preset.lr_scheduler_factor),
        lr_scheduler_patience=int(preset.lr_scheduler_patience),
        display_step=display_step,
        eval_step=int(preset.eval_every),
        eval_start_epoch=eval_start_epoch,
        log_every=int(preset.log_every),
    )


# Reddit has no published tunedGNN command; the original profile and SCAFFOLD
# both intentionally use the OGBN-Products implementation contract.
REFERENCE_PRESETS = {
    (dataset, gnn): preset
    for (dataset, gnn), preset in _REFERENCE_TUNEDGNN.ORIGINAL_TUNEDGNN_PRESETS.items()
    if gnn in {"gcn", "sage"}
}
PRESETS: dict[tuple[str, str], TunedGNNPreset] = {
    (dataset, gnn): _translate_reference_preset(dataset, preset)
    for (dataset, gnn), preset in REFERENCE_PRESETS.items()
}


ALIASES = {
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


def canonical_dataset(name: str) -> str:
    key = "-".join(str(name).strip().lower().replace("_", "-").split())
    return ALIASES.get(key, key)


# Backbones with no tuned preset of their own. TunedGNN swept GCN and
# GraphSAGE; GAT and GIN are used to show that SCAFFOLD is not tied to one
# aggregator, so they reuse that dataset's GCN training recipe rather than
# inventing an untuned one. Any per-backbone tuning would confound the
# comparison with the sparsifier being tested.
PRESET_FALLBACK_GNN = {"gat": "gcn", "gin": "gcn"}


def get_preset(dataset: str, gnn: str) -> TunedGNNPreset | None:
    key = canonical_dataset(dataset)
    gnn = str(gnn).lower()
    preset = PRESETS.get((key, gnn))
    if preset is None and gnn in PRESET_FALLBACK_GNN:
        preset = PRESETS.get((key, PRESET_FALLBACK_GNN[gnn]))
    return preset


def argparse_defaults(dataset: str, gnn: str) -> dict[str, object]:
    """Translate a preset to parser defaults for Scaffold/full only."""

    preset = get_preset(dataset, gnn)
    if preset is None:
        return {}
    defaults = asdict(preset)
    defaults.update(
        {
            "gnn": str(gnn).lower(),
            "data_dir": os.environ.get("SCAFFOLD_DATA_ROOT", SHARED_DATA_ROOT),
            "neighbor_tmp_dir": os.path.join(
                os.environ.get("SCAFFOLD_SCRATCH_ROOT", DEFAULT_SCAFFOLD_SCRATCH_ROOT),
                "tmp",
                canonical_dataset(dataset),
                str(gnn).lower(),
            ),
            "tunedgnn_strict": True,
            "rand_split": False,
            "rand_split_class": canonical_dataset(dataset) in {"cora", "citeseer", "pubmed"},
            "label_num_per_class": 20,
            "valid_num": 500,
            "test_num": 1000,
        }
    )
    return defaults


def validate_complete_matrix() -> None:
    datasets = {
        "cora", "citeseer", "pubmed", "amazon-computer", "amazon-photo",
        "coauthor-cs", "coauthor-physics", "wikics", "squirrel", "chameleon",
        "roman-empire", "amazon-ratings", "minesweeper", "questions", "reddit",
        "ogbn-products", "ogbn-arxiv", "ogbn-proteins", "pokec",
    }
    missing = sorted((dataset, gnn) for dataset in datasets for gnn in ("gcn", "sage") if (dataset, gnn) not in PRESETS)
    if missing:
        raise RuntimeError(f"Incomplete tunedGNN Scaffold preset matrix: {missing}")


validate_complete_matrix()
