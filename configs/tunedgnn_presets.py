"""Single source of shared TunedGNN architecture, optimizer and schedule presets.

The original profile follows the upstream per-dataset commands; paper is an
explicit alternate profile. All method launchers read this table.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

TUNEDGNN_REPOSITORY = "https://github.com/LUOyk1999/tunedGNN"
TUNEDGNN_REVISION = "23f9604e8b13a9a6d3faa2f691cd844006979153"


@dataclass(frozen=True)
class TunedGNNPreset:
    learning_rate: float
    hidden_channels: int
    num_layers: int
    weight_decay: float
    dropout: float
    epochs: int
    runs: int
    seed: int = 42
    pre_linear: bool = False
    residual_connections: bool = False
    layer_norm: bool = False
    batch_norm: bool = False
    jumping_knowledge: bool = False
    input_dropout: float = 0.15
    metric: str = "acc"
    model_profile: str = "medium"
    heads: int = 1
    train_mode: str = "full"
    partition_parts: int | None = None
    batch_size: int | None = None
    eval_batch_size: int | None = None
    neighbor_fanouts: tuple[int, ...] | None = None
    eval_neighbor_fanouts: tuple[int, ...] | None = None
    loader_workers: int = 0
    optimizer: str = "adam"
    lr_scheduler: str = "none"
    lr_scheduler_factor: float = 0.75
    lr_scheduler_patience: int = 50
    eval_every: int = 1
    eval_start_epoch: int = 1
    log_every: int = 10


def _medium(
    learning_rate: float,
    hidden_channels: int,
    num_layers: int,
    weight_decay: float,
    dropout: float,
    *,
    epochs: int = 500,
    runs: int = 5,
    seed: int = 42,
    pre_linear: bool = False,
    residual_connections: bool = False,
    layer_norm: bool = False,
    batch_norm: bool = False,
    metric: str = "acc",
) -> TunedGNNPreset:
    return TunedGNNPreset(
        learning_rate=learning_rate,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        weight_decay=weight_decay,
        dropout=dropout,
        epochs=epochs,
        runs=runs,
        seed=seed,
        pre_linear=pre_linear,
        residual_connections=residual_connections,
        layer_norm=layer_norm,
        batch_norm=batch_norm,
        # tunedGNN's medium_graph MPNN has no input-dropout stage.
        input_dropout=0.0,
        metric=metric,
    )


PAPER_TUNEDGNN_PRESETS: dict[tuple[str, str], TunedGNNPreset] = {
    ("amazon-computer", "gcn"): _medium(0.001, 512, 3, 5e-5, 0.5, epochs=1000, layer_norm=True),
    ("amazon-computer", "sage"): _medium(0.001, 64, 4, 5e-5, 0.3, epochs=1000, layer_norm=True),
    ("amazon-computer", "gat"): _medium(0.001, 64, 2, 5e-5, 0.5, epochs=1000, layer_norm=True),
    ("amazon-photo", "gcn"): _medium(
        0.001, 256, 6, 5e-5, 0.5, epochs=1000, layer_norm=True, residual_connections=True
    ),
    ("amazon-photo", "sage"): _medium(
        0.001, 64, 6, 5e-5, 0.2, epochs=1000, layer_norm=True, residual_connections=True
    ),
    ("amazon-photo", "gat"): _medium(
        0.001, 64, 3, 5e-5, 0.5, epochs=1000, layer_norm=True, residual_connections=True
    ),
    ("coauthor-cs", "gcn"): _medium(
        0.001, 512, 2, 5e-4, 0.3, epochs=1500, layer_norm=True, residual_connections=True
    ),
    ("coauthor-cs", "sage"): _medium(
        0.001, 512, 2, 5e-4, 0.5, epochs=1500, layer_norm=True, residual_connections=True
    ),
    ("coauthor-cs", "gat"): _medium(
        0.001, 256, 1, 5e-4, 0.3, epochs=1500, layer_norm=True, residual_connections=True
    ),
    ("coauthor-physics", "gcn"): _medium(
        0.001, 64, 2, 5e-4, 0.3, epochs=1500, layer_norm=True, residual_connections=True
    ),
    ("coauthor-physics", "sage"): _medium(
        0.001, 64, 2, 5e-4, 0.7, epochs=1500, batch_norm=True, residual_connections=True
    ),
    ("coauthor-physics", "gat"): _medium(
        0.001, 256, 2, 5e-4, 0.7, epochs=1500, batch_norm=True, residual_connections=True
    ),
    ("wikics", "gcn"): _medium(0.001, 256, 3, 0.0, 0.5, epochs=1000, layer_norm=True),
    ("wikics", "sage"): _medium(0.001, 256, 2, 0.0, 0.7, epochs=1000, layer_norm=True),
    ("wikics", "gat"): _medium(
        0.001, 512, 2, 0.0, 0.7, epochs=1000, layer_norm=True, residual_connections=True
    ),
    ("cora", "gcn"): _medium(
        0.001, 512, 3, 5e-4, 0.7, runs=5, seed=123
    ),
    ("cora", "sage"): _medium(
        0.001, 256, 3, 5e-4, 0.7, runs=5, seed=123
    ),
    ("cora", "gat"): _medium(
        0.001, 512, 3, 5e-4, 0.2, runs=5, seed=123, residual_connections=True
    ),
    ("citeseer", "gcn"): _medium(
        0.001, 512, 2, 0.01, 0.5, runs=5, seed=123
    ),
    ("citeseer", "sage"): _medium(
        0.001, 512, 3, 0.01, 0.2, runs=5, seed=123
    ),
    ("citeseer", "gat"): _medium(
        0.001, 256, 3, 0.01, 0.5, runs=5, seed=123, residual_connections=True
    ),
    ("pubmed", "gcn"): _medium(
        0.005, 256, 2, 5e-4, 0.7, runs=5, seed=123
    ),
    ("pubmed", "sage"): _medium(
        0.005, 512, 4, 5e-4, 0.7, runs=5, seed=123
    ),
    ("pubmed", "gat"): _medium(
        0.01, 512, 2, 5e-4, 0.5, runs=5, seed=123
    ),
    ("amazon-ratings", "gcn"): _medium(
        0.001, 512, 4, 0.0, 0.5, epochs=2500, batch_norm=True, residual_connections=True
    ),
    ("amazon-ratings", "sage"): _medium(
        0.001, 512, 9, 0.0, 0.5, epochs=2500, batch_norm=True, residual_connections=True
    ),
    ("amazon-ratings", "gat"): _medium(
        0.001, 512, 4, 0.0, 0.5, epochs=2500, batch_norm=True, residual_connections=True
    ),
    ("minesweeper", "gcn"): _medium(
        0.01,
        64,
        12,
        0.0,
        0.2,
        epochs=2000,
        batch_norm=True,
        residual_connections=True,
        metric="rocauc",
    ),
    ("minesweeper", "sage"): _medium(
        0.01,
        64,
        15,
        0.0,
        0.2,
        epochs=2000,
        batch_norm=True,
        residual_connections=True,
        metric="rocauc",
    ),
    ("minesweeper", "gat"): _medium(
        0.01,
        64,
        15,
        0.0,
        0.2,
        epochs=2000,
        batch_norm=True,
        residual_connections=True,
        metric="rocauc",
    ),
    ("roman-empire", "gcn"): _medium(
        0.001,
        512,
        9,
        0.0,
        0.5,
        epochs=2500,
        pre_linear=True,
        batch_norm=True,
        residual_connections=True,
    ),
    ("roman-empire", "sage"): _medium(
        0.001, 256, 9, 0.0, 0.3, epochs=2500, pre_linear=True, batch_norm=True
    ),
    ("roman-empire", "gat"): _medium(
        0.001,
        512,
        10,
        0.0,
        0.3,
        epochs=2500,
        pre_linear=True,
        batch_norm=True,
        residual_connections=True,
    ),
    ("questions", "gcn"): _medium(
        0.001,
        512,
        10,
        0.0,
        0.3,
        epochs=1500,
        pre_linear=True,
        residual_connections=True,
        metric="rocauc",
    ),
    ("questions", "sage"): _medium(
        0.001,
        512,
        6,
        0.0,
        0.2,
        epochs=1500,
        pre_linear=True,
        layer_norm=True,
        metric="rocauc",
    ),
    ("questions", "gat"): _medium(
        3e-5,
        512,
        3,
        0.0,
        0.2,
        epochs=1500,
        pre_linear=True,
        layer_norm=True,
        residual_connections=True,
        metric="rocauc",
    ),
    ("squirrel", "gcn"): _medium(
        0.01, 256, 4, 5e-4, 0.7, batch_norm=True, residual_connections=True
    ),
    ("squirrel", "sage"): _medium(
        0.01, 256, 3, 5e-4, 0.7, batch_norm=True, residual_connections=True
    ),
    ("squirrel", "gat"): _medium(
        0.005, 512, 7, 5e-4, 0.5, batch_norm=True, residual_connections=True
    ),
    ("chameleon", "gcn"): _medium(
        0.005, 512, 5, 0.001, 0.2, epochs=200
    ),
    ("chameleon", "sage"): _medium(
        0.01,
        256,
        4,
        0.001,
        0.7,
        epochs=200,
        batch_norm=True,
        residual_connections=True,
    ),
    ("chameleon", "gat"): _medium(
        0.01,
        256,
        2,
        0.001,
        0.7,
        epochs=200,
        batch_norm=True,
        residual_connections=True,
    ),
    ("ogbn-arxiv", "gcn"): TunedGNNPreset(
        0.0005,
        512,
        5,
        5e-4,
        0.5,
        2000,  # Prior local override: 200.
        5,
        residual_connections=True,
        batch_norm=True,
        model_profile="large",
    ),
    ("ogbn-arxiv", "sage"): TunedGNNPreset(
        0.0005,
        256,
        4,
        5e-4,
        0.5,
        2000,  # Prior local override: 200.
        5,
        residual_connections=True,
        batch_norm=True,
        model_profile="large",
    ),
    ("pokec", "gcn"): TunedGNNPreset(
        0.0005,
        256,
        7,
        0.0,
        0.2,
        2000,  # Prior local override: 200.
        5,
        input_dropout=0.0,
        residual_connections=True,
        batch_norm=True,
        model_profile="large",
        train_mode="partition",
        batch_size=550000,
        eval_every=9,
        eval_start_epoch=1001,
    ),
    ("pokec", "sage"): TunedGNNPreset(
        0.0005,
        256,
        7,
        0.0,
        0.2,
        2000,  # Prior local override: 200.
        5,
        input_dropout=0.0,
        residual_connections=True,
        batch_norm=True,
        model_profile="large",
        train_mode="partition",
        batch_size=550000,
        eval_every=9,
        eval_start_epoch=1001,
    ),
    ("pokec", "gat"): TunedGNNPreset(
        0.0005,
        256,
        7,
        0.0,
        0.2,
        2000,
        1,
        input_dropout=0.0,
        residual_connections=True,
        batch_norm=True,
        model_profile="large",
    ),
    ("ogbn-products", "gcn"): TunedGNNPreset(
        0.003,
        256,
        5,
        0.0,
        0.5,
        300,  # Prior local override: 200.
        5,
        layer_norm=True,
        model_profile="products",
        train_mode="partition",
        partition_parts=10,
    ),
    ("ogbn-products", "sage"): TunedGNNPreset(
        0.003,
        256,
        5,
        0.0,
        0.5,
        1000,  # Prior local override: 200.
        5,
        layer_norm=True,
        model_profile="products",
        train_mode="partition",
        partition_parts=10,
    ),
    # tunedGNN does not publish a Reddit command.  Benchmark's experiment
    # protocol deliberately reuses the corresponding OGBN-Products model
    # family and paper hyperparameters, as requested for the Reddit baselines
    # and their sparse counterparts.
    ("reddit", "gcn"): TunedGNNPreset(
        0.003,
        256,
        5,
        0.0,
        0.5,
        300,  # Prior local override: 100.
        5,
        layer_norm=True,
        model_profile="products",
        train_mode="partition",
        partition_parts=10,
    ),
    ("reddit", "sage"): TunedGNNPreset(
        0.003,
        256,
        5,
        0.0,
        0.5,
        1000,  # Prior local override: 100.
        5,
        layer_norm=True,
        model_profile="products",
        train_mode="partition",
        partition_parts=10,
    ),
    ("ogbn-proteins", "gcn"): TunedGNNPreset(
        0.01,
        512,
        3,
        0.0,
        0.3,
        100,
        5,
        seed=0,
        input_dropout=0.1,
        residual_connections=True,
        batch_norm=True,
        metric="rocauc",
        model_profile="proteins",
        heads=1,
        train_mode="neighbor",
        batch_size=8662,
        eval_batch_size=65536,
        neighbor_fanouts=(32, 32, 32),
        eval_neighbor_fanouts=(100, 100, 100),
        loader_workers=10,
        optimizer="adamw",
        lr_scheduler="plateau",
        eval_every=5,
        log_every=5,
    ),
    ("ogbn-proteins", "sage"): TunedGNNPreset(
        0.01,
        512,
        6,
        0.0,
        0.3,
        1000,
        5,
        seed=0,
        input_dropout=0.1,
        residual_connections=True,
        batch_norm=True,
        metric="rocauc",
        model_profile="proteins",
        heads=1,
        train_mode="neighbor",
        batch_size=8662,
        eval_batch_size=65536,
        neighbor_fanouts=(32, 32, 32, 32, 32, 32),
        eval_neighbor_fanouts=(100, 100, 100, 100, 100, 100),
        loader_workers=10,
        optimizer="adamw",
        lr_scheduler="plateau",
        eval_every=5,
        log_every=5,
    ),
}


# The source repository's run scripts do not use one uniform five-run policy.
# Build the default profile from the paper table, apply its implementation
# settings, and enforce the requested two-seed floor for large experiments.
_ORIGINAL_MEDIUM_RUNS = {
    "cora": 5,
    "citeseer": 5,
    "pubmed": 5,
    "squirrel": 10,
    "chameleon": 10,
}

ORIGINAL_TUNEDGNN_PRESETS: dict[tuple[str, str], TunedGNNPreset] = {
    key: replace(
        preset,
        runs=_ORIGINAL_MEDIUM_RUNS.get(key[0], 3),
        learning_rate=(
            3e-5
            if key[0] == "questions" and key[1] in {"gcn", "sage"}
            else preset.learning_rate
        ),
    )
    for key, preset in PAPER_TUNEDGNN_PRESETS.items()
    if preset.model_profile == "medium"
}

ORIGINAL_TUNEDGNN_PRESETS.update(
    {
        ("ogbn-arxiv", "gcn"): replace(
            PAPER_TUNEDGNN_PRESETS[("ogbn-arxiv", "gcn")],
            runs=2,
        ),
        ("ogbn-arxiv", "sage"): replace(
            PAPER_TUNEDGNN_PRESETS[("ogbn-arxiv", "sage")],
            runs=2,
        ),
        ("pokec", "gcn"): replace(
            PAPER_TUNEDGNN_PRESETS[("pokec", "gcn")],
            runs=2,
        ),
        ("pokec", "sage"): replace(
            PAPER_TUNEDGNN_PRESETS[("pokec", "sage")],
            runs=2,
        ),
        ("pokec", "gat"): PAPER_TUNEDGNN_PRESETS[("pokec", "gat")],
        ("ogbn-products", "gcn"): replace(
            PAPER_TUNEDGNN_PRESETS[("ogbn-products", "gcn")],
            hidden_channels=200,
            epochs=300,
            runs=2,
        ),
        ("ogbn-products", "sage"): replace(
            PAPER_TUNEDGNN_PRESETS[("ogbn-products", "sage")],
            hidden_channels=200,
            epochs=1001,
            runs=2,
        ),
        # tunedGNN-org has no Reddit command.  Keep the previously documented
        # Products fallback, using the Products implementation settings.
        ("reddit", "gcn"): replace(
            PAPER_TUNEDGNN_PRESETS[("reddit", "gcn")],
            hidden_channels=200,
            epochs=300,
            runs=2,
        ),
        ("reddit", "sage"): replace(
            PAPER_TUNEDGNN_PRESETS[("reddit", "sage")],
            hidden_channels=200,
            epochs=1001,
            runs=2,
        ),
        ("ogbn-proteins", "gcn"): replace(
            PAPER_TUNEDGNN_PRESETS[("ogbn-proteins", "gcn")],
            hidden_channels=80,
            heads=6,
            dropout=0.25,
            runs=2,
            residual_connections=False,
            batch_norm=False,
        ),
        ("ogbn-proteins", "sage"): replace(
            PAPER_TUNEDGNN_PRESETS[("ogbn-proteins", "sage")],
            hidden_channels=80,
            heads=4,
            dropout=0.25,
            runs=2,
            residual_connections=False,
            batch_norm=False,
        ),
    }
)

TUNEDGNN_PRESET_PROFILES = ("original", "paper")
TUNEDGNN_PRESETS_BY_PROFILE = {
    "original": ORIGINAL_TUNEDGNN_PRESETS,
    "paper": PAPER_TUNEDGNN_PRESETS,
}

# Backward compatibility for code that imports the old constant directly.
TUNEDGNN_PRESETS = ORIGINAL_TUNEDGNN_PRESETS


def get_tunedgnn_preset(
    dataset: str,
    model: str,
    profile: str = "original",
) -> TunedGNNPreset | None:
    """Return the configured tunedGNN-based preset, if present."""

    profile = str(profile).lower()
    if profile not in TUNEDGNN_PRESETS_BY_PROFILE:
        raise ValueError(
            f"Unknown tunedGNN preset profile '{profile}'. "
            f"Available: {', '.join(TUNEDGNN_PRESET_PROFILES)}"
        )
    return TUNEDGNN_PRESETS_BY_PROFILE[profile].get(
        (str(dataset).lower(), str(model).lower())
    )
