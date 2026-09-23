#!/usr/bin/env python3
"""Sparse mini-batch variants of legacy pruning baselines for large graphs.

The original MoG, AdaGLT, and Unified-LTH node-classification programs keep
full-graph edge state on the accelerator (and the latter two construct dense
N-by-N tensors).  This entry point preserves Benchmark's datasets and splits,
but applies each method's edge-selection idea to sparse NeighborLoader
mini-batches.  It is intentionally selected by the method wrappers only for
the configured large datasets; small graphs continue to use the original
authors' programs.
"""

from __future__ import annotations

import argparse
import importlib
import hashlib
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, RandomNodeLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import (
    add_remaining_self_loops,
    coalesce,
    is_undirected,
    scatter,
)

SUPPORT_GRAPH_ROOT = Path(__file__).resolve().parents[2]
if str(SUPPORT_GRAPH_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPPORT_GRAPH_ROOT))

from scaffold_gnn.data.connector import load_pyg_data, select_pyg_split
from scripts.common.baseline_result_utils import (
    RunTimeBudget,
    append_baseline_result,
    single_label_roc_auc_percent,
)


METHOD_LABELS = {
    "mog": "MoG",
    "adaglt": "AdaGLT",
    "unified-lth": "Unified-LTH",
}


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(value: str) -> torch.device:
    requested = str(value).strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {value!r} was requested but CUDA is unavailable; "
            "refusing a silent CPU fallback"
        )
    if requested.startswith("cuda"):
        return torch.device(requested)
    return torch.device(f"cuda:{int(requested)}")


def parse_fanouts(value: str, layers: int) -> list[int]:
    fanouts = [int(token.strip()) for token in str(value).split(",") if token.strip()]
    # Preserve the common ``15,10`` shorthand for deep tunedGNN backbones by
    # treating its second entry as the fanout for every remaining layer.
    if len(fanouts) == 2 and int(layers) > 2:
        fanouts = [fanouts[0], *([fanouts[1]] * (int(layers) - 1))]
    if len(fanouts) != int(layers):
        raise ValueError(
            f"--fanouts needs one value per layer ({layers}), got {fanouts}"
        )
    if any(value == 0 or value < -1 for value in fanouts):
        raise ValueError("fanouts must be positive integers or -1")
    return fanouts


def exact_straight_through_mask(
    scores: torch.Tensor,
    kept_ratio: float,
    *,
    temperature: float,
) -> torch.Tensor:
    """Return an exact top-k hard mask with a sigmoid straight-through gradient."""

    num_edges = int(scores.numel())
    if num_edges == 0:
        return scores
    keep = max(1, min(num_edges, int(round(num_edges * float(kept_ratio)))))
    finite_scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)
    soft = torch.sigmoid(finite_scores / max(float(temperature), 1e-6))
    hard = torch.zeros_like(soft)
    if keep == num_edges:
        hard.fill_(1.0)
    else:
        hard.scatter_(0, torch.topk(finite_scores, keep, sorted=False).indices, 1.0)
    return hard + soft - soft.detach()


def stable_masked_gcn_norm(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    *,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GCN normalization with stable gradients for exact hard edge masks.

    PyG's differentiable ``gcn_norm`` computes ``degree.pow(-0.5)`` directly
    from ``edge_weight``. An exact mask can give a sampled node zero weighted
    degree. Although the forward path replaces the resulting infinity with
    zero, ``PowBackward0`` still sees the singular derivative and emits NaN.

    Self-loops remain structural GCN edges, and the degree normalization is
    computed from the detached hard topology. Gradients continue to flow
    through the selected message weights, which is the quantity learned by
    MoG and AdaGLT.
    """

    normalized_edges, normalized_weights = add_remaining_self_loops(
        edge_index,
        edge_weight,
        fill_value=1.0,
        num_nodes=int(num_nodes),
    )
    source, target = normalized_edges
    # Existing input self-loops may have received a zero hard mask. Treat all
    # self-loops like the structural loops GCNConv would add itself.
    normalized_weights = torch.where(
        source == target,
        torch.ones_like(normalized_weights),
        normalized_weights,
    )
    degree = scatter(
        normalized_weights.detach(),
        target,
        dim=0,
        dim_size=int(num_nodes),
        reduce="sum",
    )
    inverse_sqrt_degree = degree.clamp_min(1.0).pow(-0.5)
    normalized_weights = (
        inverse_sqrt_degree[source]
        * normalized_weights
        * inverse_sqrt_degree[target]
    )
    return normalized_edges, normalized_weights


def deterministic_ticket_indices(
    num_edges: int,
    kept_ratio: float,
    seed: int,
) -> torch.Tensor:
    """Choose exactly k spread-out edges without allocating an E-element ranking."""

    if num_edges < 1:
        return torch.empty(0, dtype=torch.long)
    keep = max(1, min(num_edges, int(round(num_edges * float(kept_ratio)))))
    if keep == num_edges:
        return torch.arange(num_edges, dtype=torch.long)
    # floor(i * E / k) is unique for k <= E. A seeded cyclic translation gives
    # independent tickets while retaining exact cardinality and O(k) memory.
    indices = torch.div(
        torch.arange(keep, dtype=torch.int64) * int(num_edges),
        keep,
        rounding_mode="floor",
    )
    offset = (int(seed) * 2_654_435_761) % int(num_edges)
    return (indices + offset) % int(num_edges)


class MoGEdgeScorer(nn.Module):
    """Memory-bounded mixture-of-graph-experts scoring on sampled edges."""

    def __init__(self, channels: int, experts: int = 3, projection: int = 32):
        super().__init__()
        projection = max(8, min(int(projection), int(channels)))
        self.gate = nn.Linear(channels, experts)
        self.source = nn.ModuleList(
            nn.Linear(channels, projection, bias=False) for _ in range(experts)
        )
        self.target = nn.ModuleList(
            nn.Linear(channels, projection, bias=False) for _ in range(experts)
        )
        self.scale = math.sqrt(float(projection))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source_nodes, target_nodes = edge_index
        scores = torch.zeros(source_nodes.numel(), device=x.device, dtype=x.dtype)
        # Bound scorer temporaries independently of sampled-subgraph size.
        # GCN propagation still receives the complete sampled subgraph, but
        # edge scoring no longer materializes several E-by-projection tensors.
        for start in range(0, int(source_nodes.numel()), 262_144):
            end = min(start + 262_144, int(source_nodes.numel()))
            source = source_nodes[start:end]
            target = target_nodes[start:end]
            gates = F.softmax(self.gate(x[source]), dim=-1)
            chunk_scores = scores[start:end]
            for expert, (source_layer, target_layer) in enumerate(
                zip(self.source, self.target)
            ):
                similarity = (
                    source_layer(x[source]) * target_layer(x[target])
                ).sum(dim=-1) / self.scale
                chunk_scores.add_(gates[:, expert] * similarity)
        return scores


_NATIVE_MOG_CLASSES: dict[str, type[nn.Module]] = {}


def load_native_mog_class(source_profile: str):
    """Load MoG's dataset-specific MoE class from the upstream source tree."""

    profile = str(source_profile).strip().lower()
    source_directory = (
        "ogbn_proteins" if profile == "ogbn-proteins" else "ogbn_arxiv"
    )
    if source_directory in _NATIVE_MOG_CLASSES:
        return _NATIVE_MOG_CLASSES[source_directory]
    support_root = Path(__file__).resolve().parents[2]
    native_root = support_root / "RelatedMethods" / "MoG-main" / source_directory
    if not native_root.is_dir():
        raise FileNotFoundError(f"native MoG source is missing: {native_root}")

    # Upstream MoE.py imports SpLearner by its bare module name. Insert only
    # while importing and discard generic module names afterward so this
    # adapter cannot leak one dataset's implementation into another.
    native_path = str(native_root)
    sys.path.insert(0, native_path)
    try:
        for name in ("MoE", "SpLearner"):
            sys.modules.pop(name, None)
        module = importlib.import_module("MoE")
        native_class = module.MoE
    finally:
        sys.path.remove(native_path)
        for name in ("MoE", "SpLearner"):
            sys.modules.pop(name, None)
    _NATIVE_MOG_CLASSES[source_directory] = native_class
    return native_class


class NativeMoGEdgeScorer(nn.Module):
    """Adapter around MoG's native Arxiv/Proteins mixture-of-experts learner."""

    def __init__(
        self,
        channels: int,
        kept_ratio: float,
        source_profile: str,
        *,
        experts: int = 3,
        hidden_channels: int = 32,
    ):
        super().__init__()
        self.source_profile = str(source_profile).strip().lower()
        native_class = load_native_mog_class(self.source_profile)
        k_list = torch.full((experts,), float(kept_ratio), dtype=torch.float32)
        self.learner = native_class(
            input_size=int(channels),
            hidden_size=int(hidden_channels),
            num_experts=int(experts),
            nlayers=2,
            activation=nn.ReLU(),
            k_list=k_list,
            expert_select=int(experts),
            lam=0.1,
        )
        self.auxiliary_loss = torch.tensor(0.0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.learner.k_list = self.learner.k_list.to(x.device)
        for expert in self.learner.experts:
            if torch.is_tensor(expert.k):
                expert.k = expert.k.to(x.device)
        if self.source_profile == "ogbn-proteins":
            if edge_attr is None:
                edge_attr = torch.zeros(
                    (edge_index.size(1), 8), device=x.device, dtype=x.dtype
                )
            elif edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            if edge_attr.size(-1) != 8:
                raise ValueError(
                    "native ogbn-proteins MoG expects eight edge features; "
                    f"observed {tuple(edge_attr.shape)}"
                )
            mask, loss = self.learner(
                x, edge_index, 0.5, edge_attr.float(), self.training
            )
        else:
            values = torch.ones(
                edge_index.size(1), device=x.device, dtype=x.dtype
            )
            mask, loss = self.learner(
                x,
                edge_index,
                0.5,
                (int(x.size(0)), int(x.size(0))),
                edge_attr=values,
                training=self.training,
            )
        self.auxiliary_loss = loss
        return mask


class AdaGLTEdgeScorer(nn.Module):
    """Sparse analogue of AdaGLT's learned edge score and node threshold."""

    def __init__(self, channels: int, projection: int = 64):
        super().__init__()
        projection = max(8, min(int(projection), int(channels)))
        self.source = nn.Linear(channels, projection, bias=False)
        self.target = nn.Linear(channels, projection, bias=False)
        self.threshold = nn.Linear(channels, 1)
        self.scale = math.sqrt(float(projection))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source_nodes, target_nodes = edge_index
        scores = torch.empty(source_nodes.numel(), device=x.device, dtype=x.dtype)
        for start in range(0, int(source_nodes.numel()), 262_144):
            end = min(start + 262_144, int(source_nodes.numel()))
            source = source_nodes[start:end]
            target = target_nodes[start:end]
            similarity = (
                self.source(x[source]) * self.target(x[target])
            ).sum(dim=-1) / self.scale
            node_threshold = F.softplus(self.threshold(x[source]).squeeze(-1))
            scores[start:end] = similarity - node_threshold
        return scores


class SampledSparseGCN(nn.Module):
    """Neighbor-sampled GCN with method-specific graph sparsification."""

    def __init__(
        self,
        method: str,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kept_ratio: float,
        dropout: float,
        input_dropout: float,
        temperature: float,
        layers: int,
        residual: bool,
        layer_norm: bool,
        batch_norm: bool,
        pre_linear: bool = False,
        jumping_knowledge: bool = False,
        mog_source_profile: str = "scaled-legacy",
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        self.method = method
        self.learned_edge_mask = method in {"mog", "adaglt"}
        # Every scalable related-method comparison is evaluated with the same
        # tunedGNN GCN backbone.  MoG used to be accidentally omitted here,
        # despite the wrapper reporting ``tunedgnn_backbone=true``.
        self.tuned_backbone = method in {"mog", "adaglt", "unified-lth"}
        self.kept_ratio = float(kept_ratio)
        self.dropout = float(dropout)
        self.input_dropout = float(input_dropout)
        self.temperature = float(temperature)
        self.residual = bool(residual)
        self.layer_norm = bool(layer_norm)
        self.batch_norm = bool(batch_norm)
        self.pre_linear = bool(pre_linear) if self.tuned_backbone else False
        self.mog_source_profile = str(mog_source_profile)
        self.native_mog = (
            method == "mog" and self.mog_source_profile != "scaled-legacy"
        )
        self.auxiliary_loss = torch.tensor(0.0)
        self.jumping_knowledge = (
            bool(jumping_knowledge) if self.tuned_backbone else False
        )
        if self.tuned_backbone:
            if self.pre_linear:
                dimensions = [hidden_channels] * (layers + 1)
                self.lin_in = nn.Linear(in_channels, hidden_channels)
            else:
                dimensions = [in_channels] + [hidden_channels] * layers
                self.lin_in = None
            self.pred_local = nn.Linear(hidden_channels, out_channels)
        else:
            dimensions = [in_channels]
            if layers > 1:
                dimensions.extend([hidden_channels] * (layers - 1))
            dimensions.append(out_channels)
            self.lin_in = None
            self.pred_local = None
        self.convolutions = nn.ModuleList(
            GCNConv(
                dimensions[index],
                dimensions[index + 1],
                cached=False,
                normalize=not self.learned_edge_mask,
                add_self_loops=not self.learned_edge_mask,
            )
            for index in range(layers)
        )
        self.residual_lins = nn.ModuleList(
            nn.Linear(
                dimensions[index],
                dimensions[index + 1],
                bias=self.tuned_backbone,
            )
            for index in range(
                layers if self.tuned_backbone else layers - 1
            )
        )
        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(dimensions[index + 1])
            for index in range(
                layers if self.tuned_backbone else layers - 1
            )
        )
        self.batch_norms = nn.ModuleList(
            nn.BatchNorm1d(dimensions[index + 1])
            for index in range(
                layers if self.tuned_backbone else layers - 1
            )
        )
        if self.native_mog:
            # Upstream MoG owns one graph learner and reuses its single learned
            # mask in every GNN layer.  Creating one learner per GCN layer was
            # both algorithmically wrong and multiplied Pokec's topology-
            # learning cost by seven.
            self.scorers = nn.ModuleList(
                [NativeMoGEdgeScorer(
                    dimensions[0],
                    self.kept_ratio,
                    self.mog_source_profile,
                )]
            )
        elif method in {"mog", "adaglt"}:
            scorer_type = MoGEdgeScorer if method == "mog" else AdaGLTEdgeScorer
            self.scorers = nn.ModuleList(
                scorer_type(dimensions[index]) for index in range(layers)
            )
        else:
            self.scorers = nn.ModuleList()
        self.mask_kept = 0
        self.mask_total = 0

    def reset_mask_statistics(self) -> None:
        self.mask_kept = 0
        self.mask_total = 0

    @property
    def achieved_kept_ratio(self) -> float:
        return self.mask_kept / self.mask_total if self.mask_total else 1.0

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        *,
        full_topology: bool = False,
    ) -> torch.Tensor:
        self.auxiliary_loss = x.new_zeros(())
        if self.input_dropout > 0:
            x = F.dropout(
                x, p=self.input_dropout, training=self.training
            )
        if self.lin_in is not None:
            x = self.lin_in(x)
            x = F.dropout(
                x,
                p=self.dropout,
                training=self.training,
            )
        shared_convolution_edges = edge_index
        shared_edge_weight = None
        if full_topology and self.learned_edge_mask:
            shared_convolution_edges, shared_edge_weight = stable_masked_gcn_norm(
                edge_index,
                torch.ones(edge_index.size(1), device=edge_index.device),
                num_nodes=int(x.size(0)),
            )
        elif self.native_mog:
            shared_edge_weight = self.scorers[0](x, edge_index, edge_attr)
            self.auxiliary_loss = self.scorers[0].auxiliary_loss
            self.mask_kept += int((shared_edge_weight.detach() > 0.5).sum())
            self.mask_total += int(shared_edge_weight.numel())
            shared_convolution_edges, shared_edge_weight = stable_masked_gcn_norm(
                edge_index,
                shared_edge_weight,
                num_nodes=int(x.size(0)),
            )
        x_final = 0
        for layer, convolution in enumerate(self.convolutions):
            previous = x
            edge_weight = shared_edge_weight
            convolution_edges = shared_convolution_edges
            if (
                self.method in {"mog", "adaglt"}
                and not self.native_mog
                and not full_topology
            ):
                scores = self.scorers[layer](x, edge_index, edge_attr) \
                    if isinstance(self.scorers[layer], NativeMoGEdgeScorer) \
                    else self.scorers[layer](x, edge_index)
                if isinstance(self.scorers[layer], NativeMoGEdgeScorer):
                    edge_weight = scores
                    self.auxiliary_loss = (
                        self.auxiliary_loss + self.scorers[layer].auxiliary_loss
                    )
                else:
                    edge_weight = exact_straight_through_mask(
                        scores,
                        self.kept_ratio,
                        temperature=self.temperature,
                    )
                self.mask_kept += int((edge_weight.detach() > 0.5).sum())
                self.mask_total += int(edge_weight.numel())
                convolution_edges = edge_index
                convolution_edges, edge_weight = stable_masked_gcn_norm(
                    edge_index,
                    edge_weight,
                    num_nodes=int(x.size(0)),
                )
            x = convolution(
                x,
                convolution_edges,
                edge_weight=edge_weight,
            )
            if self.tuned_backbone:
                if self.residual:
                    x = x + self.residual_lins[layer](previous)
                if self.layer_norm:
                    x = self.layer_norms[layer](x)
                elif self.batch_norm:
                    x = self.batch_norms[layer](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x_final = (
                    x_final + x if self.jumping_knowledge else x
                )
            elif layer + 1 < len(self.convolutions):
                if self.residual:
                    x = x + self.residual_lins[layer](previous)
                if self.layer_norm:
                    x = self.layer_norms[layer](x)
                elif self.batch_norm:
                    x = self.batch_norms[layer](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.pred_local(x_final) if self.tuned_backbone else x


def ticket_cache_path(
    cache_root: Path,
    dataset: str,
    edge_index: torch.Tensor,
    kept_ratio: float,
    seed: int,
) -> Path:
    identity = (
        "edge-only-paired-v2|"
        f"{dataset}|edges={edge_index.size(1)}|ratio={kept_ratio:.12g}|seed={seed}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return (
        cache_root
        / "scalable_unified_lth"
        / dataset.lower().replace("_", "-")
        / digest
        / "edge_index.pt"
    )


def load_or_create_ticket(
    edge_index: torch.Tensor,
    *,
    cache_root: Path,
    dataset: str,
    kept_ratio: float,
    seed: int,
) -> torch.Tensor:
    destination = ticket_cache_path(
        cache_root, dataset, edge_index, kept_ratio, seed
    )
    expected = max(
        1,
        min(
            int(edge_index.size(1)),
            int(round(edge_index.size(1) * float(kept_ratio))),
        ),
    )
    if destination.exists():
        cached = torch.load(destination, map_location="cpu", weights_only=True)
        cached_edges = cached["edge_index"]
        if (
            int(cached_edges.size(1))
            in {expected - 1, expected, expected + 1}
            and int(cached.get("original_edges", -1)) == int(edge_index.size(1))
            and int(cached.get("ticket_version", -1)) == 2
        ):
            print(
                "Unified-LTH sparse ticket is already prepared; "
                f"loading from {destination}",
                flush=True,
            )
            return cached_edges.contiguous()

    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if is_undirected(edge_index):
        num_nodes = int(edge_index.max().item()) + 1
        source, target = edge_index
        low = torch.minimum(source, target)
        high = torch.maximum(source, target)
        pair_ids = low * num_nodes + high
        unique_pairs = torch.unique(pair_ids, sorted=True)
        pair_count = max(
            1,
            min(int(unique_pairs.numel()), expected // 2),
        )
        pair_indices = deterministic_ticket_indices(
            int(unique_pairs.numel()),
            pair_count / max(1, int(unique_pairs.numel())),
            seed,
        )
        selected_pairs = unique_pairs[pair_indices]
        selected_low = torch.div(
            selected_pairs,
            num_nodes,
            rounding_mode="floor",
        )
        selected_high = selected_pairs.remainder(num_nodes)
        non_loop = selected_low != selected_high
        sparse_edges = torch.cat(
            [
                torch.stack([selected_low, selected_high]),
                torch.stack(
                    [
                        selected_high[non_loop],
                        selected_low[non_loop],
                    ]
                ),
            ],
            dim=1,
        ).contiguous()
    else:
        indices = deterministic_ticket_indices(
            int(edge_index.size(1)), kept_ratio, seed
        )
        sparse_edges = edge_index[:, indices].contiguous()
    payload = {
        "edge_index": sparse_edges,
        "original_edges": int(edge_index.size(1)),
        "kept_ratio": float(kept_ratio),
        "seed": int(seed),
        "ticket_version": 2,
    }
    with tempfile.NamedTemporaryFile(
        prefix=".edge_index.",
        suffix=".pt",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        "Prepared Unified-LTH sparse ticket "
        f"at {destination} in {time.perf_counter() - started:.2f}s",
        flush=True,
    )
    return sparse_edges


def make_loader(
    data: Data,
    input_nodes: torch.Tensor,
    *,
    fanouts: list[int],
    batch_size: int,
    workers: int,
    shuffle: bool,
) -> NeighborLoader:
    return NeighborLoader(
        data,
        input_nodes=input_nodes,
        num_neighbors=fanouts,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        # Reusing workers avoids paying process start-up and CSC-sharing costs
        # once per epoch on the large datasets.
        persistent_workers=int(workers) > 0,
    )


def make_partition_loader(
    data: Data,
    *,
    parts: int,
    workers: int,
    shuffle: bool,
) -> RandomNodeLoader:
    """Return the induced random-node partitions used by native large MoG.

    MoG's upstream OGBN-Proteins implementation trains on random induced
    subgraphs, and tunedGNN uses the same strategy for Products and Pokec.  A
    full pass through thousands of NeighborLoader seed batches is not the
    equivalent runtime contract for these models.
    """

    return RandomNodeLoader(
        data,
        num_parts=int(parts),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        persistent_workers=int(workers) > 0,
    )


class FixedRandomNodePartitions:
    """Materialize one seeded validation partitioning and reuse it.

    This matches native MoG-Proteins, which prepares validation subgraphs once
    instead of repartitioning or filtering the full edge list at every
    evaluation epoch.
    """

    def __init__(self, data: Data, *, parts: int, seed: int):
        self.data = data
        self.parts = int(parts)
        self.seed = int(seed)
        if self.parts == 1:
            self.batches = [data]
            return
        self.batches = None

    def _materialize(self) -> None:
        if self.batches is not None:
            return
        started = time.perf_counter()
        generator = torch.Generator().manual_seed(self.seed)
        permutation = torch.randperm(
            int(self.data.num_nodes), generator=generator
        )
        self.batches = [
            self.data.subgraph(index.contiguous())
            for index in torch.tensor_split(permutation, self.parts)
            if int(index.numel()) > 0
        ]
        print(
            f"[MoGPartitions] materialized_fixed_eval_parts="
            f"{len(self.batches)} seconds={time.perf_counter() - started:.2f}",
            flush=True,
        )

    def __iter__(self):
        self._materialize()
        assert self.batches is not None
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches) if self.batches is not None else self.parts


def masked_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    multilabel: bool,
) -> torch.Tensor:
    if not multilabel:
        labels = labels.reshape(-1)
        valid = labels >= 0
        return F.cross_entropy(logits[valid], labels[valid].long())
    valid = torch.isfinite(labels) & (labels >= 0)
    losses = F.binary_cross_entropy_with_logits(
        logits,
        torch.nan_to_num(labels, nan=0.0).float(),
        reduction="none",
    )
    return losses[valid].mean()


def nonfinite_parameter_names(
    model: nn.Module,
    *,
    gradients: bool = False,
) -> list[str]:
    """Return a bounded list of parameters containing NaN/Inf values."""

    invalid: list[str] = []
    for name, parameter in model.named_parameters():
        value = parameter.grad if gradients else parameter
        if value is not None and not bool(torch.isfinite(value).all()):
            invalid.append(name)
            if len(invalid) >= 8:
                break
    return invalid


def single_label_metrics(
    labels: torch.Tensor,
    logits: torch.Tensor,
    indices: torch.Tensor,
    *,
    metric: str = "acc",
) -> tuple[float, float]:
    truth = labels[indices].reshape(-1).numpy()
    scores = logits[indices].numpy()
    prediction = scores.argmax(axis=-1)
    valid = (truth >= 0) & np.isfinite(scores).all(axis=1)
    if not valid.any():
        return float("nan"), float("nan")
    macro_f1 = float(
        f1_score(
            truth[valid],
            prediction[valid],
            average="macro",
            zero_division=0,
        )
    )
    if str(metric).lower() == "rocauc":
        # Minesweeper and Questions are scored by ROC-AUC. Their accuracy is
        # pinned at the 80%/97% majority-class rate whatever the model learns,
        # which would also make the best-epoch selection in run_once degenerate.
        primary = single_label_roc_auc_percent(truth[valid], scores[valid]) / 100.0
    else:
        primary = float((truth[valid] == prediction[valid]).mean())
    return primary, macro_f1


def multilabel_metrics(
    labels: torch.Tensor,
    logits: torch.Tensor,
    indices: torch.Tensor,
    *,
    metric: str = "rocauc",
) -> tuple[float, float]:
    # Multi-label targets (OGBN-Proteins) are always per-task ROC-AUC; the
    # keyword only exists so metric_triplet can call either implementation.
    del metric
    truth = labels[indices].numpy()
    scores = logits[indices].numpy()
    aucs: list[float] = []
    binary_truth: list[np.ndarray] = []
    binary_prediction: list[np.ndarray] = []
    for task in range(truth.shape[1]):
        valid = (
            np.isfinite(truth[:, task])
            & (truth[:, task] >= 0)
            & np.isfinite(scores[:, task])
        )
        task_truth = truth[valid, task]
        if task_truth.size == 0 or np.unique(task_truth).size < 2:
            continue
        task_scores = scores[valid, task]
        aucs.append(float(roc_auc_score(task_truth, task_scores)))
        binary_truth.append(task_truth.astype(np.int64))
        binary_prediction.append((task_scores >= 0.0).astype(np.int64))
    if not aucs:
        return float("nan"), float("nan")
    return (
        float(np.mean(aucs)),
        float(
            f1_score(
                np.concatenate(binary_truth),
                np.concatenate(binary_prediction),
                average="macro",
                zero_division=0,
            )
        ),
    )


@torch.no_grad()
def evaluate(
    model: SampledSparseGCN,
    loader: NeighborLoader,
    *,
    num_nodes: int,
    output_channels: int,
    device: torch.device,
    max_batches: int | None,
    partitioned: bool = False,
    full_topology: bool = False,
) -> torch.Tensor:
    output = torch.full(
        (int(num_nodes), int(output_channels)),
        float("nan"),
        dtype=torch.float32,
    )
    model.eval()
    for batch_number, batch in enumerate(loader):
        if max_batches is not None and batch_number >= max_batches:
            break
        if partitioned:
            global_ids = batch.n_id.clone()
        else:
            global_ids = batch.n_id[: int(batch.batch_size)].clone()
        batch = batch.to(device)
        batch_output = model(
            batch.x.float(),
            batch.edge_index,
            getattr(batch, "edge_attr", None),
            full_topology=full_topology,
        )
        if not partitioned:
            batch_output = batch_output[: int(batch.batch_size)]
        output[global_ids] = batch_output.detach().cpu()
    return output


def mask_indices(mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask.reshape(-1).bool())[0].cpu()


def metric_display_name(metric: str, multilabel: bool) -> str:
    if multilabel or str(metric).lower() == "rocauc":
        return "ROC-AUC"
    return "Accuracy"


def metric_triplet(
    data: Data,
    logits: torch.Tensor,
    *,
    multilabel: bool,
    metric: str = "acc",
) -> tuple[float, float, float, float, float]:
    implementation = multilabel_metrics if multilabel else single_label_metrics
    labels = data.y.cpu()
    train_metric, train_f1 = implementation(
        labels, logits, mask_indices(data.train_mask), metric=metric
    )
    valid_metric, _ = implementation(
        labels, logits, mask_indices(data.val_mask), metric=metric
    )
    test_metric, test_f1 = implementation(
        labels, logits, mask_indices(data.test_mask), metric=metric
    )
    return train_metric, valid_metric, test_metric, train_f1, test_f1


def selected_output_nodes(data: Data) -> torch.Tensor:
    return torch.unique(
        torch.cat(
            [
                mask_indices(data.train_mask),
                mask_indices(data.val_mask),
                mask_indices(data.test_mask),
            ]
        )
    )


def run_once(
    args: argparse.Namespace,
    *,
    run_index: int,
    data: Data,
    bundle,
    device: torch.device,
    fanouts: list[int],
    eval_fanouts: list[int],
) -> None:
    seed = int(args.seed)
    select_pyg_split(data, run_index)
    # NeighborLoader's pyg-lib CSC conversion requires a contiguous tensor.
    # Pokec's cached edge index can be a non-contiguous view.
    edge_index = data.edge_index.contiguous()
    original_edge_count = int(edge_index.size(1))
    fixed_achieved_ratio = float(args.kept_ratio)
    native_mog_run = (
        args.method == "mog"
        and getattr(args, "mog_source_profile", "scaled-legacy") != "scaled-legacy"
    )
    if native_mog_run:
        # MoG's SpLearner builds a sparse COO matrix from edge_index and then
        # reads back ``pi.coalesce().values()``. coalesce() merges duplicate
        # (row, col) pairs and sorts, so a graph carrying duplicate or
        # unsorted edges yields fewer/reordered values than the edge_index it
        # is paired with -- Products crashes with "indices and values must
        # have same nnz", and an unsorted-but-unique graph would silently
        # misalign scores instead. Coalescing here makes that a no-op.
        deduplicated = coalesce(
            edge_index,
            getattr(data, "edge_attr", None),
            num_nodes=int(data.num_nodes),
            reduce="mean",
        )
        if isinstance(deduplicated, tuple):
            edge_index, mog_edge_attr = deduplicated
        else:
            edge_index, mog_edge_attr = deduplicated, None
        edge_index = edge_index.contiguous()
        removed = original_edge_count - int(edge_index.size(1))
        if removed:
            print(
                f"[MoGSource] coalesced edge_index: dropped {removed} duplicate "
                f"edge(s); {original_edge_count} -> {int(edge_index.size(1))}",
                flush=True,
            )
        original_edge_count = int(edge_index.size(1))
    if args.method == "unified-lth":
        original_eval_edge_index = edge_index
        edge_index = load_or_create_ticket(
            edge_index,
            cache_root=Path(args.cache_root).expanduser().resolve(),
            dataset=bundle.name,
            kept_ratio=args.kept_ratio,
            seed=seed,
        )
        fixed_achieved_ratio = (
            int(edge_index.size(1)) / max(1, original_edge_count)
        )
    else:
        original_eval_edge_index = edge_index
    # Keep one shared CPU feature/label allocation across runs. Data.clone()
    # would duplicate hundreds of MiB for Products and its full edge index.
    graph = Data(
        x=data.x,
        y=data.y,
        edge_index=edge_index.contiguous(),
        # RandomNodeLoader preserves node attributes in induced subgraphs.  The
        # explicit global id lets partitioned evaluation place every prediction
        # back in the canonical dataset order.
        n_id=torch.arange(int(data.num_nodes), dtype=torch.long),
        num_nodes=int(data.num_nodes),
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
    )
    if native_mog_run:
        # Keep edge features aligned with the coalesced edge_index above.
        if mog_edge_attr is not None:
            graph.edge_attr = mog_edge_attr
    elif getattr(data, "edge_attr", None) is not None:
        graph.edge_attr = data.edge_attr

    evaluation_graph = graph
    if args.eval_graph == "original":
        evaluation_graph = Data(
            x=data.x,
            y=data.y,
            edge_index=original_eval_edge_index.contiguous(),
            n_id=torch.arange(int(data.num_nodes), dtype=torch.long),
            num_nodes=int(data.num_nodes),
            train_mask=data.train_mask,
            val_mask=data.val_mask,
            test_mask=data.test_mask,
        )
        if native_mog_run and mog_edge_attr is not None:
            evaluation_graph.edge_attr = mog_edge_attr
        elif getattr(data, "edge_attr", None) is not None:
            evaluation_graph.edge_attr = data.edge_attr
        print(
            "[EvaluationGraph] topology=original-full "
            f"directed_edges={evaluation_graph.edge_index.size(1)} "
            "training_topology=sparse",
            flush=True,
        )

    model = SampledSparseGCN(
        method=args.method,
        in_channels=int(graph.x.size(1)),
        hidden_channels=args.hidden_channels,
        out_channels=int(bundle.num_classes),
        kept_ratio=args.kept_ratio,
        dropout=args.dropout,
        input_dropout=getattr(args, "input_dropout", 0.0),
        temperature=args.temperature,
        layers=args.layers,
        residual=bool(getattr(args, "residual", 0)),
        layer_norm=bool(getattr(args, "layer_norm", 0)),
        batch_norm=bool(getattr(args, "batch_norm", 0)),
        pre_linear=bool(getattr(args, "pre_linear", 0)),
        jumping_knowledge=bool(
            getattr(args, "jumping_knowledge", 0)
        ),
        mog_source_profile=getattr(args, "mog_source_profile", "scaled-legacy"),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    partitioned = getattr(args, "loader_mode", "neighbor") == "random-node"
    if partitioned:
        train_loader = make_partition_loader(
            graph,
            parts=args.train_parts,
            workers=args.num_workers,
            shuffle=True,
        )
        evaluation_loader = FixedRandomNodePartitions(
            evaluation_graph, parts=args.eval_parts, seed=seed + 10_000
        )
        first_scheduled_eval = (
            (int(args.eval_start_epoch) + int(args.eval_step) - 1)
            // int(args.eval_step)
        ) * int(args.eval_step)
        first_scheduled_eval = min(int(args.epochs), max(1, first_scheduled_eval))
        print(
            f"[MoGPartitions] fixed_eval_parts={len(evaluation_loader)} "
            f"materialize_at_epoch={first_scheduled_eval}",
            flush=True,
        )
    else:
        train_nodes = mask_indices(graph.train_mask)
        output_nodes = selected_output_nodes(graph)
        # NeighborLoader converts the input edge index to its sampling CSC form
        # at construction time. Reusing these loaders avoids rebuilding that
        # O(E) structure on every epoch, especially for Products.
        train_loader = make_loader(
            graph,
            train_nodes,
            fanouts=fanouts,
            batch_size=args.batch_size,
            workers=args.num_workers,
            shuffle=True,
        )
        evaluation_loader = make_loader(
            evaluation_graph,
            output_nodes,
            fanouts=eval_fanouts,
            batch_size=args.eval_batch_size,
            workers=0,
            shuffle=False,
        )
    best_valid = float("-inf")
    best_epoch = 0
    best_values = (float("nan"),) * 4
    best_achieved_ratio = float(args.kept_ratio)
    training_time = 0.0
    budget = RunTimeBudget().start()

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        model.reset_mask_statistics()
        total_loss = 0.0
        total_examples = 0
        epoch_started = time.perf_counter()
        for batch_number, batch in enumerate(train_loader):
            if (
                args.max_train_batches is not None
                and batch_number >= args.max_train_batches
            ):
                break
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            batch_logits = model(
                batch.x.float(),
                batch.edge_index,
                getattr(batch, "edge_attr", None),
            )
            if partitioned:
                selected = batch.train_mask.reshape(-1).bool()
                if not bool(selected.any()):
                    continue
                logits = batch_logits[selected]
                labels = batch.y[selected]
                examples = int(selected.sum())
            else:
                examples = int(batch.batch_size)
                logits = batch_logits[:examples]
                labels = batch.y[:examples]
            if args.debug_numerics and not bool(torch.isfinite(logits).all()):
                raise FloatingPointError(
                    "non-finite logits before loss at "
                    f"epoch={epoch} batch={batch_number + 1}; "
                    f"parameters={nonfinite_parameter_names(model)}"
                )
            loss = masked_loss(
                logits,
                labels,
                multilabel=bundle.is_multilabel,
            )
            if args.method == "mog" and args.mog_source_profile != "scaled-legacy":
                loss = loss + 0.1 * model.auxiliary_loss
            loss_value = float(loss.detach().cpu())
            if not math.isfinite(loss_value):
                raise FloatingPointError(
                    "non-finite training loss at "
                    f"epoch={epoch} batch={batch_number + 1}; "
                    f"parameters={nonfinite_parameter_names(model)}"
                )
            if args.debug_numerics:
                with torch.autograd.detect_anomaly(check_nan=True):
                    loss.backward()
            else:
                loss.backward()
            if args.debug_numerics:
                invalid_gradients = nonfinite_parameter_names(
                    model,
                    gradients=True,
                )
                if invalid_gradients:
                    raise FloatingPointError(
                        "non-finite gradients at "
                        f"epoch={epoch} batch={batch_number + 1}; "
                        f"gradients={invalid_gradients}"
                    )
            optimizer.step()
            total_loss += loss_value * examples
            total_examples += examples
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_training_time = time.perf_counter() - epoch_started
        training_time += epoch_training_time
        if total_examples == 0:
            raise RuntimeError(
                f"training epoch {epoch} produced zero examples; "
                f"loader_mode={getattr(args, 'loader_mode', 'neighbor')}"
            )

        achieved = (
            fixed_achieved_ratio
            if args.method == "unified-lth"
            else model.achieved_kept_ratio
        )
        if epoch == 1 or epoch % int(args.display_step) == 0:
            print(
                f"[Progress] Run {run_index + 1:02d} Epoch {epoch:04d}/"
                f"{int(args.epochs):04d} "
                f"Loss {total_loss / total_examples:.4f} "
                f"Kept {100.0 * achieved:.2f}% "
                f"TrainSeconds {epoch_training_time:.2f}",
                flush=True,
            )

        should_evaluate = (
            epoch == int(args.epochs)
            or budget.expired
            or (
                epoch >= int(args.eval_start_epoch)
                and epoch % int(args.eval_step) == 0
            )
        )
        if not should_evaluate:
            continue
        logits = evaluate(
            model,
            evaluation_loader,
            num_nodes=int(graph.num_nodes),
            output_channels=bundle.num_classes,
            device=device,
            max_batches=args.max_eval_batches,
            partitioned=partitioned,
            full_topology=args.eval_graph == "original",
        )
        train_metric, valid_metric, test_metric, train_f1, test_f1 = metric_triplet(
            graph,
            logits,
            multilabel=bundle.is_multilabel,
            metric=args.metric,
        )
        if np.isfinite(valid_metric) and valid_metric > best_valid:
            best_valid = valid_metric
            best_epoch = epoch
            best_values = (train_metric, test_metric, train_f1, test_f1)
            best_achieved_ratio = achieved
        print(
            f"Run {run_index + 1:02d} Epoch {epoch:04d} "
            f"Loss {total_loss / total_examples:.4f} "
            f"Train {100.0 * train_metric:.2f}% "
            f"Valid {100.0 * valid_metric:.2f}% "
            f"Test {100.0 * test_metric:.2f}% "
            f"F1 {100.0 * test_f1:.2f}% "
            f"Kept {100.0 * achieved:.2f}%",
            flush=True,
        )
        if budget.exhausted(run_index + 1, epoch, int(args.epochs)):
            break

    train_metric, test_metric, train_f1, test_f1 = best_values
    print(
        f"{METHOD_LABELS[args.method]} scalable result: "
        f"metric={metric_display_name(args.metric, bundle.is_multilabel)} "
        f"best_epoch={best_epoch} valid={100.0 * best_valid:.2f}% "
        f"test={100.0 * test_metric:.2f}% f1={100.0 * test_f1:.2f}% "
        f"target_ratio={args.kept_ratio:.6f} "
        f"achieved_ratio={best_achieved_ratio:.6f} "
        f"training_time={training_time:.2f}s",
        flush=True,
    )
    append_baseline_result(
        method=args.method.replace("-", "_"),
        dataset=bundle.name,
        run=run_index + 1,
        seed=seed,
        epochs=args.epochs,
        kept_ratio=args.kept_ratio,
        sparsity=100.0 * (1.0 - best_achieved_ratio),
        train_acc=100.0 * train_metric,
        valid_acc=100.0 * best_valid,
        test_acc=100.0 * test_metric,
        train_f1_macro=100.0 * train_f1,
        test_f1_macro=100.0 * test_f1,
        chosen_epoch=best_epoch,
        metric="rocauc" if bundle.is_multilabel else args.metric,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHOD_LABELS), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", "--data_root", required=True)
    parser.add_argument("--cache-root", "--cache_root", required=True)
    parser.add_argument(
        "--kept-ratio", "--kept_ratio", type=float, required=True
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", "--num-layers", "--num_layers", type=int, default=2)
    parser.add_argument("--hidden-channels", "--hidden_channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", "--learning_rate", "--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=5e-4)
    parser.add_argument("--input-dropout", "--input_dropout", type=float, default=0.0)
    parser.add_argument("--metric", choices=("acc", "rocauc"), default="acc")
    parser.add_argument(
        "--eval-graph",
        "--eval_graph",
        choices=("sparse", "original"),
        default=os.environ.get("BASELINE_EVAL_GRAPH", "sparse"),
        help=(
            "Topology used by validation/test inference. Training always uses "
            "the method's existing sparse or learned-mask topology."
        ),
    )
    parser.add_argument("--pre-linear", "--pre_linear", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--residual",
        "--use_res_value",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument(
        "--layer-norm",
        "--layer_norm",
        "--use_ln",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument(
        "--batch-norm",
        "--batch_norm",
        "--use_bn_value",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument("--jumping-knowledge", "--jumping_knowledge", type=int, choices=(0, 1), default=0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument(
        "--mog-source-profile",
        choices=("scaled-legacy", "ogbn-arxiv", "ogbn-proteins"),
        default="scaled-legacy",
        help="Native MoG topology-learner implementation used for MoG runs.",
    )
    parser.add_argument("--fanouts", default="15,10")
    parser.add_argument(
        "--eval-fanouts",
        "--eval_fanouts",
        help="Evaluation fanout per model layer; defaults to --fanouts.",
    )
    parser.add_argument("--batch-size", "--batch_size", type=int, default=8192)
    parser.add_argument(
        "--eval-batch-size", "--eval_batch_size", type=int, default=8192
    )
    parser.add_argument("--num-workers", "--num_workers", type=int, default=8)
    parser.add_argument(
        "--loader-mode",
        choices=("neighbor", "random-node"),
        default="neighbor",
        help=(
            "neighbor sampling or native large-graph random induced-node "
            "partitions"
        ),
    )
    parser.add_argument("--train-parts", "--train_parts", type=int, default=10)
    parser.add_argument("--eval-parts", "--eval_parts", type=int, default=10)
    parser.add_argument("--eval-step", "--eval_step", type=int, default=10)
    parser.add_argument(
        "--eval-start-epoch",
        "--eval_start_epoch",
        type=int,
        default=1,
    )
    parser.add_argument("--display-step", "--display_step", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument(
        "--debug-numerics",
        action="store_true",
        help="Fail at the first non-finite logit or gradient and report its batch.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.kept_ratio <= 1.0:
        raise ValueError("--kept-ratio must be in (0, 1]")
    for name in (
        "epochs",
        "runs",
        "layers",
        "hidden_channels",
        "batch_size",
        "eval_batch_size",
        "train_parts",
        "eval_parts",
        "eval_step",
        "eval_start_epoch",
        "display_step",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if args.loader_mode == "neighbor":
        fanouts = parse_fanouts(args.fanouts, args.layers)
        eval_fanouts = parse_fanouts(
            args.eval_fanouts if args.eval_fanouts is not None else args.fanouts,
            args.layers,
        )
        loader_description = (
            f"neighbor train_fanouts={fanouts} eval_fanouts={eval_fanouts} "
            f"batch_size={args.batch_size} "
            f"eval_batch_size={args.eval_batch_size}"
        )
    else:
        fanouts = []
        eval_fanouts = []
        loader_description = (
            f"random-node train_parts={args.train_parts} "
            f"eval_parts={args.eval_parts}"
        )
    device = resolve_device(args.device)
    print(
        f"{METHOD_LABELS[args.method]} large-graph mode: "
        f"sparse {loader_description} (no dense adjacency); "
        f"workers={args.num_workers} "
        f"eval_start={args.eval_start_epoch} eval_every={args.eval_step} "
        f"device={device} target_ratio={args.kept_ratio}",
        flush=True,
    )
    if args.method == "mog":
        print(
            f"[MoGSource] profile={args.mog_source_profile} "
            "native_topology_learner=true tunedgnn_backbone=true",
            flush=True,
        )
    data, bundle = load_pyg_data(
        args.data_root,
        args.dataset,
        seed=args.seed,
        run_index=0,
    )
    data = data.cpu()
    data.x = data.x.float()
    set_seed(args.seed)
    for run_index in range(args.runs):
        run_once(
            args,
            run_index=run_index,
            data=data,
            bundle=bundle,
            device=device,
            fanouts=fanouts,
            eval_fanouts=eval_fanouts,
        )


if __name__ == "__main__":
    main()
