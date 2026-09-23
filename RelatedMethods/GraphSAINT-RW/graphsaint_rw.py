"""GraphSAINT-RW node classification baseline.

Follows the canonical PyG example ``examples/graph_saint.py`` verbatim in
sampler + model, adapted to:
  - Datasets loaded through ``scripts.common.baseline_dataset_bridge``
    (Cora / Reddit / OGB-products / OGB-arxiv / OGB-proteins / Pokec).
  - Full-input-graph training via random-walk sampled subgraphs.
  - Layer-wise NeighborLoader inference for graphs where full-graph eval OOMs
    (e.g. OGB-products, 2.4 M nodes → 46 GiB message tensor on an 80 GiB GPU).

Reference: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_saint.py
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.loader import GraphSAINTRandomWalkSampler, NeighborLoader
from torch_geometric.nn import GraphConv
from torch_geometric.utils import coalesce, degree, is_undirected, remove_self_loops, to_undirected

SUPPORT_GRAPH_ROOT = Path(os.environ.get("SUPPORT_GRAPH_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(SUPPORT_GRAPH_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPPORT_GRAPH_ROOT))
from BenchmarkDataset import load_pyg_data, select_pyg_split
from scripts.common.baseline_result_utils import (
    RunTimeBudget,
    append_baseline_result,
    macro_f1_percent,
    multilabel_roc_auc_f1_percent,
    single_label_roc_auc_percent,
)
from scaffold_gnn.utils.defaults import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR


def parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def graphsaint_sampler_cache_dir(cache_root: str, dataset_name: str, data, args) -> str:
    """Return a cache directory unique to this graph and sampler setup.

    PyG's normalization filename does not identify the dataset or every sampler
    setting. Reusing one save directory across datasets can therefore load
    tensors with the wrong node and edge counts. Keep a small human-readable
    dataset level plus a stable fingerprint of every setting that changes the
    normalization statistics.
    """
    if not cache_root:
        return ""
    canonical = str(dataset_name).strip().lower().replace("_", "-")
    metadata = {
        "schema": 3,
        "dataset": canonical,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "batch_size": int(args.batch_size),
        "walk_length": int(args.walk_length),
        "num_steps": int(args.num_steps),
        "sample_coverage": int(args.sample_coverage),
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()[:16]
    cache_dir = Path(cache_root).expanduser() / canonical / fingerprint
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    # PyG 2.x derives the file name from the lower-cased sampler class, walk
    # length, and coverage. Keep the former local spelling in the validation
    # list as well so caches produced by either supported environment are
    # checked before the loader consumes them.
    norm_paths = (
        cache_dir
        / (
            f"graphsaintrandomwalksampler_{args.walk_length}_"
            f"{args.sample_coverage}.pt"
        ),
        cache_dir
        / f"graphsaint_random_walk_sampler_{args.sample_coverage}.pt",
    )
    for norm_path in norm_paths:
        if not norm_path.exists():
            continue
        try:
            node_norm, edge_norm = torch.load(
                norm_path, map_location="cpu", weights_only=False
            )
            valid = (
                int(node_norm.numel()) == int(data.num_nodes)
                and int(edge_norm.numel()) == int(data.edge_index.size(1))
            )
        except Exception:
            valid = False
        if not valid:
            print(
                "[Sampler cache] removing incompatible normalization cache: "
                f"{norm_path}",
                flush=True,
            )
            norm_path.unlink(missing_ok=True)
    return str(cache_dir)


def build_sgs_split(num_nodes: int, labels=None, seed: int = 1):
    if labels is not None:
        labels = np.asarray(labels).reshape(-1)
        indices = np.where(labels != -1)[0]
    else:
        indices = np.arange(num_nodes)
    train_idx, temp_idx = train_test_split(indices, test_size=0.8, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def load_sgs_dataset(dataset_name: str, data_root: str):
    data, dataset = load_pyg_data(data_root, dataset_name)
    if not is_undirected(data.edge_index):
        data.edge_index = to_undirected(data.edge_index)
    if data.y.dim() > 1:
        if data.y.size(-1) == 1:
            data.y = data.y.view(-1)
        else:
            # Preserve OGBN-Proteins' 112 independent binary targets.
            data.y = data.y.to(torch.float)
    else:
        data.y = data.y.view(-1)
    if data.y.dim() == 1:
        data.y = data.y.to(torch.long)
    return dataset, data


class SaintNet(torch.nn.Module):
    """GraphSAINT GraphConv backbone using tunedGNN's dataset preset."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.2,
        input_dropout: float = 0.0,
        pre_linear: bool = False,
        residual: bool = False,
        layer_norm: bool = False,
        batch_norm: bool = False,
        multilabel: bool = False,
    ):
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("num_layers must be positive")
        self.pre_linear = bool(pre_linear)
        self.residual = bool(residual)
        self.layer_norm = bool(layer_norm)
        self.batch_norm = bool(batch_norm)
        self.input_dropout = float(input_dropout)
        self.input_linear = torch.nn.Linear(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        self.residual_lins = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(int(num_layers)):
            layer_input = (
                hidden_channels
                if self.pre_linear or layer > 0
                else in_channels
            )
            self.convs.append(GraphConv(layer_input, hidden_channels))
            self.residual_lins.append(
                torch.nn.Linear(layer_input, hidden_channels)
            )
            self.layer_norms.append(torch.nn.LayerNorm(hidden_channels))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lin = torch.nn.Linear(
            int(num_layers) * hidden_channels, out_channels
        )
        self.dropout = dropout
        self.multilabel = bool(multilabel)

    def set_aggr(self, aggr: str):
        for conv in self.convs:
            conv.aggr = aggr

    def forward(self, x0, edge_index, edge_weight=None):
        x = x0
        if self.pre_linear:
            x = self.input_linear(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        elif self.input_dropout > 0:
            x = F.dropout(
                x, p=self.input_dropout, training=self.training
            )
        representations = []
        for layer, conv in enumerate(self.convs):
            previous = x
            x = conv(x, edge_index, edge_weight)
            if self.residual:
                x = x + self.residual_lins[layer](previous)
            if self.layer_norm:
                x = self.layer_norms[layer](x)
            elif self.batch_norm:
                x = self.batch_norms[layer](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            representations.append(x)
        x = torch.cat(representations, dim=-1)
        x = self.lin(x)
        return x if self.multilabel else x.log_softmax(dim=-1)

    @torch.no_grad()
    def inference(self, x_all: torch.Tensor, subgraph_loader: NeighborLoader, device: torch.device):
        """Layer-wise NeighborLoader inference for graphs too big for full-graph eval.

        Streams each conv over all nodes with a 1-hop NeighborLoader, storing
        intermediate activations on CPU. Mirrors the pattern used in PyG's
        ``examples/reddit.py``. The final linear head is applied per-batch to
        avoid materializing the 3H-wide concat over all nodes at once.
        """
        self.eval()
        self.set_aggr("mean")

        def _run_layer(x_all_cpu, conv, apply_relu):
            xs = torch.empty((x_all_cpu.size(0), conv.out_channels))
            for batch in subgraph_loader:
                n_id_cpu = batch.n_id.cpu()
                xb = x_all_cpu[n_id_cpu].to(device)
                edge_index = batch.edge_index.to(device)
                h = conv(xb, edge_index)
                if apply_relu:
                    h = F.relu(h)
                bs = int(batch.batch_size)
                xs[n_id_cpu[:bs]] = h[:bs].cpu()
            return xs

        if self.pre_linear:
            projected = []
            step = 200_000
            for start in range(0, x_all.size(0), step):
                projected.append(
                    self.input_linear(
                        x_all[start : start + step].to(device)
                    ).cpu()
                )
            x_all = torch.cat(projected, dim=0)
        representations = []
        for layer, conv in enumerate(self.convs):
            previous = x_all
            x_all = _run_layer(previous, conv, apply_relu=False)
            if self.residual:
                residual_blocks = []
                step = 200_000
                for start in range(0, previous.size(0), step):
                    residual_blocks.append(
                        self.residual_lins[layer](
                            previous[start : start + step].to(device)
                        ).cpu()
                    )
                x_all = x_all + torch.cat(residual_blocks, dim=0)
            if self.layer_norm:
                x_all = self.layer_norms[layer](x_all.to(device)).cpu()
            elif self.batch_norm:
                x_all = self.batch_norms[layer](x_all.to(device)).cpu()
            x_all = F.relu(x_all)
            representations.append(x_all)
        num_nodes = x_all.size(0)
        out = torch.empty((num_nodes, self.lin.out_features))
        step = 200_000
        for start in range(0, num_nodes, step):
            end = min(start + step, num_nodes)
            block = torch.cat(
                [value[start:end] for value in representations], dim=-1
            ).to(device)
            block_output = self.lin(block)
            if not self.multilabel:
                block_output = block_output.log_softmax(dim=-1)
            out[start:end] = block_output.cpu()
        return out


def build_edge_inputs(model, batch, use_saint_norm: bool, device):
    """Return the ``(edge_index, edge_weight)`` pair for one forward pass.

    The GraphConv backbone keeps the PyG example's contract: normalized
    sampled training folds ``edge_norm`` into ``edge_weight`` and switches the
    aggregation to ``add``; every other pass aggregates unweighted with
    ``mean``. A model that needs a different convention -- e.g. a GCN backbone
    whose symmetric normalization must be computed on the *full* graph rather
    than on whatever the sampler happened to return -- supplies its own
    ``build_edge_inputs``.
    """
    builder = getattr(model, "build_edge_inputs", None)
    if builder is not None:
        return builder(batch, use_saint_norm, device)
    edge_index = batch.edge_index.to(device)
    if not use_saint_norm:
        return edge_index, None
    return edge_index, (batch.edge_norm * batch.edge_weight).to(device)


def labeled_node_mask(y: torch.Tensor, multilabel: bool) -> torch.Tensor:
    """Nodes carrying at least one supervised target."""
    if multilabel:
        return (torch.isfinite(y) & (y >= 0)).any(dim=-1)
    return y >= 0


def normalized_loss_scale(args, data, multilabel: bool) -> float:
    """Rescale GraphSAINT's node-normalized loss to a per-train-node mean.

    ``(loss * node_norm)[train].sum()`` is an unbiased estimate of
    ``sum(loss over train nodes) / num_nodes``, not of the *mean* over train
    nodes that full-graph training minimizes. The two differ by the factor
    ``num_train / num_nodes`` -- 140/2,708 on Cora's tunedGNN split.

    Adam rescales the data gradient away, but ``weight_decay`` is added to that
    gradient *before* the moment estimates, so shrinking the data term by 19x
    silently multiplies the effective regularization by the same amount. The
    tunedGNN presets pin ``weight_decay`` against a per-train-node mean loss,
    so keeping that scale is what makes the inherited preset mean here what it
    means for full-graph tunedGNN. GraphSAINT's sampling-bias correction is
    untouched: only a constant multiplies the estimator.

    Measured on Cora over 3 runs this is a wash (79.7 +- 0.6 with
    ``graphsaint_sum`` vs 79.0 +- 0.8 with ``train_mean`` for GraphSAINT-RW;
    79.1 +- 1.4 vs 79.3 +- 1.4 for the tunedGNN backbone), so the PyG scale
    stays the default and this is opt-in. It is inert on every large dataset
    anyway, because ``use_normalization=auto`` disables the correction there.

    **That Cora measurement does not generalize, and reading it as "this never
    matters" was wrong.** What matters is the product ``weight_decay *
    num_nodes / num_train``, and Cora sits at 0.0005 * 19.3 = 0.0097, near the
    bottom of the range. The tunedGNN presets put citeseer at 0.01 * 27.7 =
    0.277 and pubmed at 0.0005 * 328.6 = 0.164 -- 17x and 10x higher than any
    other dataset in the study. Both collapse under ``graphsaint_sum``: train
    accuracy pins at exactly 1/num_classes with a flat loss, i.e. weight decay
    zeroes the weights before the data term can move them (citeseer 24.8 vs
    67.8 for the untuned GraphSAINT-RW baseline, pubmed 38.4 vs 59.9). Every
    other dataset in the study has ``weight_decay * N/num_train`` <= 0.01 and
    is unaffected either way.

    ``run_tunedgnn_graphsaint.sh`` therefore passes ``train_mean``. The default
    here stays ``graphsaint_sum`` because plain GraphSAINT-RW is meant to
    reproduce the PyG example and does not inherit the tunedGNN presets.
    """
    if not args.use_normalization:
        return 1.0
    if args.normalized_loss_scale != "train_mean":
        return 1.0
    labeled = labeled_node_mask(data.y, multilabel)
    num_train = int((data.train_mask.bool() & labeled).sum())
    return float(data.num_nodes) / max(num_train, 1)


def macro_f1(pred: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.bool()
    labels = labels[mask]
    preds = pred[mask]
    if labels.numel() == 0:
        return 0.0
    return macro_f1_percent(labels, preds) / 100.0


@torch.no_grad()
def evaluate(
    model,
    data,
    use_normalization,
    device,
    full_graph_eval,
    subgraph_loader,
    multilabel,
    metric="acc",
):
    # Follows PyG's examples/graph_saint.py test(): plain
    #   correct[mask].sum() / mask.sum()
    # accuracy on the full graph. Both eval paths (full-graph vs layer-wise
    # NeighborLoader) return log-softmax logits; argmax is the prediction.
    # Minesweeper and Questions instead pass --metric rocauc, because their
    # accuracy is the 80%/97% majority-class rate however the model ranks nodes.
    model.eval()
    model.set_aggr("mean")
    if full_graph_eval:
        edge_index, edge_weight = build_edge_inputs(
            model, data, use_saint_norm=False, device=device
        )
        logits = model(data.x.to(device), edge_index, edge_weight)
    else:
        logits = model.inference(data.x, subgraph_loader, device)
    if multilabel:
        labels = data.y.cpu()
        logits = logits.cpu()

        def _metric(mask):
            mask = mask.cpu().bool()
            metric, f1 = multilabel_roc_auc_f1_percent(
                labels[mask], logits[mask]
            )
            return metric / 100.0, f1 / 100.0

        train_metric, train_f1_macro = _metric(data.train_mask)
        val_metric, _ = _metric(data.val_mask)
        test_metric, test_f1_macro = _metric(data.test_mask)
        return (
            train_metric,
            val_metric,
            test_metric,
            train_f1_macro,
            test_f1_macro,
        )

    pred = logits.argmax(dim=-1)
    labels = data.y.to(pred.device)
    correct = pred.eq(labels)

    def _acc(mask):
        mask = mask.to(pred.device).bool()
        valid = mask & (labels >= 0)
        denom = int(valid.sum().item())
        if denom == 0:
            return 0.0
        return float(correct[valid].sum().item()) / denom

    def _rocauc(mask):
        mask = mask.to(pred.device).bool()
        valid = mask & (labels >= 0)
        if not bool(valid.any()):
            return 0.0
        return single_label_roc_auc_percent(
            labels[valid].cpu(), logits[valid].cpu()
        ) / 100.0

    _metric = _rocauc if str(metric).lower() == "rocauc" else _acc
    train_acc = _metric(data.train_mask)
    val_acc = _metric(data.val_mask)
    test_acc = _metric(data.test_mask)
    train_f1_macro = macro_f1(pred, labels, data.train_mask)
    test_f1_macro = macro_f1(pred, labels, data.test_mask)
    return train_acc, val_acc, test_acc, train_f1_macro, test_f1_macro


def _multilabel_loss(logits, labels, node_mask, node_norm=None):
    valid = torch.isfinite(labels) & (labels >= 0)
    losses = F.binary_cross_entropy_with_logits(
        logits,
        torch.nan_to_num(labels, nan=0.0).float(),
        reduction="none",
    )
    valid_count = valid.sum(dim=-1)
    per_node = (losses * valid).sum(dim=-1) / valid_count.clamp(min=1)
    selected = node_mask.bool() & (valid_count > 0)
    if not bool(selected.any()):
        return logits.sum() * 0.0
    if node_norm is not None:
        return (per_node * node_norm)[selected].sum()
    return per_node[selected].mean()


def train_one_run(
    args,
    dataset_name,
    data,
    num_features,
    num_classes,
    run_seed,
    run_id,
    multilabel,
    model_factory=None,
    result_method="graphsaint",
):
    select_pyg_split(data, run_id - 1)
    single_label_metric = str(getattr(args, "metric", "acc")).lower()
    metric_name = (
        "ROC-AUC" if multilabel or single_label_metric == "rocauc" else "Accuracy"
    )
    loss_scale = normalized_loss_scale(args, data, multilabel)

    if model_factory is None:
        model_factory = SaintNet
    model = model_factory(
        in_channels=num_features,
        hidden_channels=args.hidden_channels,
        out_channels=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        input_dropout=args.input_dropout,
        pre_linear=bool(args.pre_linear),
        residual=bool(args.residual),
        layer_norm=bool(args.layer_norm),
        batch_norm=bool(args.batch_norm),
        multilabel=multilabel,
    ).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    saint_kwargs = dict(
        batch_size=args.batch_size,
        walk_length=args.walk_length,
        num_steps=args.num_steps,
        sample_coverage=args.sample_coverage,
        num_workers=args.sampler_num_workers,
    )
    sampler_cache_dir = graphsaint_sampler_cache_dir(
        args.sampler_cache_dir, dataset_name, data, args
    )
    if sampler_cache_dir:
        saint_kwargs["save_dir"] = sampler_cache_dir
    print(
        f"[Sampler init] batch_size={args.batch_size} walk_length={args.walk_length} "
        f"num_steps={args.num_steps} sample_coverage={args.sample_coverage} "
        f"save_dir={sampler_cache_dir or None}",
        flush=True,
    )
    loader = GraphSAINTRandomWalkSampler(data, **saint_kwargs)

    full_graph_eval = (
        int(data.num_nodes) <= args.full_graph_eval_max_nodes
        and int(data.edge_index.size(1)) <= args.full_graph_eval_max_edges
    )
    subgraph_loader = None
    if not full_graph_eval:
        subgraph_loader = NeighborLoader(
            data,
            num_neighbors=[-1],
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.sampler_num_workers,
        )
    print(
        f"[Eval mode] full_graph_eval={full_graph_eval} "
        f"nodes={int(data.num_nodes)} edges={int(data.edge_index.size(1))}",
        flush=True,
    )
    print(
        f"[Loss scale] use_normalization={bool(args.use_normalization)} "
        f"mode={args.normalized_loss_scale} factor={loss_scale:.4f}",
        flush=True,
    )

    best_val = float("-inf")
    chosen_test = 0.0
    chosen_train = 0.0
    chosen_test_macro = 0.0
    chosen_train_macro = 0.0
    chosen_epoch = 0
    train_f1 = val_f1 = test_f1 = 0.0
    budget = RunTimeBudget().start()

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.set_aggr("add" if args.use_normalization else "mean")
        total_loss = 0.0
        total_examples = 0

        for batch in loader:
            batch = batch.to(args.device)
            optimizer.zero_grad()

            # Datasets such as Pokec carry -1 for unlabeled nodes. The
            # normalized branch scores every node in the subgraph, so those
            # targets have to be excluded before they reach ``nll_loss``.
            train_mask_batch = batch.train_mask.bool() & labeled_node_mask(
                batch.y, multilabel
            )
            n_train = int(train_mask_batch.sum())
            if n_train == 0:
                continue

            edge_index, edge_weight = build_edge_inputs(
                model, batch, bool(args.use_normalization), args.device
            )
            out = model(batch.x, edge_index, edge_weight)
            if args.use_normalization:
                if multilabel:
                    loss = _multilabel_loss(
                        out,
                        batch.y,
                        train_mask_batch,
                        node_norm=batch.node_norm,
                    )
                else:
                    loss = F.nll_loss(
                        out, batch.y.clamp(min=0), reduction="none"
                    )
                    loss = (loss * batch.node_norm)[train_mask_batch].sum()
                loss = loss * loss_scale
            else:
                if multilabel:
                    loss = _multilabel_loss(
                        out, batch.y, train_mask_batch
                    )
                else:
                    loss = F.nll_loss(
                        out[train_mask_batch], batch.y[train_mask_batch]
                    )

            loss.backward()
            optimizer.step()
            # Match the official PyG GraphSAINT example. In particular, do not
            # divide the node-normalized loss by the number of labeled nodes a
            # second time merely for reporting.
            total_loss += float(loss.item()) * int(batch.num_nodes)
            total_examples += int(batch.num_nodes)

        if (
            epoch == 1
            or epoch % args.eval_step == 0
            or epoch == args.epochs
            or budget.expired
        ):
            train_f1, val_f1, test_f1, train_f1_macro, test_f1_macro = evaluate(
                model,
                data,
                args.use_normalization,
                args.device,
                full_graph_eval,
                subgraph_loader,
                multilabel,
                metric=single_label_metric,
            )
            avg_loss = total_loss / max(total_examples, 1)
            if val_f1 > best_val:
                best_val = val_f1
                chosen_test = test_f1
                chosen_train = train_f1
                chosen_test_macro = test_f1_macro
                chosen_train_macro = train_f1_macro
                chosen_epoch = epoch
            print(
                f"Epoch: {epoch:03d}, Loss: {avg_loss:.4f}, "
                f"Train: {train_f1 * 100:.2f}%, Val: {val_f1 * 100:.2f}%, "
                f"Test: {test_f1 * 100:.2f}%, Test F1 Macro: {test_f1_macro * 100:.2f}%",
                flush=True,
            )
        if budget.exhausted(run_id, epoch, args.epochs):
            break

    print(
        f"run_{run_id} metric={metric_name} train_metric: {train_f1:.6f} "
        f"val_metric: {val_f1:.6f} test_metric: {test_f1:.6f} "
        f"best_val_metric: {best_val:.6f} "
        f"chosen_test_metric: {chosen_test:.6f} "
        f"chosen_test_f1_macro: {chosen_test_macro:.6f} "
        f"chosen_epoch: {chosen_epoch}"
    )
    append_baseline_result(
        method=result_method,
        dataset=dataset_name,
        run=run_id,
        seed=run_seed,
        epochs=args.epochs,
        train_acc=100 * chosen_train,
        valid_acc=100 * best_val,
        test_acc=100 * chosen_test,
        train_f1_macro=100 * chosen_train_macro,
        test_f1_macro=100 * chosen_test_macro,
        chosen_epoch=chosen_epoch,
        metric="rocauc" if metric_name == "ROC-AUC" else "acc",
    )
    return chosen_test


def _default_batch_size(num_nodes: int) -> int:
    # Rough tier defaults tuned for graph coverage and GPU memory:
    #  small graphs (< 50k nodes): 500
    #  reddit-scale (< 500k):      6000  (matches PyG example)
    #  OGB-products / pokec:       20000
    if num_nodes < 50_000:
        return 500
    if num_nodes < 500_000:
        return 6000
    return 20_000


def resolve_sampler_settings(args, num_nodes: int):
    """Resolve graph-scale GraphSAINT defaults while preserving CLI overrides.

    GraphSAINT defines an epoch by graph coverage. A Cora batch with 2,000
    random-walk roots already covers most of its 2,708 nodes, so the previous
    universal 30-step setting performed many updates before the first
    validation. Small graphs instead use smaller subgraphs, three coverage
    steps, and the paper's node/edge sampling-bias normalization.
    """

    is_small_graph = int(num_nodes) < 50_000
    if int(args.batch_size) <= 0:
        args.batch_size = _default_batch_size(int(num_nodes))
    if int(args.walk_length) <= 0:
        args.walk_length = 2 if is_small_graph else 4
    if int(args.num_steps) <= 0:
        args.num_steps = 3 if is_small_graph else 30

    normalization = str(args.use_normalization).strip().lower()
    if normalization == "auto":
        args.use_normalization = is_small_graph
    else:
        args.use_normalization = parse_bool(args.use_normalization)

    if int(args.sample_coverage) < 0:
        args.sample_coverage = 100 if args.use_normalization else 0

    if int(args.batch_size) < 1:
        raise ValueError("batch_size must be positive after default resolution")
    if int(args.walk_length) < 1:
        raise ValueError("walk_length must be positive after default resolution")
    if int(args.num_steps) < 1:
        raise ValueError("num_steps must be positive after default resolution")
    if int(args.sample_coverage) < 0:
        raise ValueError("sample_coverage must be non-negative")
    return args


def main(
    model_factory=None,
    result_method="graphsaint",
    description=None,
    data_prep=None,
):
    parser = argparse.ArgumentParser(
        description=description
        or "GraphSAINT-RW baseline (PyG official sampler)"
    )
    parser.add_argument("--dataset", required=True)
    # Resolve to the SAME tree main.py / scaffold_fast use (DEFAULT_DATA_DIR in
    # utils/defaults.py). That way we always load the same cached Reddit /
    # ogbn_products / etc. and don't re-download per node.
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--input_dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--metric", choices=("acc", "rocauc"), default="acc")
    parser.add_argument("--pre_linear", type=int, choices=(0, 1), default=0)
    parser.add_argument("--residual", type=int, choices=(0, 1), default=0)
    parser.add_argument("--layer_norm", type=int, choices=(0, 1), default=0)
    parser.add_argument("--batch_norm", type=int, choices=(0, 1), default=0)
    parser.add_argument("--batch_size", type=int, default=0,
                        help="0 = graph-scale default (500 / 6000 / 20000).")
    parser.add_argument("--walk_length", type=int, default=0,
                        help="0 = graph-scale default (2 small, 4 large).")
    parser.add_argument("--num_steps", type=int, default=0,
                        help="0 = graph-coverage default (3 small, 30 large).")
    parser.add_argument("--sample_coverage", type=int, default=-1,
                        help="-1 = 100 with bias normalization, otherwise 0.")
    parser.add_argument("--sampler_num_workers", type=int, default=0)
    parser.add_argument("--eval_step", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=4096)
    parser.add_argument("--full_graph_eval_max_nodes", type=int, default=50_000,
                        help="Use full-graph eval below this many nodes; else layer-wise NeighborLoader eval.")
    parser.add_argument("--full_graph_eval_max_edges", type=int, default=2_000_000,
                        help="Also fall back to layer-wise NeighborLoader eval above this many edges.")
    parser.add_argument("--sampler_cache_dir", type=str,
                        default=os.environ.get(
                            "GRAPHSAINT_SAMPLER_CACHE",
                            str(Path(DEFAULT_CACHE_DIR) / "graphsaint" / "sampler"),
                        ))
    parser.add_argument(
        "--normalized_loss_scale",
        choices=("graphsaint_sum", "train_mean"),
        default="graphsaint_sum",
        help=(
            "graphsaint_sum (default) keeps the PyG example's sum/num_nodes "
            "scale. train_mean rescales the node-normalized loss to the "
            "per-train-node mean the tunedGNN presets pin weight_decay "
            "against; measured as a wash on Cora."
        ),
    )
    parser.add_argument(
        "--use_normalization",
        type=str,
        default="auto",
        help=(
            "true/false/auto. Auto enables GraphSAINT node/edge bias "
            "normalization on small graphs and disables it on large graphs."
        ),
    )
    args = parser.parse_args()

    if args.cpu or not torch.cuda.is_available():
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(f"cuda:{args.device}")

    fix_seed(args.seed)
    _dataset, data = load_sgs_dataset(args.dataset, args.data_root)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    data.edge_index = coalesce(data.edge_index, num_nodes=data.num_nodes)
    num_edges = int(data.edge_index.size(1))

    # PyG graph_saint.py convention: edge_weight = 1 / in_degree(col)
    _row, col = data.edge_index
    deg = degree(col, data.num_nodes, dtype=torch.float)
    deg = torch.clamp(deg, min=1.0)
    data.edge_weight = 1.0 / deg[col]

    multilabel = data.y.dim() > 1 and data.y.size(-1) > 1
    # Heterophilous ROC-AUC datasets (minesweeper, questions) are single label,
    # so the reported metric follows --metric there rather than multilabel.
    rocauc_metric = multilabel or str(getattr(args, "metric", "acc")).lower() == "rocauc"
    if multilabel:
        num_classes = int(data.y.size(-1))
    else:
        labeled = data.y[data.y != -1]
        num_classes = int(labeled.max().item()) + 1 if labeled.numel() else 0
    num_features = int(data.x.size(1))

    # Backbones that normalize on the full graph attach their own edge-level
    # tensors here, before the sampler indexes them alongside edge_weight.
    if data_prep is not None:
        data_prep(data)

    resolve_sampler_settings(args, int(data.num_nodes))

    print(
        f"dataset {args.dataset} | num nodes {data.num_nodes} | "
        f"num edge {num_edges} | num node feats {num_features} | "
        f"num outputs {num_classes} | "
        f"metric {'ROC-AUC' if rocauc_metric else 'Accuracy'}"
    )
    print(
        f"[Resolved sampler] batch_size={args.batch_size} "
        f"walk_length={args.walk_length} num_steps={args.num_steps} "
        f"sample_coverage={args.sample_coverage} "
        f"use_normalization={args.use_normalization}"
    )

    run_scores = []
    for run_idx in range(1, args.runs + 1):
        # tunedGNN seeds once before its run loop; repeated runs advance the
        # same RNG stream instead of replacing the configured seed.
        run_seed = args.seed
        score = train_one_run(
            args,
            args.dataset,
            data,
            num_features,
            num_classes,
            run_seed,
            run_idx,
            multilabel,
            model_factory=model_factory,
            result_method=result_method,
        )
        run_scores.append(score)

    run_scores = np.asarray(run_scores, dtype=np.float64)
    mean = float(run_scores.mean()) if len(run_scores) else 0.0
    std = float(run_scores.std(ddof=1)) if len(run_scores) > 1 else 0.0
    metric_key = "roc_auc" if rocauc_metric else "accuracy"
    print(
        f"all_runs chosen_test_{metric_key}_mean: {mean:.6f} "
        f"chosen_test_{metric_key}_std: {std:.6f}"
    )


if __name__ == "__main__":
    main()
