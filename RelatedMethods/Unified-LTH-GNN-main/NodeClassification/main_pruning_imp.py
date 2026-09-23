"""Unified-LTH IMP adapted to the shared Benchmark benchmark protocol."""

from __future__ import annotations

import argparse
import copy
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score

import net
import pruning
import utils
from utils import load_data

SUPPORT_GRAPH_ROOT = Path(
    os.environ.get(
        "SUPPORT_GRAPH_ROOT",
        Path(__file__).resolve().parents[3],
    )
).resolve()
if str(SUPPORT_GRAPH_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPPORT_GRAPH_ROOT))

from scripts.common.baseline_result_utils import (  # noqa: E402
    RunTimeBudget,
    append_baseline_result,
)
from scaffold_gnn.utils.defaults import DEFAULT_DATA_DIR  # noqa: E402

warnings.filterwarnings("ignore")


def _device(value: str) -> torch.device:
    requested = str(value).strip().lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"[Device] requested={value} CUDA unavailable; using CPU",
            flush=True,
        )
        return torch.device("cpu")
    return torch.device(value)


def _build_model(
    args: dict[str, Any],
    adjacency: torch.Tensor,
    device: torch.device,
) -> nn.Module:
    model = net.net_gcn(
        embedding_dim=args["embedding_dim"],
        adj=adjacency,
        dropout=args["dropout"],
        input_dropout=args["input_dropout"],
        pre_linear=bool(args["pre_linear"]),
        residual=bool(args["residual"]),
        layer_norm=bool(args["layer_norm"]),
        batch_norm=bool(args["batch_norm"]),
        jumping_knowledge=bool(args["jumping_knowledge"]),
        tuned_backbone=True,
    )
    return model.to(device)


def _load_tensors(args: dict[str, Any], device: torch.device):
    tensors = load_data(args["dataset"], args["data_root"])
    return tuple(tensor.to(device) for tensor in tensors)


def _evaluate(
    model: nn.Module,
    features: torch.Tensor,
    adjacency: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    metric: str,
    *,
    full_graph: bool = False,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(
            features,
            adjacency,
            val_test=True,
            full_graph=full_graph,
        )
    prediction = logits[indices].argmax(dim=-1)
    truth = labels[indices]
    if metric == "rocauc":
        truth_numpy = truth.detach().cpu().numpy()
        probabilities = torch.softmax(
            logits[indices],
            dim=-1,
        ).detach().cpu().numpy()
        if probabilities.shape[1] == 2:
            score = float(
                roc_auc_score(truth_numpy, probabilities[:, 1])
            )
        else:
            score = float(
                roc_auc_score(
                    truth_numpy,
                    probabilities,
                    multi_class="ovr",
                )
            )
    else:
        score = float((prediction == truth).float().mean().item())
    macro_f1 = float(
        f1_score(
            truth.detach().cpu().numpy(),
            prediction.detach().cpu().numpy(),
            average="macro",
            zero_division=0,
        )
    )
    return score, macro_f1


def _apply_ticket_masks(
    rewind_state: dict[str, torch.Tensor],
    mask_dict: dict[str, Any],
) -> dict[str, torch.Tensor]:
    ticket_state = copy.deepcopy(rewind_state)
    adjacency_mask = mask_dict["adj_mask"]
    ticket_state["adj_mask1_train"] = adjacency_mask.clone()
    ticket_state["adj_mask2_fixed"] = adjacency_mask.clone()
    return ticket_state


def _discover_ticket(
    args: dict[str, Any],
    tensors,
    device: torch.device,
    starting_state: dict[str, torch.Tensor] | None,
    adjacency_keep_count: int,
    budget: RunTimeBudget | None = None,
    run_number: int = 1,
):
    adjacency, features, labels, idx_train, idx_val, _idx_test = tensors
    model = _build_model(args, adjacency, device)
    if starting_state is not None:
        model.load_state_dict(starting_state)
    pruning.soft_edge_mask_init(
        model,
        args["init_soft_mask_type"],
        args["seed"],
    )
    rewind_state = copy.deepcopy(model.state_dict())

    mask_parameters = [model.adj_mask1_train]
    mask_parameter_ids = {id(parameter) for parameter in mask_parameters}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in mask_parameter_ids and parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(
        [
            {
                "params": backbone_parameters,
                "weight_decay": args["weight_decay"],
            },
            {
                "params": mask_parameters,
                "weight_decay": 0.0,
            },
        ],
        lr=args["lr"],
    )
    loss_function = nn.CrossEntropyLoss()
    best_validation = -1.0
    best_mask = None

    for epoch in range(args["mask_epoch"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, adjacency)
        loss = loss_function(logits[idx_train], labels[idx_train])
        loss.backward()
        pruning.subgradient_update_edge_mask(model, args)
        optimizer.step()

        validation_accuracy, _validation_f1 = _evaluate(
            model,
            features,
            adjacency,
            labels,
            idx_val,
            args["metric"],
        )
        if validation_accuracy > best_validation:
            best_validation = validation_accuracy
            best_mask = pruning.get_final_edge_mask_epoch(
                model,
                adj_keep_count=adjacency_keep_count,
            )
        print(
            "[Mask Search] "
            f"epoch={epoch} loss={loss.item():.6f} "
            f"val={100.0 * validation_accuracy:.2f} "
            f"best_val={100.0 * best_validation:.2f}",
            flush=True,
        )
        if budget is not None and budget.exhausted(
            run_number,
            f"mask:{epoch + 1}",
            f"mask:{args['mask_epoch']}+fixed:{args['total_epoch']}",
        ):
            break

    if best_mask is None:
        raise RuntimeError("Unified-LTH mask discovery produced no ticket")
    return _apply_ticket_masks(rewind_state, best_mask)


def _train_fixed_ticket(
    args: dict[str, Any],
    tensors,
    device: torch.device,
    ticket_state: dict[str, torch.Tensor],
    budget: RunTimeBudget | None = None,
    run_number: int = 1,
) -> dict[str, float | int]:
    adjacency, features, labels, idx_train, idx_val, idx_test = tensors
    model = _build_model(args, adjacency, device)
    model.load_state_dict(ticket_state)
    model.adj_mask1_train.requires_grad = False

    adjacency_kept_percent = pruning.print_edge_sparsity(model)
    weight_kept_percent = 100.0
    optimizer = torch.optim.Adam(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=args["lr"],
        weight_decay=args["weight_decay"],
    )
    loss_function = nn.CrossEntropyLoss()
    best = {
        "valid_acc": -1.0,
        "test_acc": 0.0,
        "train_acc": 0.0,
        "train_f1": 0.0,
        "test_f1": 0.0,
        "epoch": 0,
    }
    full_graph_eval = args["eval_graph"] == "original"
    if full_graph_eval:
        print(
            "[EvaluationGraph] topology=original-full "
            "training_topology=fixed-sparse-ticket",
            flush=True,
        )

    for epoch in range(args["total_epoch"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, adjacency)
        loss = loss_function(logits[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        valid_acc, _valid_f1 = _evaluate(
            model, features, adjacency, labels, idx_val, args["metric"],
            full_graph=full_graph_eval,
        )
        test_acc, test_f1 = _evaluate(
            model, features, adjacency, labels, idx_test, args["metric"],
            full_graph=full_graph_eval,
        )
        train_acc, train_f1 = _evaluate(
            model, features, adjacency, labels, idx_train, args["metric"],
            full_graph=full_graph_eval,
        )
        if valid_acc > best["valid_acc"]:
            best.update(
                {
                    "valid_acc": valid_acc,
                    "test_acc": test_acc,
                    "train_acc": train_acc,
                    "train_f1": train_f1,
                    "test_f1": test_f1,
                    "epoch": epoch,
                }
            )
        print(
            "[Fixed Ticket] "
            f"epoch={epoch} loss={loss.item():.6f} "
            f"val={100.0 * valid_acc:.2f} "
            f"test={100.0 * test_acc:.2f} "
            f"best_val={100.0 * best['valid_acc']:.2f} "
            f"best_test={100.0 * best['test_acc']:.2f} "
            f"best_epoch={best['epoch']}",
            flush=True,
        )
        if budget is not None and budget.exhausted(
            run_number,
            f"fixed:{epoch + 1}",
            f"mask:{args['mask_epoch']}+fixed:{args['total_epoch']}",
        ):
            break

    best["adjacency_kept_percent"] = adjacency_kept_percent
    best["weight_kept_percent"] = weight_kept_percent
    return best


def _scheduled_keep_count(
    original_count: int,
    target_ratio: float,
    round_index: int,
    rounds: int,
) -> int:
    if round_index == rounds:
        return max(1, min(original_count, round(original_count * target_ratio)))
    intermediate_ratio = target_ratio ** (round_index / rounds)
    return max(
        1,
        min(original_count, round(original_count * intermediate_ratio)),
    )


def _dense_backbone_parameter_count(args: dict[str, Any]) -> int:
    """Count the always-retained tunedGNN model parameters without building it."""

    layers = int(args["num_layers"])
    hidden = int(args["hidden_channels"])
    input_channels = int(args["embedding_dim"][0])
    output_channels = int(args["embedding_dim"][-1])
    if args["pre_linear"]:
        message_dims = [hidden] * (layers + 1)
    else:
        message_dims = [input_channels] + [hidden] * layers
    transforms = sum(
        message_dims[index] * message_dims[index + 1]
        + message_dims[index + 1]
        for index in range(layers)
    )
    # GCN transforms and tunedGNN's residual linears have the same shapes;
    # layer norm and batch norm are both present in the original module even
    # when their corresponding flags are disabled.
    normalizations = 4 * sum(message_dims[1:])
    pre_linear = (
        input_channels * hidden + hidden if args["pre_linear"] else 0
    )
    predictor = hidden * output_channels + output_channels
    return 2 * transforms + normalizations + pre_linear + predictor


def run_protocol(args: dict[str, Any]) -> None:
    device = _device(args["device"])
    print(
        "[PruningMode] "
        + (
            "edge_only=true weight_masks=disabled model_weights_kept=100%"
            if args["weight_kept_ratio"] == 1.0
            else "edge_only=false joint_weight_pruning=true"
        ),
        flush=True,
    )
    base_dim = utils.infer_embedding_dim(
        args["data_root"],
        args["dataset"],
    )
    args["embedding_dim"] = (
        [base_dim[0]]
        + [args["hidden_channels"]] * args["num_layers"]
        + [base_dim[-1]]
    )

    # Match tunedGNN: seed once, then let consecutive reset/initialization
    # calls produce distinct runs instead of replaying one identical run.
    pruning.setup_seed(args["seed"])
    for run in range(args["runs"]):
        os.environ["SCAFFOLD_SPLIT_RUN"] = str(run)
        print(
            f"[TunedGNNProtocol] run={run + 1}/{args['runs']} "
            f"seed={args['seed']}",
            flush=True,
        )
        tensors = _load_tensors(args, device)
        budget = RunTimeBudget().start()
        original_adjacency_count = int(
            torch.count_nonzero(tensors[0]).item()
        )
        original_weight_count = _dense_backbone_parameter_count(args)

        ticket_state = None
        final_result = None
        for round_index in range(1, args["prune_rounds"] + 1):
            adjacency_keep_count = _scheduled_keep_count(
                original_adjacency_count,
                args["target_kept_ratio"],
                round_index,
                args["prune_rounds"],
            )
            weight_keep_count = original_weight_count
            print(
                "[PruningRound] "
                f"round={round_index}/{args['prune_rounds']} "
                f"adjacency_keep={adjacency_keep_count}/"
                f"{original_adjacency_count} "
                f"weight_keep={weight_keep_count}/"
                f"{original_weight_count}",
                flush=True,
            )
            ticket_state = _discover_ticket(
                args,
                tensors,
                device,
                ticket_state,
                adjacency_keep_count,
                budget=budget,
                run_number=run + 1,
            )
            final_result = _train_fixed_ticket(
                args,
                tensors,
                device,
                ticket_state,
                budget=budget,
                run_number=run + 1,
            )
            print(
                "[TargetRatio] "
                f"requested_kept={args['target_kept_ratio']:.8f} "
                f"round={round_index}/{args['prune_rounds']} "
                "achieved_kept="
                f"{final_result['adjacency_kept_percent'] / 100.0:.8f} "
                f"weight_kept="
                f"{final_result['weight_kept_percent'] / 100.0:.8f}",
                flush=True,
            )

        if final_result is None:
            raise RuntimeError("Unified-LTH produced no fixed ticket")
        append_baseline_result(
            method="unified_lth",
            dataset=args["dataset"],
            run=run + 1,
            seed=args["seed"],
            epochs=args["total_epoch"],
            kept_ratio=args["target_kept_ratio"],
            sparsity=100.0 - final_result["adjacency_kept_percent"],
            train_acc=100.0 * final_result["train_acc"],
            valid_acc=100.0 * final_result["valid_acc"],
            test_acc=100.0 * final_result["test_acc"],
            train_f1_macro=100.0 * final_result["train_f1"],
            test_f1_macro=100.0 * final_result["test_f1"],
            chosen_epoch=final_result["epoch"],
        )


def parser_loader() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified-LTH GCN IMP")
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--data_root", default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--total_epoch", "--total-epochs", type=int, default=200)
    parser.add_argument("--mask_epoch", "--mask-epochs", type=int, default=200)
    parser.add_argument("--prune_rounds", "--prune-rounds", type=int, default=2)
    parser.add_argument(
        "--target_kept_ratio",
        "--target-kept-ratio",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--weight_kept_ratio",
        "--weight-kept-ratio",
        type=float,
        default=1.0,
    )
    parser.add_argument("--s1", type=float, default=1e-2)
    parser.add_argument("--s2", type=float, default=1e-2)
    parser.add_argument(
        "--init_soft_mask_type",
        choices=("all_one", "kaiming", "normal", "uniform"),
        default="all_one",
    )
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden_channels", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--input_dropout", type=float, default=0.0)
    parser.add_argument(
        "--metric",
        choices=("acc", "rocauc"),
        default="acc",
    )
    parser.add_argument(
        "--eval_graph",
        "--eval-graph",
        choices=("sparse", "original"),
        default=os.environ.get("BASELINE_EVAL_GRAPH", "sparse"),
        help="Topology used for fixed-ticket validation/test inference.",
    )
    parser.add_argument("--pre_linear", type=int, choices=(0, 1), default=0)
    parser.add_argument("--residual", type=int, choices=(0, 1), default=0)
    parser.add_argument("--layer_norm", type=int, choices=(0, 1), default=0)
    parser.add_argument("--batch_norm", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--jumping_knowledge",
        type=int,
        choices=(0, 1),
        default=0,
    )
    return parser


def _validate_args(args: dict[str, Any]) -> None:
    for key in ("target_kept_ratio", "weight_kept_ratio"):
        if not 0.0 < args[key] <= 1.0:
            raise ValueError(f"{key} must be in (0, 1]")
    for key in ("total_epoch", "mask_epoch", "prune_rounds", "runs"):
        if args[key] < 1:
            raise ValueError(f"{key} must be positive")
    if args["num_layers"] < 2:
        raise ValueError("num_layers must be at least 2")
    if args["weight_kept_ratio"] != 1.0:
        raise ValueError(
            "Unified-LTH is edge-only in this benchmark; "
            "--weight-kept-ratio must be 1.0"
        )


if __name__ == "__main__":
    parsed_args = vars(parser_loader().parse_args())
    _validate_args(parsed_args)
    print(parsed_args, flush=True)
    run_protocol(parsed_args)
