"""Shared result helpers for the related-method baseline wrappers."""

from __future__ import annotations

import csv
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score


FIELDNAMES = [
    "method",
    "dataset",
    "run",
    "seed",
    "epochs",
    "kept_ratio",
    "sparsity",
    "status",
    "metric",
    "train_acc",
    "valid_acc",
    "test_acc",
    "train_f1_macro",
    "test_f1_macro",
    "chosen_epoch",
    "log_file",
    "raw_results",
    "initial_sparsification_time_sec",
    "resparsification_time_sec",
    "sparsification_time_sec",
    "training_time_sec",
    "total_runtime_sec",
    # training_time_sec is the run wall minus FOREGROUND sparsification, so it
    # still contains evaluation and support-selection work. These three split
    # it: train_only + eval + selection ~= training_time_sec.
    "train_only_time_sec",
    "eval_time_sec",
    "selection_time_sec",
]


def macro_f1_percent(labels: Any, logits_or_pred: Any) -> float:
    labels_np = _to_numpy(labels).reshape(-1)
    pred = _to_prediction(logits_or_pred)
    pred_np = _to_numpy(pred).reshape(-1)
    valid = labels_np >= 0
    if valid.sum() == 0:
        return float("nan")
    return 100.0 * float(f1_score(labels_np[valid], pred_np[valid], average="macro", zero_division=0))


class RunTimeBudget:
    """Per-run wall-clock budget shared by every baseline training loop.

    The batch runner used to enforce its limit from the outside, by killing the
    process.  That produced TIMEOUT_PARTIAL cases with no recorded result, and
    for the internal-protocol methods it also meant runs 2 and 3 never started:
    one run consumed the whole invocation.  Stopping from the inside instead
    lets a run end on a completed epoch, report the best epoch it reached, and
    hand the remaining budget to the next run.

    Seconds come from BASELINE_RUN_TIME_BUDGET_SECONDS, which the runner exports
    from --time-limit-seconds.  Zero, unset or unparsable means no budget, so
    every caller keeps its previous behaviour by default.
    """

    ENV_VAR = "BASELINE_RUN_TIME_BUDGET_SECONDS"

    def __init__(self, seconds: Any = None) -> None:
        if seconds is None:
            seconds = os.environ.get(self.ENV_VAR, "")
        try:
            self.seconds = float(seconds)
        except (TypeError, ValueError):
            self.seconds = 0.0
        if not math.isfinite(self.seconds) or self.seconds < 0.0:
            self.seconds = 0.0
        self.stopped_early = False
        self._started = time.monotonic()

    @property
    def enabled(self) -> bool:
        return self.seconds > 0.0

    def start(self) -> "RunTimeBudget":
        """Restart the clock; call once at the top of each run."""

        self._started = time.monotonic()
        self.stopped_early = False
        return self

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    @property
    def expired(self) -> bool:
        return self.enabled and self.elapsed >= self.seconds

    def exhausted(self, run: Any, epoch: Any, total_epochs: Any) -> bool:
        """True when this run must stop now; logs the reason exactly once.

        Call at the end of an epoch, after the metrics for that epoch have been
        folded into best-so-far selection, so the reported result always
        describes a fully completed epoch.
        """

        if not self.expired:
            return False
        if not self.stopped_early:
            self.stopped_early = True
            print(
                f"[RunBudget] run {run} stopped after epoch {epoch} of "
                f"{total_epochs}: {self.elapsed:.0f}s of {self.seconds:.0f}s "
                f"budget used; reporting the best epoch reached so far",
                flush=True,
            )
        return True


def accuracy_percent(labels: Any, logits_or_pred: Any) -> float:
    labels_np = _to_numpy(labels).reshape(-1)
    pred_np = _to_numpy(_to_prediction(logits_or_pred)).reshape(-1)
    valid = labels_np >= 0
    if valid.sum() == 0:
        return float("nan")
    return 100.0 * float((labels_np[valid] == pred_np[valid]).mean())


def single_label_roc_auc_percent(labels: Any, logits: Any) -> float:
    """Return single-label ROC-AUC in percent from raw class logits.

    Minesweeper and Questions are scored by ROC-AUC rather than accuracy: both
    are imbalanced binary tasks whose accuracy sits at the majority-class rate
    (80% and 97%) regardless of what the model learns.  This mirrors
    ``utils.data_utils.eval_rocauc``, which is what tunedGNN, Benchmark and
    Scaffold report for those datasets, so baseline cells stay comparable.
    """

    labels_np = _to_numpy(labels).reshape(-1)
    scores_np = _to_numpy(logits)
    if scores_np.ndim == 1:
        scores_np = scores_np.reshape(-1, 1)
    valid = (
        np.isfinite(labels_np)
        & (labels_np >= 0)
        & np.isfinite(scores_np).all(axis=1)
    )
    if not valid.any():
        return float("nan")
    truth = labels_np[valid].astype(np.int64)
    scores = scores_np[valid]
    classes = np.unique(truth)
    if classes.size < 2:
        return float("nan")
    if scores.shape[1] == 1:
        return 100.0 * float(roc_auc_score(truth, scores[:, 0]))
    if scores.shape[1] == 2:
        # Ranking by the positive-class logit margin is monotone in the softmax
        # probability of class 1, so this is the same ROC-AUC without needing to
        # normalize first.
        return 100.0 * float(roc_auc_score(truth, scores[:, 1] - scores[:, 0]))

    probabilities = _softmax(scores)
    class_scores = []
    for class_id in classes:
        column = int(class_id)
        if column >= probabilities.shape[1]:
            continue
        one_versus_rest = (truth == class_id).astype(np.int64)
        positives = int(one_versus_rest.sum())
        if positives == 0 or positives == one_versus_rest.size:
            continue
        class_scores.append(
            float(roc_auc_score(one_versus_rest, probabilities[:, column]))
        )
    if not class_scores:
        return float("nan")
    return 100.0 * float(np.mean(class_scores))


def single_label_metric_percent(labels: Any, logits: Any, metric: Any) -> float:
    """Return the campaign's primary single-label metric for one split.

    Every baseline wrapper receives ``--metric`` from the shared tunedGNN
    preset contract.  Routing that flag through here is what keeps a column
    from silently reporting accuracy where the dataset is scored by ROC-AUC.
    """

    if str(metric).lower() == "rocauc":
        return single_label_roc_auc_percent(labels, logits)
    return accuracy_percent(labels, logits)


def multilabel_roc_auc_f1_percent(
    labels: Any,
    logits: Any,
) -> tuple[float, float]:
    """Return OGB-style mean task ROC-AUC and thresholded macro-F1.

    OGBN-Proteins is a 112-task multi-label problem, not a 112-class
    classification problem.  Tasks without both positive and negative
    examples in the selected split are excluded from the ROC-AUC mean, as in
    OGB's evaluator.  Missing labels (NaN or negative) are ignored.
    """

    labels_np = _to_numpy(labels)
    logits_np = _to_numpy(logits)
    if labels_np.ndim != 2 or logits_np.ndim != 2:
        raise ValueError("multilabel metrics require two-dimensional labels and logits")
    if labels_np.shape != logits_np.shape:
        raise ValueError(
            f"label/logit shape mismatch: {labels_np.shape} != {logits_np.shape}"
        )

    aucs = []
    binary_truth = []
    binary_prediction = []
    for task in range(labels_np.shape[1]):
        valid = (
            np.isfinite(labels_np[:, task])
            & (labels_np[:, task] >= 0)
            & np.isfinite(logits_np[:, task])
        )
        truth = labels_np[valid, task]
        if truth.size == 0 or np.unique(truth).size < 2:
            continue
        scores = logits_np[valid, task]
        aucs.append(float(roc_auc_score(truth, scores)))
        binary_truth.append(truth.astype(np.int64))
        binary_prediction.append((scores >= 0.0).astype(np.int64))

    if not aucs:
        return float("nan"), float("nan")
    macro_f1 = f1_score(
        np.concatenate(binary_truth),
        np.concatenate(binary_prediction),
        average="macro",
        zero_division=0,
    )
    return 100.0 * float(np.mean(aucs)), 100.0 * float(macro_f1)


def append_baseline_result(
    *,
    method: Optional[str] = None,
    dataset: Optional[str] = None,
    run: Optional[int] = None,
    seed: Optional[int] = None,
    epochs: Optional[int] = None,
    kept_ratio: Optional[float] = None,
    sparsity: Optional[float] = None,
    status: str = "PASS",
    metric: Optional[str] = None,
    train_acc: Optional[float] = None,
    valid_acc: Optional[float] = None,
    test_acc: Optional[float] = None,
    train_f1_macro: Optional[float] = None,
    test_f1_macro: Optional[float] = None,
    chosen_epoch: Optional[int] = None,
    log_file: Optional[str] = None,
    raw_results: Optional[str] = None,
    initial_sparsification_time_sec: Optional[float] = None,
    resparsification_time_sec: Optional[float] = None,
    sparsification_time_sec: Optional[float] = None,
    training_time_sec: Optional[float] = None,
    total_runtime_sec: Optional[float] = None,
    train_only_time_sec: Optional[float] = None,
    eval_time_sec: Optional[float] = None,
    selection_time_sec: Optional[float] = None,
) -> None:
    csv_path = os.environ.get("BASELINE_INDIVIDUAL_RUNS_CSV")
    if not csv_path:
        return

    row: Dict[str, Any] = {
        "method": method or os.environ.get("BASELINE_METHOD", ""),
        "dataset": dataset or os.environ.get("BASELINE_DATASET", ""),
        "run": run if run is not None else os.environ.get("BASELINE_RUN_ID", ""),
        "seed": seed if seed is not None else os.environ.get("BASELINE_SEED", ""),
        "epochs": epochs if epochs is not None else os.environ.get("BASELINE_EPOCHS", ""),
        "kept_ratio": os.environ.get("BASELINE_KEPT_RATIO", kept_ratio if kept_ratio is not None else ""),
        "sparsity": sparsity if sparsity is not None else "",
        "status": status,
        # Which quantity the *_acc columns actually hold. Recorded so a mixed
        # accuracy/ROC-AUC table can be detected instead of silently assembled.
        "metric": metric or os.environ.get("BASELINE_METRIC", ""),
        "train_acc": _fmt(train_acc),
        "valid_acc": _fmt(valid_acc),
        "test_acc": _fmt(test_acc),
        "train_f1_macro": _fmt(train_f1_macro),
        "test_f1_macro": _fmt(test_f1_macro),
        "chosen_epoch": chosen_epoch if chosen_epoch is not None else "",
        "log_file": log_file or os.environ.get("BASELINE_LOG_FILE", ""),
        "raw_results": raw_results or os.environ.get("RESULTS_ROOT", ""),
        "initial_sparsification_time_sec": _fmt(initial_sparsification_time_sec),
        "resparsification_time_sec": _fmt(resparsification_time_sec),
        "sparsification_time_sec": _fmt(sparsification_time_sec),
        "training_time_sec": _fmt(training_time_sec),
        "total_runtime_sec": _fmt(total_runtime_sec),
        "train_only_time_sec": _fmt(train_only_time_sec),
        "eval_time_sec": _fmt(eval_time_sec),
        "selection_time_sec": _fmt(selection_time_sec),
    }

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(FIELDNAMES)
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = list(reader.fieldnames or ())
            existing_rows = list(reader)
        merged_fieldnames = existing_fieldnames + [
            field for field in fieldnames if field not in existing_fieldnames
        ]
        if merged_fieldnames != existing_fieldnames:
            with tempfile.NamedTemporaryFile(
                "w",
                newline="",
                dir=path.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                writer = csv.DictWriter(handle, fieldnames=merged_fieldnames)
                writer.writeheader()
                writer.writerows(existing_rows)
            os.replace(temporary, path)
        fieldnames = merged_fieldnames
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_prediction(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.dim() > 1:
            return value.argmax(dim=-1)
        return value
    arr = np.asarray(value)
    if arr.ndim > 1:
        return arr.argmax(axis=-1)
    return arr


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    try:
        parsed = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(parsed):
        return ""
    return f"{parsed:.6f}"
