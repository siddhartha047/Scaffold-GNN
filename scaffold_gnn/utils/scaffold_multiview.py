"""Final-only, same-model Scaffold 1/3/5/full evaluation and durable records."""

from __future__ import annotations

import csv
import fcntl
import gc
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch_geometric.utils import coalesce

from scaffold_gnn.utils.eval import evaluate
from scaffold_gnn.utils.data_utils import eval_acc, eval_f1_macro
from scaffold_gnn.utils.scaffold_checkpoint import checkpoint_selection_policy


FIELDS = (
    "dataset", "metric", "target_ratio", "method", "seed", "run",
    "epochs_requested", "epochs_completed", "model_state_sha256",
    "split_fingerprint", "view", "k_requested", "k_effective",
    "selected_epochs", "validation_scores", "directed_edges",
    "canonical_retained_edges", "achieved_ratio", "train_metric",
    "valid_metric", "test_metric", "valid_loss", "train_f1", "test_f1",
    "inference_time_sec", "sparsification_time_sec", "training_time_sec",
    "total_runtime_sec", "stopped_early", "status", "error",
    "selection_policy", "graph_bank_size", "historical_validation_scores",
    "candidate_epochs", "candidate_validation_scores", "graph_selection_time_sec",
    "model_policy", "model_epoch", "last_model_state_sha256", "model_checkpoint",
    "test_accuracy", "highest_test_metric", "highest_test_metric_epoch",
    "highest_test_accuracy", "highest_test_accuracy_epoch", "best_validation_model_epoch",
    "effective_seed",
    "checkpoint_validation_metric",
    "budget_q", "budgeted_union_metadata", "budgeted_union_artifact",
    "prediction_reducer", "forward_passes", "per_graph_retained_edges", "total_edge_visits",
    "sample_weight_mode", "sample_mix_alpha",
    "epoch_count_policy",
)


def validate_final_views(args):
    views = tuple(getattr(args, "scaffold_final_eval_views", None) or ())
    if not views:
        if (getattr(args, 'scaffold_final_reselect_best1', False)
                or getattr(args, 'scaffold_final_report_historical1', False)
                or getattr(args, 'scaffold_final_report_best_checkpoint', False)
                or getattr(args, 'scaffold_final_best_single_only', False)
                or getattr(args, 'scaffold_final_best_full_checkpoint', False)
                or getattr(args, 'scaffold_final_best_ensemble_views', None)
                or getattr(args, 'scaffold_final_budgeted_union', False)
                or getattr(args, 'scaffold_final_prediction_ensembles', False)
                or getattr(args, 'scaffold_final_skip_last_controls', False)
                or getattr(args, 'scaffold_protocol_run_ids', None)
                or getattr(args, 'scaffold_final_graph_bank_size', None) is not None):
            raise ValueError('final graph-bank/re-selection options require --scaffold_final_eval_views')
        return views
    if args.sparsifier not in {"scaffold_sample", "scaffold_fast", "scaffold_batch"}:
        raise ValueError("final multiview evaluation supports Scaffold Sample/Fast/Batch only")
    if getattr(args, "scaffold_consensus_mode", "off") != "off":
        raise ValueError("final multiview evaluation does not enable Consensus")
    if args.eval_graph != "sparse" or args.scaffold_eval_ensemble_source != "best":
        raise ValueError("final multiview requires --eval_graph sparse and source=best")
    if args.scaffold_eval_ensemble_size != 1 or not args.report_final_epoch_only:
        raise ValueError("use ensemble_size=1 and --report_final_epoch_only for multiview")
    if args.eval_step <= 0:
        raise ValueError("positive eval_step is needed to rank intermediate training supports")
    if not getattr(args, 'scaffold_final_eval_csv', None) and not os.environ.get('SCAFFOLD_MULTIVIEW_CSV'):
        raise ValueError('provide --scaffold_final_eval_csv before starting multiview training')
    if len(set(views)) != len(views) or any(v not in {"1", "3", "5", "10", "all"} for v in views):
        raise ValueError("final views must be unique entries from 1,3,5,10,all")
    if getattr(args, 'scaffold_final_reselect_best1', False) and '1' not in views:
        raise ValueError('final best-1 re-selection requires view 1')
    if (getattr(args, 'scaffold_final_report_historical1', False)
            and not getattr(args, 'scaffold_final_reselect_best1', False)):
        raise ValueError('historical-best-1 control requires final best-1 re-selection')
    if getattr(args, 'scaffold_final_report_best_checkpoint', False):
        if '1' not in views:
            raise ValueError('best-checkpoint comparison requires final view 1')
        if getattr(args, 'scaffold_final_reselect_best1', False):
            raise ValueError('best-checkpoint pilot keeps historical graph selection; do not enable final re-selection')
    if (getattr(args, 'scaffold_final_best_single_only', False)
            and not getattr(args, 'scaffold_final_report_best_checkpoint', False)):
        raise ValueError('best-only single-graph inference requires best-checkpoint reporting')
    if getattr(args, 'scaffold_final_best_full_checkpoint', False):
        if not getattr(args, 'scaffold_final_report_best_checkpoint', False) or 'all' not in views:
            raise ValueError('best full-graph checkpoint requires best-checkpoint reporting and final view all')
    best_ensemble_views = tuple(getattr(args, 'scaffold_final_best_ensemble_views', None) or ())
    if best_ensemble_views:
        if not getattr(args, 'scaffold_final_report_best_checkpoint', False):
            raise ValueError('best-ensemble checkpoints require best-checkpoint reporting')
        if len(set(best_ensemble_views)) != len(best_ensemble_views) or any(v not in {'3', '5', '10'} or v not in views for v in best_ensemble_views):
            raise ValueError('best-ensemble views must be unique requested sparse views from 3,5,10')
    if getattr(args, 'scaffold_final_skip_last_controls', False):
        if (not getattr(args, 'scaffold_final_best_single_only', False)
                or not getattr(args, 'scaffold_final_best_full_checkpoint', False)
                or any(v not in best_ensemble_views for v in views if v not in ('1', 'all'))):
            raise ValueError('skipping last controls requires a validation-best tracker for every requested view')
    if getattr(args, 'scaffold_final_prediction_ensembles', False):
        from scaffold_gnn.utils.scaffold_prediction_ensemble import prediction_views
        predictors = prediction_views(args)
        if ({str(k) for k, _ in predictors.values()} != set(best_ensemble_views)
                or not getattr(args, 'scaffold_final_best_single_only', False)
                or not getattr(args, 'scaffold_final_best_full_checkpoint', False)):
            raise ValueError('prediction ensembles require matching validation-best single, union-k and full controls')
        if args.dataset != 'questions' and any(mode == 'majority-vote' for _, mode in predictors.values()):
            raise ValueError('majority voting is supported for binary Questions only')
        if getattr(args, 'scaffold_final_budgeted_union', False):
            raise ValueError('use separate campaigns for prediction ensembles and budgeted pruning')
        ratio = getattr(args, 'target_ratio', None)
        if ratio is None or not math.isfinite(ratio) or not 0. < ratio <= 1.:
            raise ValueError('prediction pilot requires an explicit target_ratio in (0, 1]')
    if getattr(args, 'scaffold_final_budgeted_union', False):
        if not best_ensemble_views:
            raise ValueError('budgeted union requires validation-best ensemble checkpoints')
        if getattr(args, 'joint_node_beta', 0.) != 0.:
            raise ValueError('budgeted union requires node congestion disabled')
        ratio = getattr(args, 'target_ratio', None)
        if ratio is None or not math.isfinite(ratio) or not 0. < ratio <= 1.:
            raise ValueError('budgeted union requires an explicit target_ratio in (0, 1]')
    run_ids = getattr(args, 'scaffold_protocol_run_ids', None)
    if run_ids is not None:
        if not getattr(args, 'scaffold_final_report_best_checkpoint', False):
            raise ValueError('distributed run IDs require the best-checkpoint pilot')
        if not run_ids or len(set(run_ids)) != len(run_ids) or any(r < 1 or r > args.runs for r in run_ids):
            raise ValueError('distributed run IDs must be unique and within 1..runs')
    final_graph_bank_capacity(args, views)
    return views


def final_graph_bank_capacity(args, views):
    minimum = max((int(v) for v in views if v != 'all'), default=1)
    configured = getattr(args, 'scaffold_final_graph_bank_size', None)
    if configured is None:
        return max(minimum, 5 if getattr(args, 'scaffold_final_reselect_best1', False) else 1)
    if configured < minimum or configured < 1:
        raise ValueError('final graph bank must hold at least every requested sparse view')
    return int(configured)


def final_graph_views(bank, full_edge_index, requested, num_nodes):
    """Yield nested validation-ranked supports; never construct Consensus."""
    ranked = sorted(bank, key=lambda x: (x["validation_score"], x["epoch"]), reverse=True)
    for view in requested:
        if view == "all":
            yield view, full_edge_index, None, (), "OK"
            continue
        k = int(view)
        selected = ranked[:k]
        if len(selected) < k:
            # In particular an async run may produce <5 distinct graphs before
            # its budget expires. Do not silently label a smaller union as k=5.
            yield view, None, None, selected, "INSUFFICIENT_GRAPHS"
            continue
        if k == 1:
            edges, weights = selected[0]["view"]
        else:
            edges = coalesce(torch.cat([x["view"][0].cpu() for x in selected], dim=1), num_nodes=num_nodes)
            weights = None  # Same unweighted UNION policy as the existing k=5 runs.
        yield view, edges, weights, selected, "OK"


def evaluation_graph_on_device(node_feat, edge_index, edge_weight=None):
    """Colocate an inference view with features without mutating its CPU bank.

    Original full-graph views and saved supports may be stored on CPU even
    when the current model is training on CUDA. Apply this to both full and
    union views before their intermediate validation/test forward passes.
    """
    target = node_feat.device
    return (edge_index.to(target),
            None if edge_weight is None else edge_weight.to(target))


def upsert_record(path, record):
    """Persist each final view immediately, without losing earlier completed runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    key_fields = ("dataset", "method", "target_ratio", "seed", "run", "view")
    key = tuple(str(record.get(k, "")) for k in key_fields)
    with Path(str(path) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        if path.is_file():
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        rows = [r for r in rows if tuple(str(r.get(k, "")) for k in key_fields) != key]
        rows.append(record)
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


@torch.no_grad()
def _consume_final_full_batch(model, dataset, split, edges, weights, args,
                              device, decomposition_context, consume):
    """Exact whole-topology inference, including for neighbor-trained models.

    No RandomNodeLoader partition or finite-neighbor sampler is used here.
    DGL ProteinGNN.forward accepts the full graph through the same call boundary.
    CPU fallback on CUDA OOM preserves every edge and the same model weights.
    """
    original = (dataset.graph["node_feat"], dataset.graph["edge_index"],
                dataset.graph.get("edge_weight"), dataset.label, model.training)
    model_device = next(model.parameters()).device
    original_threads = torch.get_num_threads()

    def attempt(target):
        for module in model.modules():
            for attribute in ('_cached_edge_index', '_cached_adj_t'):
                if hasattr(module, attribute):
                    setattr(module, attribute, None)
        if target.type == 'cpu':
            torch.set_num_threads(max(1, int(getattr(args, 'joint_parallel_workers', original_threads))))
        model.to(target)
        dataset.graph["node_feat"] = original[0].to(target)
        dataset.graph["edge_index"] = edges.to(target)
        if weights is None:
            dataset.graph.pop("edge_weight", None)
        else:
            dataset.graph["edge_weight"] = weights.to(target)
        dataset.label = original[3].to(target)
        split_target = {k: v.to(target) for k, v in split.items()}
        # DGL's fused GraphConv does not expose PyG feature decomposition.
        context = decomposition_context(model, args, target) if any(
            m.__class__.__name__ == "GCNConv" for m in model.modules()
        ) else nullcontext()
        with context:
            model.eval()
            outputs = model(dataset.graph['node_feat'], dataset.graph['edge_index'],
                            dataset.graph.get('edge_weight'))
            return consume(outputs, split_target)

    try:
        target = torch.device("cpu") if getattr(args, "large_eval_cpu", False) else device
        try:
            return attempt(target)
        except RuntimeError as exc:
            if target.type != "cuda" or "out of memory" not in str(exc).lower():
                raise
            print("[ScaffoldMultiView] CUDA OOM: retrying exact full-batch inference on CPU", flush=True)
            dataset.graph["node_feat"] = original[0]
            dataset.graph["edge_index"] = original[1]
            dataset.graph.pop("edge_weight", None)
            dataset.label = original[3]
            gc.collect()
            torch.cuda.empty_cache()
            return attempt(torch.device("cpu"))
    finally:
        dataset.graph["node_feat"], dataset.graph["edge_index"] = original[:2]
        if original[2] is None:
            dataset.graph.pop("edge_weight", None)
        else:
            dataset.graph["edge_weight"] = original[2]
        dataset.label = original[3]
        model.to(model_device)
        model.train(original[4])
        torch.set_num_threads(original_threads)


def evaluate_full_batch_outputs(model, dataset, split, edges, weights, eval_func,
                                criterion, args, device, decomposition_context):
    """Return exact metrics and CPU logits while restoring the training devices."""
    def consume(outputs, split_target):
        result = evaluate(model, dataset, split_target, eval_func, criterion, args,
                          result=outputs)
        return tuple(float(x) for x in result[:4]), outputs.detach().cpu()

    return _consume_final_full_batch(
        model, dataset, split, edges, weights, args, device, decomposition_context, consume)


def evaluate_final_full_batch(model, dataset, split, edges, weights, eval_func,
                              criterion, args, device, decomposition_context):
    metrics, outputs = evaluate_full_batch_outputs(
        model, dataset, split, edges, weights, eval_func, criterion, args, device, decomposition_context)
    labels = dataset.label.detach().cpu()
    indices = {k: v.detach().cpu() for k, v in split.items()}
    f1 = {k: float(eval_f1_macro(labels[indices[k]], outputs[indices[k]]))
          for k in ('train', 'test')}
    if labels.ndim == 1 or labels.size(1) == 1:
        f1['test_accuracy'] = float(eval_acc(labels[indices['test']], outputs[indices['test']]))
    if not all(math.isfinite(v) for v in metrics[:3]):
        raise RuntimeError('final full-batch inference produced non-finite metrics')
    return metrics, f1


def rank_final_sparse_graph_bank(*, model, dataset, split, bank, eval_func,
                                 args, device, decomposition_context):
    """Re-score the current bank with one frozen model, without test metrics.

    Copies bank entries: historical scores and training-time bank membership
    stay unchanged. No new support, graph union, optimization or Consensus.
    """
    ranked = []
    for candidate in sorted(bank, key=lambda x: (x['validation_score'], x['epoch']), reverse=True):
        def consume(outputs, split_target):
            valid = split_target['valid']
            return float(eval_func(dataset.label[valid], outputs[valid]))

        edges, weights = candidate['view']
        score = _consume_final_full_batch(
            model, dataset, split, edges, weights, args, device, decomposition_context, consume)
        if not math.isfinite(score):
            raise ValueError(f'non-finite final validation score for epoch {candidate["epoch"]}')
        ranked.append(dict(candidate, historical_validation_score=candidate['validation_score'],
                           validation_score=score))
        print(f'[ScaffoldFinalReselect] candidate={len(ranked)}/{len(bank)} '
              f'epoch={candidate["epoch"]} historical_valid={candidate["validation_score"]:.8f} '
              f'final_valid={score:.8f}', flush=True)
    ranked.sort(key=lambda x: (x['validation_score'], x['epoch']), reverse=True)
    return ranked


def run_final_views(*, args, model, dataset, split, bank, original_edges,
                    original_edge_count, requested, eval_func, criterion, device,
                    decomposition_context, model_identity, metadata, checkpoint_tracker=None,
                    ensemble_checkpoint_trackers=None, full_checkpoint_tracker=None,
                    prediction_checkpoint_trackers=None):
    csv_path = args.scaffold_final_eval_csv or os.environ.get("SCAFFOLD_MULTIVIEW_CSV")
    if not csv_path:
        raise ValueError("final multiview evaluation requires --scaffold_final_eval_csv")
    identity = model_identity(model)
    best_single_only = bool(getattr(args, 'scaffold_final_best_single_only', False))
    if best_single_only and (checkpoint_tracker is None or not getattr(args, 'scaffold_final_report_best_checkpoint', False)):
        raise ValueError('best-only single-graph inference requires its validation-selected checkpoint tracker')
    best_full_checkpoint = bool(getattr(args, 'scaffold_final_best_full_checkpoint', False))
    if best_full_checkpoint != (full_checkpoint_tracker is not None):
        raise ValueError('best full-graph evaluation requires its own requested checkpoint tracker')
    if getattr(args, 'scaffold_final_report_best_checkpoint', False) and checkpoint_tracker is None:
        raise ValueError('best-checkpoint evaluation requires a captured checkpoint tracker')
    if checkpoint_tracker is not None:
        checkpoint_tracker.save_last(model, metadata['epochs_completed'])
    ensemble_checkpoint_trackers = ensemble_checkpoint_trackers or {}
    expected_ensembles = {int(k) for k in (getattr(args, 'scaffold_final_best_ensemble_views', None) or ())}
    if set(ensemble_checkpoint_trackers) != expected_ensembles:
        raise ValueError('best-ensemble evaluation requires a tracker for each requested k')
    from scaffold_gnn.utils.scaffold_prediction_ensemble import prediction_views, run_prediction_checkpoint_views
    prediction_checkpoint_trackers = prediction_checkpoint_trackers or {}
    expected_predictions = set(prediction_views(args)) if getattr(args, 'scaffold_final_prediction_ensembles', False) else set()
    if set(prediction_checkpoint_trackers) != expected_predictions:
        raise ValueError('prediction evaluation requires its own tracker for each reducer and k')
    records = []
    reselect = bool(getattr(args, 'scaffold_final_reselect_best1', False))
    rescored, selection_error, selection_status = [], '', 'OK'
    selection_time = 0.0
    if reselect:
        started = time.perf_counter()
        try:
            capacity = final_graph_bank_capacity(args, requested)
            if len(bank) < capacity:
                selection_status = 'INSUFFICIENT_GRAPHS'
                selection_error = f'final re-selection requires {capacity} distinct supports; got {len(bank)}'
            else:
                rescored = rank_final_sparse_graph_bank(
                    model=model, dataset=dataset, split=split, bank=bank,
                    eval_func=eval_func, args=args, device=device,
                    decomposition_context=decomposition_context)
        except (RuntimeError, MemoryError, ValueError) as exc:
            selection_status = 'GRAPH_SELECTION_FAILED'
            selection_error = f'{type(exc).__name__}: {exc}'
        selection_time = time.perf_counter() - started
        if model_identity(model) != identity:
            raise RuntimeError('final graph re-selection changed the trained model state')
    # Never run or emit last-model single-graph inference in best-only mode.
    # The saved single checkpoint remains raw view="best" for provenance and
    # the pilot reporter publishes it as Sample-1, without a redundant alias.
    last_model_requested = tuple(v for v in requested if not (
        (best_single_only and v == '1') or (best_full_checkpoint and v == 'all')))
    if getattr(args, 'scaffold_final_skip_last_controls', False):
        last_model_requested = ()
    views = list(final_graph_views(bank, original_edges, last_model_requested, int(dataset.graph['num_nodes'])))
    if getattr(args, 'scaffold_final_report_historical1', False):
        historical = next(final_graph_views(bank, original_edges, ('1',), int(dataset.graph['num_nodes'])))
        views.append(('1-historical', *historical[1:]))
    for view, edges, weights, selected, status in views:
        if reselect and view == '1':
            status = selection_status
            selected = rescored[:1]
            edges, weights = selected[0]['view'] if selected else (None, None)
        record = {k: "" for k in FIELDS}
        record.update(metadata)
        if checkpoint_tracker is not None:
            record.update(checkpoint_tracker.diagnostics(), model_policy='last-epoch',
                          model_epoch=metadata['epochs_completed'], last_model_state_sha256=identity,
                          model_checkpoint=str(checkpoint_tracker.last_path))
        record.update(view=view, k_requested=0 if view == "all" else (1 if view == '1-historical' else int(view)),
                      k_effective=len(selected), status=status,
                      model_state_sha256=identity,
                      selected_epochs=json.dumps([x["epoch"] for x in selected]),
                      validation_scores=json.dumps([x["validation_score"] for x in selected]),
                      historical_validation_scores=json.dumps([x.get('historical_validation_score', x['validation_score']) for x in selected]),
                      selection_policy='full-graph' if view == 'all' else ('final-model-validation' if reselect and view == '1' else 'historical-validation'),
                      graph_bank_size=len(bank),
                      candidate_epochs=json.dumps([x['epoch'] for x in rescored]) if reselect and view == '1' else '',
                      candidate_validation_scores=json.dumps([x['validation_score'] for x in rescored]) if reselect and view == '1' else '',
                      graph_selection_time_sec=selection_time if reselect and view == '1' else 0.0)
        if reselect and view == '1' and selection_error:
            record['error'] = selection_error
        if status == "OK":
            started = time.perf_counter()
            try:
                metrics, f1 = evaluate_final_full_batch(
                    model, dataset, split, edges, weights, eval_func, criterion,
                    args, device, decomposition_context,
                )
                record.update(train_metric=metrics[0], valid_metric=metrics[1], test_metric=metrics[2], valid_loss=metrics[3],
                              train_f1=f1["train"], test_f1=f1["test"], directed_edges=int(edges.size(1)))
                record['test_accuracy'] = f1.get('test_accuracy', '')
                # Finalized graphs are symmetric and unique; exclude self-loops.
                retained = int((edges[0] != edges[1]).sum().item()) // 2
                record.update(canonical_retained_edges=retained,
                              achieved_ratio=retained / original_edge_count if original_edge_count else 0.0)
            except (RuntimeError, MemoryError, ValueError) as exc:
                record.update(status='EVALUATION_FAILED', error=f'{type(exc).__name__}: {exc}')
                print(f'[ScaffoldMultiView] view={view} failed; preserving other views: {exc}', flush=True)
            record['inference_time_sec'] = time.perf_counter() - started
        if model_identity(model) != identity:
            raise RuntimeError('final evaluation unexpectedly changed the trained model state')
        upsert_record(csv_path, record)
        records.append(record)
        print("[ScaffoldMultiView] " + json.dumps(record, sort_keys=True), flush=True)
    saved_views = ([('best', checkpoint_tracker, 1)] if checkpoint_tracker is not None else [])
    saved_views.extend((f'best{k}', tracker, k) for k, tracker in sorted(ensemble_checkpoint_trackers.items()))
    if full_checkpoint_tracker is not None:
        saved_views.append(('all', full_checkpoint_tracker, 0))
    for saved_view, saved_tracker, saved_k in saved_views:
        record = dict.fromkeys(FIELDS, '')
        record.update(metadata)
        record.update(saved_tracker.diagnostics(), view=saved_view, k_requested=saved_k,
                      k_effective=saved_k if saved_tracker.best else 0,
                      status='OK' if saved_tracker.best else ('INSUFFICIENT_GRAPHS' if saved_k > 1 else 'BEST_CHECKPOINT_MISSING'),
                      model_policy='best-validation', last_model_state_sha256=identity,
                      model_checkpoint=str(saved_tracker.best_path),
                      selection_policy=checkpoint_selection_policy(saved_k),
                      graph_bank_size=len(bank), graph_selection_time_sec=0.0)
        if saved_tracker.best is not None:
            started = time.perf_counter()
            try:
                with saved_tracker.use_best(model) as best:
                    best_identity = model_identity(model)
                    edges, weights = best['edge_index'], best['edge_weight']
                    record.update(model_state_sha256=best_identity, model_epoch=best['model_epoch'],
                                  selected_epochs=json.dumps(best['graph_epochs']),
                                  validation_scores=json.dumps([best['valid_metric']] if saved_k == 1 else best['historical_validation_scores']),
                                  historical_validation_scores=json.dumps(best['historical_validation_scores']),
                                  checkpoint_validation_metric=best['valid_metric'])
                    metrics, f1 = evaluate_final_full_batch(
                        model, dataset, split, edges, weights, eval_func, criterion,
                        args, device, decomposition_context)
                    if model_identity(model) != best_identity:
                        raise RuntimeError('best-checkpoint inference changed saved model state')
                    retained = int((edges[0] != edges[1]).sum().item()) // 2
                    record.update(train_metric=metrics[0], valid_metric=metrics[1], test_metric=metrics[2],
                                  valid_loss=metrics[3], train_f1=f1['train'], test_f1=f1['test'],
                                  test_accuracy=f1.get('test_accuracy', ''), directed_edges=int(edges.size(1)),
                                  canonical_retained_edges=retained,
                                  achieved_ratio=retained / original_edge_count if original_edge_count else 0.0)
            except (RuntimeError, MemoryError, ValueError) as exc:
                record.update(status='EVALUATION_FAILED', error=f'{type(exc).__name__}: {exc}')
            record['inference_time_sec'] = time.perf_counter() - started
        if model_identity(model) != identity:
            raise RuntimeError('best-checkpoint evaluation did not restore the last model')
        upsert_record(csv_path, record)
        records.append(record)
        print('[ScaffoldMultiView] ' + json.dumps(record, sort_keys=True), flush=True)
    if getattr(args, 'scaffold_final_budgeted_union', False):
        records.extend(run_budgeted_checkpoint_views(
            args=args, model=model, dataset=dataset, split=split,
            original_edges=original_edges, original_edge_count=original_edge_count,
            trackers=ensemble_checkpoint_trackers, eval_func=eval_func,
            criterion=criterion, device=device, decomposition_context=decomposition_context,
            model_identity=model_identity, metadata=metadata, last_identity=identity,
            csv_path=csv_path))
    if prediction_checkpoint_trackers:
        records.extend(run_prediction_checkpoint_views(
            args=args, model=model, dataset=dataset, split=split,
            trackers=prediction_checkpoint_trackers, original_edge_count=original_edge_count,
            eval_func=eval_func, criterion=criterion, device=device,
            decomposition_context=decomposition_context, model_identity=model_identity,
            metadata=metadata, last_identity=identity, csv_path=csv_path))
    if model_identity(model) != identity:
        raise RuntimeError("final evaluation unexpectedly changed the trained model state")
    return records


def run_budgeted_checkpoint_views(*, args, model, dataset, split, original_edges,
                                  original_edge_count, trackers, eval_func,
                                  criterion, device, decomposition_context,
                                  model_identity, metadata, last_identity, csv_path):
    """Only after training: prune exact saved constituents of each k winner.

    The original unpruned union chose the frozen model epoch. We do not claim
    this new pruned support was validation-selected across training epochs.
    A same-model single-support control isolates graph pruning from model
    selection. Both test metrics are measurements, never selection criteria.
    """
    from scaffold_gnn.sparsifiers.scaffold.budgeted_union import build_budgeted_union
    from scaffold_gnn.sparsifiers.scaffold.consensus import (
        canonical_original_edges, resolve_consensus_budget, save_consensus_artifact,
    )

    num_nodes = int(dataset.graph['num_nodes'])
    original = canonical_original_edges(original_edges, num_nodes)
    if original.edge_count != original_edge_count:
        raise ValueError('budgeted union original edge count does not match training provenance')
    q = resolve_consensus_budget(original_edge_count, delta=metadata['target_ratio'], q=None)
    records = []
    for k, tracker in sorted(trackers.items()):
        for control in (False, True):
            view = f'budget{k}' + ('-single' if control else '')
            record = dict.fromkeys(FIELDS, '')
            record.update(metadata, view=view, status='OK', k_requested=1 if control else k,
                          k_effective=1 if control else k, budget_q=q,
                          model_policy='best-validation-union-checkpoint',
                          selection_policy='same-model-historical-single' if control else 'final-frequency-first-q-pruning',
                          model_checkpoint=str(tracker.best_path), last_model_state_sha256=last_identity,
                          graph_selection_time_sec=0., inference_time_sec=0.)
            if tracker.best is None:
                record.update(status='INSUFFICIENT_GRAPHS', error='no winning k-union checkpoint')
            else:
                try:
                    with tracker.use_best(model) as best:
                        sources = best.get('support_records')
                        if not sources or len(sources) != k:
                            raise ValueError('winning constituent snapshots missing; rerun with --scaffold_final_budgeted_union (do not substitute the later bank)')
                        if [s['epoch'] for s in sources] != best['graph_epochs']:
                            raise ValueError('constituent epochs differ from winning checkpoint')
                        if any(s.get('edge_weight') is not None for s in sources):
                            raise ValueError('budgeted union currently supports unweighted Scaffold only')
                        source_union = canonical_original_edges(
                            torch.cat([s['edge_index'].cpu() for s in sources], dim=1), num_nodes)
                        saved_union = canonical_original_edges(best['edge_index'], num_nodes)
                        if not torch.equal(torch.from_numpy(source_union.sorted_keys),
                                           torch.from_numpy(saved_union.sorted_keys)):
                            raise ValueError('constituents do not reproduce the exact saved winning union')
                        selection_start = time.perf_counter()
                        artifact = Path(csv_path).parent / 'budgeted_union' / f'run_{metadata["run"]:02d}' / f'k_{k}.pt'
                        if control:
                            edges = sources[0]['edge_index']
                            selected_sources = sources[:1]
                        else:
                            graph = build_budgeted_union(
                                original_edges, sources, num_nodes=num_nodes, k=k, q=q,
                                alpha=getattr(args, 'joint_alpha', 1.),
                                edge_beta=getattr(args, 'joint_edge_beta', 1.),
                                node_beta=getattr(args, 'joint_node_beta', 0.),
                                edge_norm_p=getattr(args, 'joint_edge_norm_p', 2.),
                                workers=getattr(args, 'joint_parallel_workers', 1))
                            # No multiplicity message weights; restore tuned
                            # self-loop handling exactly as in the saved union.
                            pairs = graph.undirected_pairs
                            edges = torch.cat((pairs, pairs.flip(0)), dim=1)
                            saved_edges = best['edge_index']
                            loops = saved_edges[:, saved_edges[0] == saved_edges[1]]
                            edges = coalesce(torch.cat((edges, loops), dim=1), num_nodes=num_nodes)
                            selected_sources = sources
                            save_consensus_artifact(graph, artifact, metadata={
                                'dataset': metadata['dataset'], 'run': metadata['run'],
                                'split_fingerprint': metadata['split_fingerprint'],
                                'model_epoch': best['model_epoch'],
                                'model_state_sha256': model_identity(model)})
                            record.update(budgeted_union_artifact=str(artifact),
                                          budgeted_union_metadata=json.dumps(graph.metadata, sort_keys=True))
                        record.update(graph_selection_time_sec=time.perf_counter() - selection_start,
                                      model_state_sha256=model_identity(model), model_epoch=best['model_epoch'],
                                      checkpoint_validation_metric=best['valid_metric'],
                                      best_validation_model_epoch=best['model_epoch'],
                                      selected_epochs=json.dumps([s['epoch'] for s in selected_sources]),
                                      historical_validation_scores=json.dumps([s['validation_score'] for s in selected_sources]))
                        started = time.perf_counter()
                        metrics, f1 = evaluate_final_full_batch(
                            model, dataset, split, edges, None, eval_func, criterion,
                            args, device, decomposition_context)
                        retained = canonical_original_edges(edges, num_nodes).edge_count
                        if retained != q:
                            raise ValueError(f'inference used {retained} canonical edges, expected q={q}')
                        record.update(train_metric=metrics[0], valid_metric=metrics[1], test_metric=metrics[2],
                                      valid_loss=metrics[3], train_f1=f1['train'], test_f1=f1['test'],
                                      test_accuracy=f1.get('test_accuracy', ''), directed_edges=edges.size(1),
                                      canonical_retained_edges=retained,
                                      achieved_ratio=retained / original_edge_count if original_edge_count else 0.,
                                      inference_time_sec=time.perf_counter() - started)
                        if model_identity(model) != record['model_state_sha256']:
                            raise RuntimeError('budgeted inference changed the frozen model')
                except (RuntimeError, MemoryError, ValueError, OSError) as exc:
                    record.update(status='EVALUATION_FAILED', error=f'{type(exc).__name__}: {exc}')
            if model_identity(model) != last_identity:
                raise RuntimeError('budgeted inference did not restore the last model')
            upsert_record(csv_path, record)
            records.append(record)
            print('[ScaffoldBudgetedUnion] ' + json.dumps(record, sort_keys=True), flush=True)
    return records
