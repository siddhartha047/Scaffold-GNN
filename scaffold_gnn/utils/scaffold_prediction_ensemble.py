"""Opt-in prediction ensembles: separate q-graph forwards, never a union.

The same model is used on every constituent. Mean logits and vote fractions
are evaluated independently on validation; test scores never select a winner.
"""
from contextlib import contextmanager
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from scaffold_gnn.utils.data_utils import eval_acc, eval_f1_macro
from scaffold_gnn.utils.scaffold_checkpoint import clear_graph_caches


PREDICTION_VIEWS = {'logits3': (3, 'mean-logits'), 'logits5': (5, 'mean-logits'),
                    'vote3': (3, 'majority-vote'), 'vote5': (5, 'majority-vote')}


def prediction_views(args):
    ks = tuple(getattr(args, 'scaffold_prediction_ks', (3, 5)))
    reducers = tuple(getattr(args, 'scaffold_prediction_reducers', ('mean-logits', 'majority-vote')))
    if not ks or len(set(ks)) != len(ks) or any(k not in (3, 5, 10) for k in ks):
        raise ValueError('prediction k values must be unique entries from 3,5,10')
    if not reducers or len(set(reducers)) != len(reducers) or any(m not in ('mean-logits', 'majority-vote') for m in reducers):
        raise ValueError('invalid prediction reducers')
    if 'majority-vote' in reducers and any(k % 2 == 0 for k in ks):
        raise ValueError('binary majority voting requires odd k')
    return {f'{"logits" if mode == "mean-logits" else "vote"}{k}': (k, mode)
            for mode in reducers for k in ks}


def prediction_policy(mode):
    if mode not in {'mean-logits', 'majority-vote'}:
        raise ValueError(f'unknown prediction reducer: {mode}')
    return f'best-validation-model-and-{mode}'


@contextmanager
def inference_without_training_side_effects(model):
    """Extra evaluations must not change the subsequent training RNG stream."""
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    was_training = model.training
    try:
        model.eval()
        clear_graph_caches(model)
        with torch.no_grad():
            yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state_all(state[3])
        model.train(was_training)
        clear_graph_caches(model)


def combine_prediction_prefixes(outputs, ks=(3, 5), reducers=('mean-logits', 'majority-vote')):
    """Consume each graph's logits once; retain only prefix aggregates.

    Voting scores are log vote-fractions (not averaged logits). Their softmax
    recovers vote fractions for ROC-AUC, and argmax is the majority label.
    Only voting requires two classes and odd k, avoiding class-vote ties.
    Mean logits supports binary, multiclass and multilabel model outputs.
    """
    ks = tuple(ks)
    if not ks or any(k not in (3, 5, 10) for k in ks):
        raise ValueError('prediction ensembles support k=3,5,10')
    voting = 'majority-vote' in reducers
    if voting and any(k % 2 == 0 for k in ks):
        raise ValueError('binary majority voting requires odd k')
    result, summed, votes = {}, None, None
    for count, logits in enumerate(outputs, 1):
        if logits.ndim != 2 or (voting and logits.size(1) != 2) or not torch.isfinite(logits).all():
            raise ValueError('finite matrix logits required; voting supports two classes only')
        if summed is None:
            summed = torch.zeros_like(logits)
            votes = torch.zeros_like(logits) if voting else None
        if logits.shape != summed.shape:
            raise ValueError('constituent prediction shapes differ')
        summed.add_(logits)
        if voting:
            votes.add_(F.one_hot(logits.argmax(dim=-1), num_classes=2).to(logits.dtype))
        if count in ks:
            if 'mean-logits' in reducers:
                result[f'logits{count}'] = summed / count
            if voting:
                result[f'vote{count}'] = (votes / count).clamp_min(torch.finfo(logits.dtype).eps).log()
        if count == max(ks):
            break
    return result


def prediction_ensemble_outputs(model, features, bank, ks=(3, 5), reducers=('mean-logits', 'majority-vote'), forward=None):
    """Share at most max(ks) forwards across prefixes; no warmup relabeling."""
    available = tuple(k for k in ks if len(bank) >= k)
    if not available:
        return {}
    with inference_without_training_side_effects(model):
        def forward_each():
            for source in bank[:max(available)]:
                edges, weights = source['view']
                clear_graph_caches(model)
                if forward is not None:
                    yield forward(edges, weights)
                else:
                    yield model(features, edges.to(features.device),
                                None if weights is None else weights.to(features.device))
        return combine_prediction_prefixes(forward_each(), available, reducers)


def score_predictions(outputs, labels, split, eval_func, criterion, args=None):
    """Use each dataset's existing loss/metric conventions on aggregated logits."""
    labels = labels.to(outputs.device)
    indices = {k: v.to(outputs.device) for k, v in split.items()}
    metrics = [float(eval_func(labels[indices[k]], outputs[indices[k]]))
               for k in ('train', 'valid', 'test')]
    multilabel = labels.ndim > 1 and labels.size(1) > 1
    if multilabel:
        target = labels.to(outputs.dtype)
        valid = indices['valid']
        mask = torch.isfinite(target[valid]) & (target[valid] >= 0)
        valid_loss = criterion(outputs[valid][mask], target[valid][mask])
    elif args is None or args.dataset == 'questions':
        target = F.one_hot(labels.reshape(-1).long(), num_classes=outputs.size(1)).to(outputs.dtype)
        valid_loss = criterion(outputs[indices['valid']], target[indices['valid']])
    else:
        valid_loss = criterion(F.log_softmax(outputs[indices['valid']], dim=1), labels[indices['valid']].reshape(-1).long())
    metrics.append(float(valid_loss))
    f1 = {k: float(eval_f1_macro(labels[indices[k]], outputs[indices[k]])) for k in ('train', 'test')}
    if not multilabel:
        f1['test_accuracy'] = float(eval_acc(labels[indices['test']], outputs[indices['test']]))
    if not all(math.isfinite(v) for v in metrics):
        raise ValueError('non-finite prediction-ensemble metrics')
    return tuple(metrics), f1


def run_prediction_checkpoint_views(*, args, model, dataset, split, trackers,
                                    original_edge_count, eval_func, criterion,
                                    device, decomposition_context, model_identity,
                                    metadata, last_identity, csv_path):
    from scaffold_gnn.utils.scaffold_multiview import FIELDS, _consume_final_full_batch, upsert_record
    q = math.ceil(float(metadata['target_ratio']) * original_edge_count)
    records = []
    for view, (k, mode) in prediction_views(args).items():
        tracker = trackers[view]
        record = dict.fromkeys(FIELDS, '')
        record.update(metadata, **tracker.diagnostics(), view=view, k_requested=k,
                      k_effective=k if tracker.best else 0,
                      status='OK' if tracker.best else 'INSUFFICIENT_GRAPHS',
                      model_policy='best-validation', selection_policy=prediction_policy(mode),
                      model_checkpoint=str(tracker.best_path), last_model_state_sha256=last_identity,
                      prediction_reducer=mode, budget_q=q, forward_passes=k,
                      graph_selection_time_sec=0.)
        started = time.perf_counter()
        if tracker.best is not None:
            try:
                with inference_without_training_side_effects(model), tracker.use_best(model) as best:
                    sources = best.get('support_records', [])
                    if (len(sources) != k or best['selection_policy'] != prediction_policy(mode)
                            or [s['epoch'] for s in sources] != best['graph_epochs']):
                        raise ValueError('winning prediction constituents/model provenance missing')
                    counts = [int((s['edge_index'][0] != s['edge_index'][1]).sum()) // 2 for s in sources]
                    if any(count != q for count in counts):
                        raise ValueError(f'constituent budgets {counts} differ from q={q}')
                    identity = model_identity(model)
                    def forward_each():
                        for source in sources:
                            yield _consume_final_full_batch(
                                model, dataset, split, source['edge_index'], source['edge_weight'],
                                args, device, decomposition_context,
                                lambda out, unused: out.detach().cpu())
                    outputs = combine_prediction_prefixes(forward_each(), (k,), (mode,))[view]
                    metrics, f1 = score_predictions(outputs, dataset.label, split, eval_func, criterion, args)
                    if abs(metrics[1] - best['valid_metric']) > 1e-4:
                        raise ValueError('saved prediction ensemble no longer reproduces validation score')
                    record.update(model_state_sha256=identity, model_epoch=best['model_epoch'],
                                  checkpoint_validation_metric=best['valid_metric'],
                                  selected_epochs=json.dumps(best['graph_epochs']),
                                  historical_validation_scores=json.dumps(best['historical_validation_scores']),
                                  validation_scores=json.dumps(best['historical_validation_scores']),
                                  train_metric=metrics[0], valid_metric=metrics[1], test_metric=metrics[2],
                                  valid_loss=metrics[3], train_f1=f1['train'], test_f1=f1['test'],
                                  test_accuracy=f1.get('test_accuracy', ''), per_graph_retained_edges=json.dumps(counts),
                                  total_edge_visits=sum(counts), canonical_retained_edges=q,
                                  achieved_ratio=q / original_edge_count,
                                  directed_edges=sum(s['edge_index'].size(1) for s in sources))
                    if model_identity(model) != identity:
                        raise RuntimeError('prediction inference mutated the frozen model')
            except (RuntimeError, MemoryError, ValueError, OSError) as exc:
                record.update(status='EVALUATION_FAILED', error=f'{type(exc).__name__}: {exc}')
        record['inference_time_sec'] = time.perf_counter() - started
        if model_identity(model) != last_identity:
            raise RuntimeError('prediction inference failed to restore the last model')
        upsert_record(csv_path, record)
        records.append(record)
        print('[ScaffoldPredictionEnsemble] ' + json.dumps(record, sort_keys=True), flush=True)
    return records
