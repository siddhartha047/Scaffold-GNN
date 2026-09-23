"""Opt-in validation-selected model/support snapshots and test-only diagnostics."""

from contextlib import contextmanager
import copy
import csv
import json
import math
import os
from pathlib import Path

import torch


def cpu_model_state(model):
    return {name: value.detach().cpu().clone() if torch.is_tensor(value)
            else copy.deepcopy(value) for name, value in model.state_dict().items()}


def clear_graph_caches(model):
    for module in model.modules():
        for name in ('_cached_edge_index', '_cached_adj_t'):
            if hasattr(module, name):
                setattr(module, name, None)


def checkpoint_selection_policy(k_effective):
    if k_effective == 0:
        return 'best-validation-model-and-full-graph'
    if k_effective == 1:
        return 'best-validation-model-and-graph'
    return 'best-validation-model-and-union'


def atomic_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


class BestValidationCheckpoint:
    """Choose only by validation; never choose a model/support by test score.

    Epochs in model_epoch/test diagnostic fields are one-based. graph_epoch
    follows the existing bank's zero-based epoch convention. Full-graph
    snapshots have k_effective=0, graph_epoch=-1 and empty source-graph lists.
    """

    LOG_FIELDS = ('model_epoch', 'graph_epoch', 'graph_epochs', 'historical_validation_scores', 'k_effective', 'valid_metric', 'test_metric',
                  'test_accuracy', 'best_validation_so_far',
                  'highest_test_metric_so_far', 'highest_test_accuracy_so_far',
                  'checkpoint_updated')

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.best = None
        self.highest_test_metric = -math.inf
        self.highest_test_metric_epoch = None
        self.highest_test_accuracy = -math.inf
        self.highest_test_accuracy_epoch = None
        self.log_path = self.directory / 'epoch_metrics.csv'
        # A retry is a new training trajectory; do not mix it with an old log.
        with self.log_path.open('w', newline='') as handle:
            csv.DictWriter(handle, fieldnames=self.LOG_FIELDS).writeheader()

    @property
    def best_path(self):
        return self.directory / 'best.pt'

    @property
    def last_path(self):
        return self.directory / 'last.pt'

    def observe(self, *, model, view, graph_epoch, model_epoch, valid_metric,
                test_metric, test_accuracy=None, graph_epochs=None,
                historical_validation_scores=None, k_effective=1, support_records=None,
                selection_policy=None):
        valid_metric, test_metric = float(valid_metric), float(test_metric)
        if not all(math.isfinite(value) for value in (valid_metric, test_metric)):
            raise ValueError('non-finite validation/test checkpoint diagnostic')
        graph_epochs = list(graph_epochs) if graph_epochs is not None else [int(graph_epoch)]
        historical_validation_scores = (list(historical_validation_scores)
                                        if historical_validation_scores is not None else [valid_metric])
        if len(graph_epochs) != k_effective or len(historical_validation_scores) != k_effective:
            raise ValueError('checkpoint graph metadata must match its actual graph count')
        if test_metric > self.highest_test_metric:
            self.highest_test_metric = test_metric
            self.highest_test_metric_epoch = model_epoch
        if test_accuracy is not None and math.isfinite(float(test_accuracy)):
            if float(test_accuracy) > self.highest_test_accuracy:
                self.highest_test_accuracy = float(test_accuracy)
                self.highest_test_accuracy_epoch = model_epoch
        updated = self.best is None or valid_metric > self.best['valid_metric']
        if updated:
            if support_records is not None:
                if (len(support_records) != k_effective
                        or [int(s['epoch']) for s in support_records] != graph_epochs):
                    raise ValueError('constituent support snapshots must match checkpoint graph epochs')
            edges, weights = view
            self.best = dict(model_state_dict=cpu_model_state(model),
                             edge_index=edges.detach().cpu().clone(),
                             edge_weight=None if weights is None else weights.detach().cpu().clone(),
                             model_epoch=int(model_epoch), graph_epoch=int(graph_epoch),
                             graph_epochs=graph_epochs.copy(), k_effective=int(k_effective),
                             historical_validation_scores=historical_validation_scores.copy(),
                             valid_metric=valid_metric, test_metric=test_metric,
                             test_accuracy=None if test_accuracy is None else float(test_accuracy),
                             selection_policy=selection_policy or checkpoint_selection_policy(k_effective))
            if support_records is not None:
                # Winning constituents may later be evicted from the rolling
                # scratch bank. Capture exact edges, not just epoch references.
                self.best['support_records'] = [dict(
                    epoch=int(s['epoch']), validation_score=float(s['validation_score']),
                    edge_index=s['view'][0].detach().cpu().clone(),
                    edge_weight=None if s['view'][1] is None else s['view'][1].detach().cpu().clone())
                    for s in support_records]
            atomic_checkpoint(self.best_path, self.best)
        row = dict(model_epoch=model_epoch, graph_epoch=graph_epoch,
                   graph_epochs=json.dumps(graph_epochs),
                   historical_validation_scores=json.dumps(historical_validation_scores),
                   k_effective=k_effective,
                   valid_metric=valid_metric, test_metric=test_metric,
                   test_accuracy='' if test_accuracy is None else float(test_accuracy),
                   best_validation_so_far=self.best['valid_metric'],
                   highest_test_metric_so_far=self.highest_test_metric,
                   highest_test_accuracy_so_far='' if self.highest_test_accuracy_epoch is None else self.highest_test_accuracy,
                   checkpoint_updated=updated)
        with self.log_path.open('a', newline='') as handle:
            csv.DictWriter(handle, fieldnames=self.LOG_FIELDS).writerow(row)
        return updated

    def diagnostics(self):
        return dict(highest_test_metric=self.highest_test_metric if self.highest_test_metric_epoch is not None else '',
                    highest_test_metric_epoch=self.highest_test_metric_epoch or '',
                    highest_test_accuracy=self.highest_test_accuracy if self.highest_test_accuracy_epoch is not None else '',
                    highest_test_accuracy_epoch=self.highest_test_accuracy_epoch or '',
                    best_validation_model_epoch=self.best['model_epoch'] if self.best else '')

    def save_last(self, model, model_epoch):
        atomic_checkpoint(self.last_path, dict(model_state_dict=cpu_model_state(model),
                                               model_epoch=int(model_epoch),
                                               selection_policy='last-epoch',
                                               **self.diagnostics()))

    @contextmanager
    def use_best(self, model):
        if self.best is None:
            raise ValueError('no validation-selected model/support was captured')
        last = cpu_model_state(model)
        was_training = model.training
        try:
            model.load_state_dict(self.best['model_state_dict'])
            clear_graph_caches(model)
            model.eval()
            yield self.best
        finally:
            model.load_state_dict(last)
            clear_graph_caches(model)
            model.train(was_training)
