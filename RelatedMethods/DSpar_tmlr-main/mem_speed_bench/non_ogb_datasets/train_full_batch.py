import sys
import os
from pathlib import Path
import numpy as np
from torch.autograd import grad
sys.path.append(os.getcwd())
import argparse
import inspect
import random
import time
import warnings
import yaml
import pdb

SUPPORT_GRAPH_ROOT = Path(os.environ.get("SUPPORT_GRAPH_ROOT", Path(__file__).resolve().parents[4])).resolve()
if str(SUPPORT_GRAPH_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPPORT_GRAPH_ROOT))
from scaffold_gnn.utils.defaults import DEFAULT_DATA_DIR
from scripts.common.baseline_result_utils import (
    RunTimeBudget,
    multilabel_roc_auc_f1_percent,
    single_label_roc_auc_percent,
)
from scaffold_gnn.models.model import MPNNs as TunedGNNMPNN

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler

from torch_geometric.utils import subgraph
from torch_geometric.utils import degree
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from dspar import get_memory_usage, compute_tensor_bytes, exp_recorder
import models
from data import get_benchmark_data, get_data
from logger import Logger
from BenchmarkDataset import select_pyg_split
from sklearn.metrics import f1_score
import torch_geometric.transforms as T

MB = 1024**2
GB = 1024**3


parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=str, required=True, 
                    help='the path to the configuration file')
parser.add_argument('--dataset', type=str, required=True, 
                    help='the name of the applied dataset')
parser.add_argument('--root', type=str, default=DEFAULT_DATA_DIR)
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=None,
                    help='Override epochs from the YAML config.')
parser.add_argument('--grad_norm', type=float, default=None)
parser.add_argument('--inductive', action='store_true')
parser.add_argument('--debug_mem', action='store_true')
parser.add_argument('--test_speed', action='store_true')
parser.add_argument('--amp', help='whether to enable apx mode', action='store_true')
parser.add_argument('--random_sparsify', help='whether to randomly sparsify the graph', action='store_true')
parser.add_argument('--spec_sparsify', help='whether to spectrally sparsify the graph', action='store_true')
parser.add_argument('--kept_ratio', type=float, default=None,
                    help='Optional common kept-edge sample budget for benchmark wrappers.')
parser.add_argument('--hidden_channels', type=int, default=None)
parser.add_argument('--num_layers', type=int, default=None)
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--weight_decay', type=float, default=None)
parser.add_argument('--dropout', type=float, default=None)
parser.add_argument('--input_dropout', type=float, default=None)
parser.add_argument('--metric', choices=('acc', 'rocauc'), default=None)
parser.add_argument(
    '--eval_graph', '--eval-graph',
    choices=('sparse', 'original'),
    default=os.environ.get('BASELINE_EVAL_GRAPH', 'sparse'),
    help='Topology used for validation/test inference.',
)
parser.add_argument('--pre_linear', type=int, choices=(0, 1), default=None)
parser.add_argument('--residual', type=int, choices=(0, 1), default=None)
parser.add_argument('--layer_norm', type=int, choices=(0, 1), default=None)
parser.add_argument('--batch_norm', type=int, choices=(0, 1), default=None)
parser.add_argument('--jumping_knowledge', type=int, choices=(0, 1), default=None)
parser.add_argument(
    '--tunedgnn_medium_backbone',
    action='store_true',
    help=(
        "Use tunedGNN's exact medium-graph MPNN architecture around DSpar's "
        "sparsified adjacency."
    ),
)



def get_optimizer(model_config, model):
    weight_decay = float(model_config.get('weight_decay', 0.0))
    if model_config['optim'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=model_config['lr'],
            weight_decay=weight_decay,
        )
    elif model_config['optim'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=model_config['lr'],
            weight_decay=weight_decay,
        )
    else:
        raise NotImplementedError
    return optimizer


def build_model(model_config, in_channels, out_channels, *, tunedgnn_backbone):
    """Build DSpar's native model or the exact tunedGNN comparison backbone."""

    architecture = model_config['architecture']
    if not tunedgnn_backbone:
        GNN = getattr(models, model_config['arch_name'])
        # The shared tunedGNN contract contains optional fields such as
        # ``pre_linear`` and ``jumping_knowledge``.  DSpar's native GCN does
        # not implement those fields; passing them through made every
        # large-graph run fail before the model was constructed.  Retain every
        # option supported by the selected native architecture and explicitly
        # report any inapplicable fields.
        accepted = set(inspect.signature(GNN.__init__).parameters)
        native_architecture = {
            key: value for key, value in architecture.items() if key in accepted
        }
        ignored = sorted(set(architecture) - set(native_architecture))
        if ignored:
            print(
                '[DSparBackbone] ignoring unsupported native fields: '
                + ', '.join(ignored)
            )
        return GNN(
            in_channels=in_channels,
            out_channels=out_channels,
            **native_architecture,
        )

    model_config['arch_name'] = 'TunedGNNMPNN'
    return TunedGNNMPNN(
        in_channels=in_channels,
        hidden_channels=int(architecture['hidden_channels']),
        out_channels=out_channels,
        local_layers=int(architecture['num_layers']),
        dropout=float(architecture.get('dropout', 0.0)),
        heads=1,
        pre_ln=False,
        pre_linear=bool(architecture.get('pre_linear', False)),
        res=bool(architecture.get('residual', False)),
        ln=bool(architecture.get('layer_norm', False)),
        bn=bool(architecture.get('batch_norm', False)),
        jk=bool(architecture.get('jumping_knowledge', False)),
        gnn='gcn',
    )


def to_inductive(data):
    mask = data.train_mask
    data.x = data.x[mask]
    data.y = data.y[mask]
    data.train_mask = data.train_mask[mask]
    data.test_mask = None
    data.edge_index, _ = subgraph(mask, data.edge_index, None,
                                  relabel_nodes=True, num_nodes=data.num_nodes)
    data.num_nodes = mask.sum().item()
    return data


def supervised_loss(logits, labels):
    if labels.dim() == 1 or (labels.dim() > 1 and labels.size(-1) == 1):
        labels = labels.reshape(-1)
        valid = labels >= 0
        return F.cross_entropy(logits[valid], labels[valid])
    valid = torch.isfinite(labels) & (labels >= 0)
    losses = F.binary_cross_entropy_with_logits(
        logits,
        torch.nan_to_num(labels, nan=0.0).float(),
        reduction='none',
    )
    return losses[valid].mean()


def train(model, optimizer, data, grad_norm, scaler, amp_mode):
    model.train()
    optimizer.zero_grad()
    with autocast(enabled=amp_mode):
        out = model(data.x, data.adj_t)
        loss = supervised_loss(
            out[data.train_mask], data.y[data.train_mask]
        )
    del data
    if amp_mode:
        scaler.scale(loss).backward()
        if grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()
    return loss.item()


def compute_micro_f1(logits, y, mask=None) -> float:
    if mask is not None:
        logits, y = logits[mask], y[mask]

    if y.dim() == 1:
        return int(logits.argmax(dim=-1).eq(y).sum()) / y.size(0)
        
    else:
        y_pred = logits > 0
        y_true = y > 0.5

        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())

        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            return 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0.


def compute_macro_f1(logits, y, mask=None) -> float:
    if mask is not None:
        logits, y = logits[mask], y[mask]
    if y.dim() == 1:
        prediction = logits.argmax(dim=-1).detach().cpu().numpy()
        truth = y.detach().cpu().numpy()
    else:
        prediction = (logits > 0).detach().cpu().numpy()
        truth = (y > 0.5).detach().cpu().numpy()
    return float(f1_score(truth, prediction, average='macro', zero_division=0))


def compute_multilabel_metrics(logits, y, mask=None):
    if mask is not None:
        logits, y = logits[mask], y[mask]
    roc_auc, macro_f1 = multilabel_roc_auc_f1_percent(y, logits)
    return roc_auc / 100.0, macro_f1 / 100.0


def compute_single_label_roc_auc(logits, y, mask=None) -> float:
    if mask is not None:
        logits, y = logits[mask], y[mask]
    return single_label_roc_auc_percent(y, logits.float()) / 100.0


@torch.no_grad()
def test(model, data, amp_mode, metric='acc'):
    model.eval()
    with autocast(enabled=amp_mode):
        out = model(data.x, data.adj_t)
    y_true = data.y
    if y_true.dim() > 1 and y_true.size(-1) > 1:
        # OGBN-Proteins' official metric is mean per-task ROC-AUC.  The old
        # non-OGB bridge reported micro-F1 in the accuracy field, which is why
        # otherwise valid runs appeared as roughly 8% accuracy.
        train_acc, train_f1 = compute_multilabel_metrics(
            out, y_true, data.train_mask
        )
        valid_acc, _ = compute_multilabel_metrics(
            out, y_true, data.val_mask
        )
        test_acc, test_f1 = compute_multilabel_metrics(
            out, y_true, data.test_mask
        )
    else:
        # Minesweeper and Questions are scored by ROC-AUC.  Their accuracy is
        # pinned at the 80%/97% majority-class rate whatever the model learns,
        # which would also make the validation-based selection in Logger
        # degenerate, so --metric has to reach this branch.
        primary = (
            compute_single_label_roc_auc
            if str(metric).lower() == 'rocauc'
            else compute_micro_f1
        )
        train_acc = primary(out, y_true, data.train_mask)
        valid_acc = primary(out, y_true, data.val_mask)
        test_acc = primary(out, y_true, data.test_mask)
        train_f1 = compute_macro_f1(out, y_true, data.train_mask)
        test_f1 = compute_macro_f1(out, y_true, data.test_mask)
    return train_acc, valid_acc, test_acc, train_f1, test_f1


def main():
    global args 
    args = parser.parse_args()
    with open(args.conf, 'r') as fp:
        model_config = yaml.load(fp, Loader=yaml.FullLoader)
        name = model_config['name']
        loop = model_config.get('loop', False)
        normalize = model_config.get('norm', False)
        if 'all' in model_config.get('params', {}):
            model_config = model_config['params']['all']
        elif args.dataset == 'reddit2':
            model_config = model_config['params']['reddit']
        elif args.dataset in model_config.get('params', {}):
            model_config = model_config['params'][args.dataset]
        else:
            model_config = model_config['params']['reddit']
        model_config['name'] = name
        model_config['loop'] = loop
        model_config['normalize'] = normalize
        if args.epochs is not None:
            model_config['epochs'] = args.epochs
        architecture = model_config['architecture']
        for argument, key in (
            (args.hidden_channels, 'hidden_channels'),
            (args.num_layers, 'num_layers'),
            (args.dropout, 'dropout'),
        ):
            if argument is not None:
                architecture[key] = argument
        if args.lr is not None:
            model_config['lr'] = args.lr
        if args.weight_decay is not None:
            model_config['weight_decay'] = args.weight_decay
        if args.residual is not None:
            architecture['residual'] = bool(args.residual)
        if args.batch_norm is not None:
            architecture['batch_norm'] = bool(args.batch_norm)
        if args.layer_norm is not None:
            architecture['layer_norm'] = bool(args.layer_norm)
        if args.input_dropout is not None:
            architecture['input_dropout'] = float(args.input_dropout)
        if args.pre_linear is not None:
            architecture['pre_linear'] = bool(args.pre_linear)
        if args.jumping_knowledge is not None:
            architecture['jumping_knowledge'] = bool(args.jumping_knowledge)
        model_config['pre_linear'] = bool(args.pre_linear or 0)
        model_config['layer_norm'] = bool(args.layer_norm or 0)
        model_config['input_dropout'] = args.input_dropout
        model_config['metric'] = args.metric
        if args.tunedgnn_medium_backbone:
            # TunedGNN's GCNConv performs both operations internally.  Leaving
            # DSpar's native preprocessing enabled would normalize twice.
            model_config['loop'] = False
            model_config['normalize'] = False

    print(f'model config: {model_config}')
    if args.tunedgnn_medium_backbone:
        print(
            '[TunedGNNBackbone] profile=medium gnn=gcn '
            f'hidden_channels={architecture["hidden_channels"]} '
            f'num_layers={architecture["num_layers"]} '
            f'dropout={architecture.get("dropout", 0.0)} '
            f'pre_linear={bool(architecture.get("pre_linear", False))} '
            f'residual={bool(architecture.get("residual", False))} '
            f'layer_norm={bool(architecture.get("layer_norm", False))} '
            f'batch_norm={bool(architecture.get("batch_norm", False))} '
            'normalization=inside_gcnconv'
        )
    print(f'clipping grad norm: {args.grad_norm}')
    args.model = model_config['arch_name']
    assert model_config['name'] in ['GCN', 'SAGE', 'GCN2']
    if args.amp:
        print('activate amp mode')
        scaler = GradScaler()
    else:
        scaler = None
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if torch.cuda.is_available() and args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
        print("Use GPU {} for training".format(args.gpu))
    else:
        device = torch.device('cpu')
        print("Use CPU for training")

    if args.spec_sparsify or args.random_sparsify:
        assert args.spec_sparsify ^ args.random_sparsify, "both the flags of random_sparsify and spec_sparsify are true."
        enable_sparsify = True
        suffix = 'mode: ' + 'random' if args.random_sparsify else 'spectral'
        print(f'enable sparsify flag, {suffix}')
    else:
        enable_sparsify = False
    data, num_features, num_classes = get_benchmark_data(
        args.root,
        args.dataset,
        False,
        enable_sparsify,
        args.random_sparsify,
        args.kept_ratio,
        preserve_undirected=args.tunedgnn_medium_backbone,
    )
    eval_data = None
    if args.eval_graph == 'original':
        eval_data, eval_num_features, eval_num_classes = get_benchmark_data(
            args.root,
            args.dataset,
            False,
            False,
            False,
            None,
            preserve_undirected=args.tunedgnn_medium_backbone,
        )
        if eval_num_features != num_features or eval_num_classes != num_classes:
            raise RuntimeError('DSpar full-evaluation dataset metadata mismatch')
        print(
            '[EvaluationGraph] topology=original-full '
            f'directed_edges={eval_data.num_edges} training_topology=sparse',
            flush=True,
        )
    args.actual_kept_ratio = getattr(
        data,
        'dspar_actual_kept_ratio',
        args.kept_ratio,
    )
    args.actual_sparsity = (
        100.0 * (1.0 - float(args.actual_kept_ratio))
        if args.actual_kept_ratio is not None
        else None
    )
    if args.kept_ratio is not None:
        print(
            f'DSpar target ratio: requested={args.kept_ratio:.8f}, '
            f'achieved={args.actual_kept_ratio:.8f}, '
            f'original_edges={getattr(data, "dspar_original_num_edges", data.num_edges)}, '
            f'kept_edges={getattr(data, "dspar_actual_num_edges", data.num_edges)}'
        )
    multi_label = data.y.dim() > 1 and data.y.size(-1) > 1

    model = build_model(
        model_config,
        num_features,
        num_classes,
        tunedgnn_backbone=args.tunedgnn_medium_backbone,
    )
    print(model)
    model.to(device)

    if args.debug_mem:
        print("========== Model and Optimizer only ===========")
        optimizer = get_optimizer(model_config, model)
        optimizer.zero_grad()
        model.reset_parameters()
        model.train()
        usage = get_memory_usage(args.gpu, False)
        exp_recorder.record("network", args.model)
        exp_recorder.record("model_only", usage / MB, 4)
        print("========== Load data to GPU ===========")
        print('converting data form...')
        s_time = time.time()
        data = T.ToSparseTensor()(data.to(device))
        print(f'done. used {time.time() - s_time} sec')

        if model_config['loop']:
            t = time.perf_counter()
            print('Adding self-loops...', end=' ', flush=True)
            data.adj_t = data.adj_t.set_diag()
            print(f'Done! [{time.perf_counter() - t:.2f}s]')
        
        if model_config['normalize']:
            t = time.perf_counter()
            print('Normalizing data...', end=' ', flush=True)
            data.adj_t = gcn_norm(data.adj_t, add_self_loops=False)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        if args.inductive:
            print('inductive learning mode')
            data = to_inductive(data)
        # data.adj_t.fill_cache_()
        init_mem = get_memory_usage(args.gpu, False)
        data_mem = init_mem / MB - exp_recorder.val_dict['model_only']
        exp_recorder.record("data", init_mem / MB - exp_recorder.val_dict['model_only'], 4)
        out = model(data.x, data.adj_t)[data.train_mask]
        loss = supervised_loss(out, data.y[data.train_mask])
        print("========== Before Backward ===========")
        before_backward = get_memory_usage(args.gpu, True)
        act_mem = get_memory_usage(args.gpu, False) - init_mem - compute_tensor_bytes([loss, out])

        res = "Total Mem: %.2f MB\tData Mem: %.2f MB\tAct Mem: %.2f MB" % (before_backward / MB,
                                                                           data_mem,
                                                                           act_mem / MB)
        print(res)

        loss.backward()
        optimizer.step()
        del loss, out
        print("========== After Backward ===========")
        after_backward = get_memory_usage(args.gpu, True)
        total_mem = before_backward + (after_backward - init_mem)
        res = "Total Mem: %.2f MB\tData Mem: %.2f MB\tAct Mem: %.2f MB" % (total_mem / MB,
                                                                           data_mem,
                                                                           act_mem / MB)
        print(res)
        exp_recorder.record("total", total_mem / MB, 2)
        exp_recorder.record("activation", act_mem / MB, 2)
        # exp_recorder.dump('mem_results.json')
        s_time = time.time()
        if args.test_speed:
            model.reset_parameters()
            optimizer.zero_grad()
            epoch_per_sec = []
            for i in range(100):
                optimizer.zero_grad()
                t = time.time()
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                out = model(data.x, data.adj_t)[data.train_mask]
                loss = supervised_loss(out, data.y[data.train_mask])
                loss.backward()
                optimizer.step()
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                duration = time.time() - t
                epoch_per_sec.append(duration)
                print(f'epoch {i}, duration: {duration} sec')
            print(f's/epoch: {np.mean(epoch_per_sec)}')
            print(f'training epoch/s: {100/np.sum(epoch_per_sec)}')

            model.eval()
            s_time = time.time()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            with torch.no_grad():
                for _ in range(100):
                    out = model(data.x, data.adj_t)           
            if device.type == 'cuda':
                torch.cuda.synchronize()
            print(f'inference epoch/s: {100/(time.time() - s_time) }') 
        exit()

    print('converting data form...')
    s_time = time.time()
    data = T.ToSparseTensor()(data.to(device))
    if eval_data is not None:
        eval_data = T.ToSparseTensor()(eval_data.to(device))
    print(f'done. used {time.time() - s_time} sec')

    if model_config['loop']:
        t = time.perf_counter()
        print('Adding self-loops...', end=' ', flush=True)
        data.adj_t = data.adj_t.set_diag()
        if eval_data is not None:
            eval_data.adj_t = eval_data.adj_t.set_diag()
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
    
    if model_config['normalize']:
        t = time.perf_counter()
        print('Normalizing data...', end=' ', flush=True)
        data.adj_t = gcn_norm(data.adj_t, add_self_loops=False)
        if eval_data is not None:
            eval_data.adj_t = gcn_norm(
                eval_data.adj_t,
                add_self_loops=False,
            )
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    if args.inductive:
        print('inductive learning mode')
        data = to_inductive(data)
        if eval_data is not None:
            eval_data = to_inductive(eval_data)
    logger = Logger(args.runs, args)
    single_label_metric = str(args.metric or model_config.get('metric') or 'acc').lower()
    metric_name = (
        'ROC-AUC' if multi_label or single_label_metric == 'rocauc' else 'Accuracy'
    )
    for run in range(args.runs):
        select_pyg_split(data, run)
        if eval_data is not None:
            select_pyg_split(eval_data, run)
        model.reset_parameters()
        optimizer = get_optimizer(model_config, model)
        budget = RunTimeBudget().start()
        for epoch in range(1, 1 + model_config['epochs']):
            loss = train(model, optimizer, data, args.grad_norm, scaler, args.amp)
            print(f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Train Loss: {loss:.4f}')
    
            result = test(
                model,
                eval_data if eval_data is not None else data,
                args.amp,
                metric=single_label_metric,
            )
            logger.add_result(run, result)
            train_acc, valid_acc, test_acc, train_f1, test_f1 = result
            print(f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Train {metric_name}: {100 * train_acc:.2f}%, '
                    f'Valid {metric_name}: {100 * valid_acc:.2f}% '
                    f'Test {metric_name}: {100 * test_acc:.2f}% '
                    f'Test F1 Macro: {100 * test_f1:.2f}%')
            # This epoch is already in the logger, so print_statistics() below
            # still selects the best epoch this run actually reached.
            if budget.exhausted(run + 1, epoch, model_config['epochs']):
                break

        logger.add_result(run, result)
        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == '__main__':
    main()
