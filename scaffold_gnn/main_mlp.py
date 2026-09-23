import argparse
import os
import time

import torch
import torch.nn.functional as F

from scaffold_gnn.models.model_mlp import NodeMLP
from scaffold_gnn.utils.data_utils import eval_acc, eval_f1_macro
from scaffold_gnn.utils.defaults import DEFAULT_DATA_DIR
from scaffold_gnn.utils.dataset import canonical_split_fingerprint, canonicalize_dataset_name, load_dataset
from scaffold_gnn.utils.eval import evaluate
from scaffold_gnn.utils.graph_utils import edge_stats
from scaffold_gnn.utils.logger import Logger, save_result
from scaffold_gnn.utils.seed import set_seed, set_seed_from_args
from scaffold_gnn.sparsifiers.scaffold.tunedgnn_presets import argparse_defaults as tunedgnn_defaults


def parser_add_main_args(parser):
    parser.add_argument('--dataset', type=str, default='roman-empire')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--gpu', '--device', dest='gpu', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--valid_prop', type=float, default=0.25)
    parser.add_argument('--rand_split', action='store_true')
    parser.add_argument('--rand_split_class', action='store_true')
    parser.add_argument('--label_num_per_class', type=int, default=20)
    parser.add_argument('--valid_num', type=int, default=500)
    parser.add_argument('--test_num', type=int, default=1000)
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc'])
    parser.add_argument('--model', type=str, default='MLP')
    parser.add_argument(
        '--model_profile', type=str, default='medium',
        choices=['small', 'medium', 'large', 'products', 'proteins'],
    )
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--local_layers', type=int, default=3)
    parser.add_argument('--pre_linear', action='store_true')
    parser.add_argument('--ln', dest='ln', action='store_true')
    parser.add_argument('--no_ln', dest='ln', action='store_false')
    parser.set_defaults(ln=False)
    parser.add_argument('--bn', dest='bn', action='store_true')
    parser.add_argument('--no_bn', dest='bn', action='store_false')
    parser.set_defaults(bn=False)
    parser.add_argument('--res', dest='res', action='store_true')
    parser.add_argument('--no_res', dest='res', action='store_false')
    parser.set_defaults(res=False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--in_dropout', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'])
    parser.add_argument('--lr_scheduler', type=str, default='none', choices=['none', 'plateau'])
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.75)
    parser.add_argument('--lr_scheduler_patience', type=int, default=50)
    parser.add_argument('--display_step', type=int, default=100)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--eval_start_epoch', type=int, default=1)
    parser.add_argument('--large_batch_size', type=str, default='auto')
    parser.add_argument('--neighbor_batch_size', type=int, default=1024)
    parser.add_argument('--neighbor_eval_batch_size', type=int, default=2048)
    parser.add_argument('--mlp_batch_size', type=str, default='auto')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--model_dir', type=str, default='./model/')
    parser.add_argument('--wandb_project', type=str, default='sparsification')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_tags', nargs='*', default=None)
    parser.add_argument('--wandb_notes', type=str, default=None)
    parser.add_argument('--wandb_mode', type=str, default='disabled', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_log_freq', type=int, default=10)
    parser.add_argument('--wandb_watch', action='store_true')
    return parser


def apply_tunedgnn_gcn_defaults(parser, argv=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--dataset', type=str, default='roman-empire')
    known, _ = pre_parser.parse_known_args(argv)
    defaults = tunedgnn_defaults(known.dataset, 'gcn')
    accepted = {action.dest for action in parser._actions}
    parser.set_defaults(**{key: value for key, value in defaults.items() if key in accepted})
    return parser


def get_device(args):
    if args.cpu:
        return torch.device('cpu')
    return torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')


def setup_wandb(args):
    wb = None
    wb_run = None
    if getattr(args, 'wandb_mode', 'disabled') == 'disabled':
        return wb, wb_run

    try:
        import wandb as wb  # noqa: N812

        os.environ.setdefault('WANDB_SILENT', 'true')
        group = args.wandb_group if args.wandb_group else f'{args.dataset}/mlp'
        wb_run = wb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            group=group,
            tags=args.wandb_tags,
            notes=args.wandb_notes,
            config=vars(args),
            mode=args.wandb_mode,
            reinit=True,
        )
    except Exception as exc:
        print(f'W&B disabled: {exc}')
        wb = None
        wb_run = None
    return wb, wb_run


def get_split_indices(dataset, args, device):
    if hasattr(dataset, 'shared_splits'):
        splits = [{key: torch.as_tensor(value).to(device) for key, value in split.items()}
                  for split in dataset.shared_splits]
        return splits[0] if len(splits) == 1 else splits
    if args.rand_split_class:
        from scaffold_gnn.utils.data_utils import class_rand_splits

        split_idx = class_rand_splits(
            dataset.label.squeeze(),
            args.label_num_per_class,
            args.valid_num,
            args.test_num,
        )
    elif args.rand_split:
        split_idx = dataset.get_idx_split('random', args.train_prop, args.valid_prop)
    else:
        from scaffold_gnn.sparsifiers.scaffold.tunedgnn_splits import tunedgnn_fixed_splits

        tuned_splits = tunedgnn_fixed_splits(args.data_dir, dataset, args.dataset)
        if tuned_splits is None:
            split_idx = dataset.get_idx_split('random', args.train_prop, args.valid_prop)
        elif len(tuned_splits) == 1:
            split_idx = tuned_splits[0]
        else:
            return [
                {key: value.to(device) for key, value in split.items()}
                for split in tuned_splits
            ]

    for key in split_idx:
        split_idx[key] = split_idx[key].to(device)
    return split_idx


def print_split_stats(split_idx, args):
    split_type = (
        'Random (Balanced Class)'
        if args.rand_split_class
        else ('Random (Proportional)' if args.rand_split else 'Fixed/Default')
    )
    print('-' * 50)
    if isinstance(split_idx, list):
        first = split_idx[0]
        print(f'Data Split Stats (Logic: {split_type}, fixed_splits={len(split_idx)}):')
        print(f"  Train Nodes (split0): {first['train'].size(0)}")
        print(f"  Valid Nodes (split0): {first['valid'].size(0)}")
        print(f"  Test Nodes  (split0): {first['test'].size(0)}")
        return
    print(f'Data Split Stats (Logic: {split_type}):')
    print(f"  Train Nodes: {split_idx['train'].size(0)}")
    print(f"  Valid Nodes: {split_idx['valid'].size(0)}")
    print(f"  Test Nodes:  {split_idx['test'].size(0)}")


def get_criterion_and_eval(args):
    dataset_name = canonicalize_dataset_name(args.dataset)
    bce_datasets = {'questions', 'ogbn-proteins'}
    criterion = torch.nn.BCEWithLogitsLoss() if dataset_name in bce_datasets else torch.nn.NLLLoss()

    if dataset_name == 'ogbn-proteins' and args.metric != 'rocauc':
        print('Overriding metric to rocauc for ogbn-proteins (multi-label task).')
        args.metric = 'rocauc'

    if args.metric == 'rocauc':
        from scaffold_gnn.utils.data_utils import eval_rocauc

        eval_func = eval_rocauc
    else:
        eval_func = eval_acc

    return criterion, eval_func, bce_datasets


def set_seed_for_run(args, run):
    set_seed(args.seed + run)


def input_edge_stats(graph):
    edge_index = graph['edge_index']
    directed = int(edge_index.size(1))
    self_loops = int((edge_index[0] == edge_index[1]).sum().item())
    if bool(graph.get('edge_index_is_undirected_unique', False)):
        return directed, directed - self_loops, self_loops
    if bool(graph.get('edge_index_is_symmetric_unique', False)):
        return directed, (directed - self_loops) // 2, self_loops
    return edge_stats(edge_index)


def use_batched_mlp(args):
    return str(args.model_profile).lower() in {'large', 'products', 'proteins'}


def resolve_mlp_batch_size(args, *, evaluation=False):
    requested = str(args.mlp_batch_size).strip().lower()
    if requested not in {'', 'auto'}:
        return max(1, int(requested))
    if str(args.model_profile).lower() == 'proteins':
        return max(
            1,
            int(
                args.neighbor_eval_batch_size
                if evaluation
                else args.neighbor_batch_size
            ),
        )
    large_requested = str(args.large_batch_size).strip().lower()
    if large_requested not in {'', 'auto'}:
        return max(1, int(large_requested))
    return 65536 if not evaluation else 131072


def train_mlp_batched(
    model,
    features,
    labels,
    train_idx,
    args,
    criterion,
    dataset_name,
    bce_datasets,
    optimizer,
    device,
):
    model.train()
    indices = train_idx.detach().cpu()
    if indices.numel() > 1:
        indices = indices[torch.randperm(indices.numel())]
    batch_size = resolve_mlp_batch_size(args)
    total_loss = 0.0
    total_examples = 0
    for start in range(0, int(indices.numel()), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_x = features[batch_idx].to(device, non_blocking=True)
        batch_y = labels[batch_idx].to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(batch_x)
        if dataset_name in bce_datasets:
            if batch_y.ndim == 1 or batch_y.shape[-1] == 1:
                batch_y = F.one_hot(batch_y.reshape(-1), labels.max() + 1)
            loss = criterion(logits, batch_y.to(torch.float))
        else:
            loss = criterion(F.log_softmax(logits, dim=1), batch_y.reshape(-1))
        loss.backward()
        optimizer.step()
        examples = int(batch_idx.numel())
        total_loss += float(loss.item()) * examples
        total_examples += examples
    return total_loss / max(1, total_examples)


@torch.no_grad()
def evaluate_mlp_batched(
    model,
    features,
    labels,
    split_idx,
    args,
    eval_func,
    criterion,
    device,
    out_channels,
):
    model.eval()
    batch_size = resolve_mlp_batch_size(args, evaluation=True)
    output = torch.empty((features.size(0), out_channels), dtype=torch.float32)
    for start in range(0, int(features.size(0)), batch_size):
        end = min(int(features.size(0)), start + batch_size)
        output[start:end] = model(
            features[start:end].to(device, non_blocking=True)
        ).detach().float().cpu()
    labels_cpu = labels.detach().cpu()
    split_cpu = {key: value.detach().cpu() for key, value in split_idx.items()}
    train_acc = eval_func(labels_cpu[split_cpu['train']], output[split_cpu['train']])
    valid_acc = eval_func(labels_cpu[split_cpu['valid']], output[split_cpu['valid']])
    test_acc = eval_func(labels_cpu[split_cpu['test']], output[split_cpu['test']])
    if canonicalize_dataset_name(args.dataset) in ('questions', 'ogbn-proteins'):
        if labels_cpu.ndim == 1 or labels_cpu.shape[1] == 1:
            true_label = F.one_hot(labels_cpu.reshape(-1), labels_cpu.max() + 1)
        else:
            true_label = labels_cpu
        valid_loss = criterion(
            output[split_cpu['valid']],
            true_label.squeeze(1)[split_cpu['valid']].to(torch.float),
        )
    else:
        valid_loss = criterion(
            F.log_softmax(output[split_cpu['valid']], dim=1),
            labels_cpu.squeeze()[split_cpu['valid']],
        )
    return train_acc, valid_acc, test_acc, valid_loss, output


def log_wandb_epoch(
    wb,
    wb_run,
    args,
    run,
    epoch,
    train_acc,
    valid_acc,
    test_acc,
    valid_loss,
    train_f1,
    test_f1,
    loss,
    best_val,
    best_test,
):
    if wb_run is None:
        return
    if epoch % max(1, int(getattr(args, 'wandb_log_freq', 1))) != 0:
        return
    try:
        wb.log(
            {
                'run': run,
                'epoch': epoch,
                'train/acc': float(train_acc),
                'valid/acc': float(valid_acc),
                'test/acc': float(test_acc),
                'valid/loss': float(valid_loss.item()) if hasattr(valid_loss, 'item') else float(valid_loss),
                'train/f1_macro': float(train_f1),
                'test/f1_macro': float(test_f1),
                'train/loss': float(loss.item()) if hasattr(loss, 'item') else float(loss),
                'best/valid_acc': float(best_val),
                'best/test_acc': float(best_test),
            }
        )
    except Exception:
        pass


def log_wandb_final(wb, wb_run, test_accs, test_f1s):
    if wb_run is None:
        return
    try:
        payload = {
            'final/test_acc_mean': float(test_accs.mean().item()) if hasattr(test_accs.mean(), 'item') else float(test_accs.mean()),
            'final/test_acc_std': float(test_accs.std(unbiased=False).item()) if test_accs.numel() > 1 else 0.0,
        }
        if test_f1s is not None:
            payload.update(
                {
                    'final/test_f1_macro_mean': float(test_f1s.mean().item()) if hasattr(test_f1s.mean(), 'item') else float(test_f1s.mean()),
                    'final/test_f1_macro_std': float(test_f1s.std(unbiased=False).item()) if test_f1s.numel() > 1 else 0.0,
                }
            )
        wb.log(payload)
    except Exception:
        pass


# Seconds spent inside the training step, summed over one run's epochs.
_MLP_TRAIN_SEC = [0.0]
_mlp_device = None


def main():
    parser = argparse.ArgumentParser(description='Node MLP Experiments (No Edges)')
    parser = parser_add_main_args(parser)
    parser = apply_tunedgnn_gcn_defaults(parser)
    args = parser.parse_args()
    args.gnn = 'mlp'
    args.sparsifier = 'mlp'

    device = get_device(args)
    set_seed_from_args(args)
    wb, wb_run = setup_wandb(args)
    dataset_name = canonicalize_dataset_name(args.dataset)

    print(f'Loading dataset: {dataset_name}...')
    dataset = load_dataset(args.data_dir, args.dataset)
    print(f'Dataset {dataset_name} loaded successfully.')

    graph, label = dataset[0]
    if not hasattr(dataset, 'label'):
        dataset.label = label
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    batched_mlp = use_batched_mlp(args)
    data_device = torch.device('cpu') if batched_mlp else device
    graph['node_feat'] = graph['node_feat'].to(data_device)
    label = label.to(data_device)
    dataset.label = dataset.label.to(data_device)

    if not hasattr(dataset, 'num_classes'):
        dataset.num_classes = max(dataset.label.max().item() + 1, dataset.label.shape[1])

    split_idx = get_split_indices(dataset, args, data_device)
    print_split_stats(split_idx, args)
    split_for_hash = split_idx[0] if isinstance(split_idx, list) else split_idx
    print(
        f'[DatasetSplit] fingerprint={canonical_split_fingerprint(split_for_hash)}',
        flush=True,
    )

    n = dataset.graph['num_nodes']
    d = dataset.graph['node_feat'].shape[1]
    c = dataset.num_classes
    raw_edge_index = dataset.graph['edge_index']
    raw_stats = input_edge_stats(dataset.graph)
    empty_edge_index = torch.empty((2, 0), dtype=raw_edge_index.dtype, device=device)
    dataset.graph['edge_index'] = empty_edge_index

    r_dir, r_undir, r_self = raw_stats
    print('-' * 50)
    print('MLP baseline: edges removed')
    print('Original graph stats:')
    print(f'  Total Directed Edges:   {r_dir}')
    print(f'  Unique Undirected Edges: {r_undir}')
    print(f'  Self Loops:             {r_self}')
    print(f'  Average Degree:         {2 * r_undir / n:.2f}')
    print('Training graph stats:')
    print('  Total Directed Edges:   0')
    print('  Unique Undirected Edges: 0')
    print('  Self Loops:             0')
    print('  Average Degree:         0.00')
    print('-' * 50)
    print(f'MLP Input: nodes={n}, feature_dim={d}, hidden={args.hidden_channels}, layers={args.local_layers}')
    if batched_mlp:
        print(
            '[MLPBatching] graph-free node batching enabled '
            f'train_batch_size={resolve_mlp_batch_size(args)} '
            f'eval_batch_size={resolve_mlp_batch_size(args, evaluation=True)}',
            flush=True,
        )

    model = NodeMLP(
        in_channels=d,
        hidden_channels=args.hidden_channels,
        out_channels=c,
        local_layers=args.local_layers,
        dropout=args.dropout,
        in_dropout=args.in_dropout,
        pre_linear=args.pre_linear,
        res=args.res,
        ln=args.ln,
        bn=args.bn,
    ).to(device)
    criterion, eval_func, bce_datasets = get_criterion_and_eval(args)

    logger = Logger(args.runs, args)
    model.train()

    print('MODEL:', model)
    print('-' * 50)
    print(f'Starting Training for {args.runs} runs...')

    if wb_run is not None and getattr(args, 'wandb_watch', False):
        try:
            wb.watch(model, log='all', log_freq=max(1, int(getattr(args, 'wandb_log_freq', 1))))
        except Exception:
            pass

    per_run_training_times = []
    for run in range(args.runs):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        # Bracket the training step so this row lands on the same clock as
        # main.py's. Without it the MLP reported whole-process wall time --
        # interpreter start, dataset load, evaluation and all -- and read
        # slower than the GCN, which is impossible: it is the same dense
        # transform with the aggregation removed.
        global _mlp_device
        _MLP_TRAIN_SEC[0] = 0.0
        _mlp_device = device
        run_start = time.perf_counter()
        set_seed_for_run(args, run)
        split_for_run = split_idx[run % len(split_idx)] if isinstance(split_idx, list) else split_idx
        train_idx = split_for_run['train'].to(data_device)
        model.reset_parameters()
        optimizer_class = torch.optim.AdamW if args.optimizer == 'adamw' else torch.optim.Adam
        optimizer = optimizer_class(
            model.parameters(), weight_decay=args.weight_decay, lr=args.lr
        )
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=args.lr_scheduler_factor,
                patience=args.lr_scheduler_patience,
            )
            if args.lr_scheduler == 'plateau'
            else None
        )
        scheduler_metric = 0.0
        best_val = float('-inf')
        best_test = float('-inf')

        for epoch in range(args.epochs):
            if _mlp_device is not None and _mlp_device.type == 'cuda':
                torch.cuda.synchronize(_mlp_device)
            _epoch_train_start = time.perf_counter()
            if batched_mlp:
                loss = train_mlp_batched(
                    model,
                    dataset.graph['node_feat'],
                    dataset.label,
                    train_idx,
                    args,
                    criterion,
                    dataset_name,
                    bce_datasets,
                    optimizer,
                    device,
                )
            else:
                model.train()
                optimizer.zero_grad()
                out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
                if dataset_name in bce_datasets:
                    if dataset.label.shape[1] == 1:
                        true_label = F.one_hot(
                            dataset.label, dataset.label.max() + 1
                        ).squeeze(1)
                    else:
                        true_label = dataset.label
                    loss = criterion(
                        out[train_idx],
                        true_label.squeeze(1)[train_idx].to(torch.float),
                    )
                else:
                    out = F.log_softmax(out, dim=1)
                    loss = criterion(
                        out[train_idx], dataset.label.squeeze(1)[train_idx]
                    )
                loss.backward()
                optimizer.step()
            if _mlp_device is not None and _mlp_device.type == 'cuda':
                torch.cuda.synchronize(_mlp_device)
            _MLP_TRAIN_SEC[0] += time.perf_counter() - _epoch_train_start

            human_epoch = epoch + 1
            eligible = human_epoch >= max(1, int(args.eval_start_epoch))
            if int(args.eval_step) <= 0:
                should_eval = eligible and epoch == args.epochs - 1
            else:
                should_eval = eligible and (
                    epoch % int(args.eval_step) == 0
                    or epoch == args.epochs - 1
                )
            if not should_eval:
                if scheduler is not None:
                    scheduler.step(scheduler_metric)
                if epoch % args.display_step == 0:
                    print(f'Epoch: {epoch:02d}, Loss: {float(loss):.4f}, Eval: skipped')
                continue

            if batched_mlp:
                result = evaluate_mlp_batched(
                    model,
                    dataset.graph['node_feat'],
                    dataset.label,
                    split_for_run,
                    args,
                    eval_func,
                    criterion,
                    device,
                    c,
                )
            else:
                result = evaluate(
                    model, dataset, split_for_run, eval_func, criterion, args
                )
            train_acc, valid_acc, test_acc, valid_loss, out = result
            scheduler_metric = float(valid_acc)
            if scheduler is not None:
                scheduler.step(scheduler_metric)

            split_train = split_for_run['train'].to(out.device)
            split_test = split_for_run['test'].to(out.device)
            labels_for_f1 = dataset.label.to(out.device)
            train_f1 = eval_f1_macro(
                labels_for_f1[split_train], out[split_train]
            )
            test_f1 = eval_f1_macro(
                labels_for_f1[split_test], out[split_test]
            )

            logger.add_result(run, (train_acc, valid_acc, test_acc, valid_loss, train_f1, test_f1))

            if valid_acc > best_val:
                best_val = valid_acc
                best_test = test_acc

            log_wandb_epoch(
                wb,
                wb_run,
                args,
                run,
                epoch,
                train_acc,
                valid_acc,
                test_acc,
                valid_loss,
                train_f1,
                test_f1,
                loss,
                best_val,
                best_test,
            )

            if epoch % args.display_step == 0:
                print(
                    f'Epoch: {epoch:02d}, Loss: {float(loss):.4f}, '
                    f'Train: {100 * train_acc:.2f}%, Valid: {100 * valid_acc:.2f}%, '
                    f'Test: {100 * test_acc:.2f}%, Test F1: {100 * test_f1:.2f}%, '
                    f'Best Valid: {100 * best_val:.2f}%, Best Test: {100 * best_test:.2f}%'
                )

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        training_time_sec = time.perf_counter() - run_start
        per_run_training_times.append(training_time_sec)
        print(
            f'[RunTiming] run={run + 1} initial_sparsification=0.000000s '
            f'resparsification=0.000000s sparsification_total=0.000000s '
            f'training={training_time_sec:.6f}s run_wall={training_time_sec:.6f}s',
            flush=True,
        )
        logger.print_statistics(
            run,
            initial_sparsification_time_sec=0.0,
            resparsification_time_sec=0.0,
            sparsification_time_sec=0.0,
            training_time_sec=training_time_sec,
            total_runtime_sec=training_time_sec,
            train_only_time_sec=_MLP_TRAIN_SEC[0],
        )

    results = logger.print_statistics()
    if results is not None:
        test_accs, test_f1s = results
        if test_accs is not None:
            total_training_time_sec = sum(per_run_training_times)
            save_result(
                args,
                test_accs,
                test_f1s,
                sparsification_time_sec=0.0,
                training_time_sec=total_training_time_sec,
                total_runtime_sec=total_training_time_sec,
                initial_sparsification_time_sec=0.0,
                resparsification_time_sec=0.0,
            )
            results_root = os.environ.get('RESULTS_ROOT', 'results')
            print(f'Results saved to: {results_root}/{args.dataset}/{args.sparsifier}/results.csv')
            log_wandb_final(wb, wb_run, test_accs, test_f1s)

    if wb_run is not None:
        try:
            wb_run.finish()
        except Exception:
            pass


if __name__ == '__main__':
    main()
