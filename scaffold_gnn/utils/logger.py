import torch
import os
import csv

try:
    from scripts.common.baseline_result_utils import append_baseline_result
except ModuleNotFoundError:
    from scripts.common.baseline_result_utils import append_baseline_result


def _get_results_root():
    return os.environ.get('RESULTS_ROOT', 'results')


def _model_path(args, run):
    """Resolve checkpoints under --model_dir while preserving the legacy default."""
    model_root = getattr(args, 'model_dir', None)
    if not model_root or os.path.normpath(str(model_root)) == os.path.normpath('./model/'):
        model_root = 'models'
    dataset_dir = os.path.join(str(model_root), str(args.dataset))
    if args.model == 'MPNN':
        filename = f'{args.model}_{args.gnn}_{run}.pt'
    else:
        filename = f'{args.model}_{run}.pt'
    return dataset_dir, os.path.join(dataset_dir, filename)


def model_checkpoint_path(args, run):
    """Public path resolver shared by final-inference artifact metadata."""

    return _model_path(args, run)[1]

class Logger(object):

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]
        self.epochs = [[] for _ in range(runs)]
        self.test = None
        self.test_f1 = None

    def add_result(self, run, result, epoch=None):
        assert len(result) in [4, 6]
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)
        self.epochs[run].append(len(self.results[run]) - 1 if epoch is None else int(epoch))

    def print_statistics(
        self,
        run=None,
        mode='max_acc',
        *,
        initial_sparsification_time_sec=None,
        resparsification_time_sec=None,
        sparsification_time_sec=None,
        training_time_sec=None,
        total_runtime_sec=None,
        train_only_time_sec=None,
        eval_time_sec=None,
        selection_time_sec=None,
    ):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            if result.dim() < 2 or result.size(-1) < 4:
                # Evaluation was disabled, so each epoch logged fewer columns
                # than the [train, valid, test, loss] this summary indexes. The
                # run itself is complete -- crashing here threw away 366 s of
                # finished pokec training for a line of reporting.
                print(f'Run {run + 1:02d}: no evaluation recorded '
                      f'(result shape {tuple(result.shape)}); '
                      f'summary skipped.')
                return
            argmax = result[:, 1].argmax().item()
            argmin = result[:, 3].argmin().item()
            ind = argmax if mode == 'max_acc' else argmin
            epoch = self.epochs[run][ind] if self.epochs[run] else ind
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'Highest Test: {result[:, 2].max():.2f}')
            print(f'Chosen epoch: {epoch}')
            print(f'Final Train: {result[ind, 0]:.2f}')
            print(f'Final Test: {result[ind, 2]:.2f}')
            self.test = result[ind, 2]
            if result.shape[1] >= 6:
                self.test_f1 = result[ind, 5]
                print(f'Final Test F1 (Macro): {result[ind, 5]:.2f}')
            append_baseline_result(
                method=(
                    os.environ.get('SCAFFOLD_EXPERIMENT_METHOD')
                    or getattr(self.info, 'sparsifier', None)
                ),
                dataset=getattr(self.info, 'dataset', None),
                run=run + 1,
                seed=(self.info.seed + run if getattr(self.info, 'seed', None) is not None else None),
                epochs=getattr(self.info, 'epochs', None),
                kept_ratio=getattr(
                    self.info,
                    'achieved_kept_ratio',
                    getattr(self.info, 'target_ratio', None),
                ),
                train_acc=result[ind, 0].item(),
                valid_acc=result[ind, 1].item(),
                test_acc=result[ind, 2].item(),
                train_f1_macro=result[ind, 4].item() if result.shape[1] >= 6 else None,
                test_f1_macro=result[ind, 5].item() if result.shape[1] >= 6 else None,
                chosen_epoch=epoch,
                initial_sparsification_time_sec=initial_sparsification_time_sec,
                resparsification_time_sec=resparsification_time_sec,
                sparsification_time_sec=sparsification_time_sec,
                training_time_sec=training_time_sec,
                total_runtime_sec=total_runtime_sec,
                train_only_time_sec=train_only_time_sec,
                eval_time_sec=eval_time_sec,
                selection_time_sec=selection_time_sec,
            )
            return None
        if not self.results or not any(self.results):
            print('No results to display.')
            return (None, None)
        valid_results = [r for r in self.results if r]
        if not valid_results:
            print('No valid results to display.')
            return (None, None)
        # Runs can stop at different epochs when the graceful time budget is
        # reached.  Keep them as separate tensors: stacking ragged histories
        # used to discard otherwise valid, fully evaluated timeout results.
        run_results = [100 * torch.tensor(values) for values in valid_results]
        run_results = [
            values
            for values in run_results
            if values.dim() >= 2 and values.size(-1) >= 4
        ]
        if not run_results:
            print('No evaluated results to display.')
            return (None, None)
        best_results = []
        has_f1 = all(values.size(-1) >= 6 for values in run_results)
        for r in run_results:
            train1 = r[:, 0].max().item()
            test1 = r[:, 2].max().item()
            valid = r[:, 1].max().item()
            best_idx = r[:, 1].argmax() if mode == 'max_acc' else r[:, 3].argmin()
            train2 = r[best_idx, 0].item()
            test2 = r[best_idx, 2].item()
            if has_f1:
                train_f1 = r[best_idx, 4].item()
                test_f1 = r[best_idx, 5].item()
                best_results.append((train1, test1, valid, train2, test2, train_f1, test_f1))
            else:
                best_results.append((train1, test1, valid, train2, test2))
        best_result = torch.tensor(best_results)
        print('=' * 70)
        print(f'Results across {len(best_result)} runs:')
        print('=' * 70)

        def _std(x):
            return x.std(unbiased=False).item() if x.numel() > 1 else 0.0
        r = best_result[:, 0]
        print(f'Highest Train Accuracy: {r.mean():.2f} ± {_std(r):.2f}')
        r = best_result[:, 1]
        print(f'Highest Test Accuracy:  {r.mean():.2f} ± {_std(r):.2f}')
        r = best_result[:, 2]
        print(f'Highest Valid Accuracy: {r.mean():.2f} ± {_std(r):.2f}')
        r = best_result[:, 3]
        print(f'Final Train Accuracy:   {r.mean():.2f} ± {_std(r):.2f}')
        r = best_result[:, 4]
        print(f'Final Test Accuracy:    {r.mean():.2f} ± {_std(r):.2f}')
        if has_f1:
            r = best_result[:, 5]
            print(f'Final Train F1 (Macro): {r.mean():.2f} ± {_std(r):.2f}')
            r = best_result[:, 6]
            print(f'Final Test F1 (Macro):  {r.mean():.2f} ± {_std(r):.2f}')
            print('=' * 70)
            self.test = best_result[:, 4].mean()
            self.test_f1 = best_result[:, 6].mean()
            return (best_result[:, 4], best_result[:, 6])
        else:
            print('=' * 70)
            self.test = best_result[:, 4].mean()
            self.test_f1 = None
            return (best_result[:, 4], None)

    def output(self, out_path, info):
        with open(out_path, 'a') as f:
            f.write(info)
            f.write(f'test acc:{self.test}\n')

def save_model(args, model, optimizer, run):
    dataset_dir, model_path = _model_path(args, run)
    os.makedirs(dataset_dir, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, model_path)

def load_model(args, model, optimizer, run):
    _, model_path = _model_path(args, run)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return (model, optimizer)

def get_sparsifier_params_string(args):
    sp = args.sparsifier
    params = []
    keep_ratio = getattr(args, 'target_ratio', None)
    dropout_prob = None if keep_ratio is None else 1.0 - float(keep_ratio)
    if sp == 'full':
        return 'full'
    elif sp == 'mst':
        params.append(f'alg={args.mst_algorithm}')
        return f"mst_{'_'.join(params)}"
    elif sp.startswith('tspanner'):
        params.append(f'stretch={args.spanner_stretch}')
        return f"{sp}_{'_'.join(params)}"
    elif sp == 'k_random_neighbor':
        params.append(f'k={args.k_neighbors}')
        return f"k_random_{'_'.join(params)}"
    elif sp == 'random':
        if dropout_prob is not None:
            params.append(f'drop={dropout_prob}')
        if args.dropout_force_undirected:
            params.append('undir=True')
        return f"random_{'_'.join(params)}"
    elif sp == 'er':
        params.append(f'eps={args.er_epsilon}')
        return f"er_{'_'.join(params)}"
    elif sp in ('support', 'support_dilation'):
        params.append(f'target_ratio={args.target_ratio}')
        params.append(f'init={args.support_init}')
        if sp == 'support':
            params.append(f'cong={getattr(args, "support_congestion", "node")}')
        return f"{sp}_{'_'.join(params)}"
    return sp

def get_all_sparsifier_params_dict(args):
    sp = args.sparsifier
    params = {}
    params['sparsifier'] = sp
    
    # Common param
    if hasattr(args, 'target_ratio'):
        params['target_ratio'] = args.target_ratio
        
    if sp == 'mst':
        params['mst_algorithm'] = getattr(args, 'mst_algorithm', 'kruskal')
        params['mst_weight'] = getattr(args, 'mst_weight', None)
    if 'tspanner' in sp:
        params['spanner_stretch'] = getattr(args, 'spanner_stretch', None)
    if sp == 'k_random_neighbor':
        params['k_neighbors'] = getattr(args, 'k_neighbors', None)
    if sp == 'random':
        if hasattr(args, 'target_ratio'):
            params['dropout_prob'] = 1.0 - float(args.target_ratio)
        else:
            params['dropout_prob'] = None
        params['dropout_force_undirected'] = getattr(args, 'dropout_force_undirected', False)
    if sp == 'er':
        params['er_epsilon'] = getattr(args, 'er_epsilon', None)
    if sp in ('support', 'support_dilation'):
        params['support_target_ratio'] = getattr(args, 'target_ratio', None)
        params['support_init'] = getattr(args, 'support_init', 'mst')
        params['support_congestion'] = getattr(args, 'support_congestion', 'node') if sp == 'support' else ''
        params['support_verbose'] = getattr(args, 'support_verbose', False)
    return params

def save_result(
    args,
    test_accs,
    test_f1s=None,
    sparsification_time_sec=None,
    training_time_sec=None,
    total_runtime_sec=None,
    initial_sparsification_time_sec=None,
    resparsification_time_sec=None,
):
    results_root = _get_results_root()
    sparsifier_dir = f'{results_root}/{args.dataset}/{args.sparsifier}'
    if not os.path.exists(sparsifier_dir):
        os.makedirs(sparsifier_dir)
    filename = f'{sparsifier_dir}/results.csv'
    sparsifier_params = get_all_sparsifier_params_dict(args)
    file_exists = os.path.exists(filename)
    header = [
        'sparsifier', 'target_ratio', 'mst_algorithm', 'mst_weight', 'spanner_stretch',
        'k_neighbors', 'dropout_prob', 'dropout_force_undirected', 'er_epsilon',
        'support_target_ratio', 'support_init', 'support_congestion', 'support_verbose',
        'model', 'gnn', 'lr', 'hidden_channels', 'local_layers', 'dropout', 'pre_ln',
        'pre_linear', 'ln', 'bn', 'res', 'jk', 'num_heads', 'epochs', 'runs', 'seed',
        'test_acc_mean', 'test_acc_std', 'test_f1_mean', 'test_f1_std',
        'sparsification_time_sec', 'training_time_sec', 'total_runtime_sec',
        'initial_sparsification_time_sec', 'resparsification_time_sec',
    ]
    row = [
        sparsifier_params.get('sparsifier', ''),
        sparsifier_params.get('target_ratio', ''),
        sparsifier_params.get('mst_algorithm', ''),
        sparsifier_params.get('mst_weight', ''),
        sparsifier_params.get('spanner_stretch', ''),
        sparsifier_params.get('k_neighbors', ''),
        sparsifier_params.get('dropout_prob', ''),
        sparsifier_params.get('dropout_force_undirected', ''),
        sparsifier_params.get('er_epsilon', ''),
        sparsifier_params.get('support_target_ratio', ''),
        sparsifier_params.get('support_init', ''),
        sparsifier_params.get('support_congestion', ''),
        sparsifier_params.get('support_verbose', ''),
        args.model,
        args.gnn,
        args.lr,
        args.hidden_channels,
        args.local_layers,
        args.dropout,
        getattr(args, 'pre_ln', False),
        getattr(args, 'pre_linear', False),
        getattr(args, 'ln', False),
        getattr(args, 'bn', False),
        getattr(args, 'res', False),
        getattr(args, 'jk', False),
        getattr(args, 'num_heads', 1),
        args.epochs,
        args.runs,
        args.seed,
        f'{test_accs.mean():.4f}',
        f'{(test_accs.std(unbiased=False).item() if len(test_accs) > 1 else 0.0):.4f}',
        f'{test_f1s.mean():.4f}' if test_f1s is not None else '',
        f'{(test_f1s.std(unbiased=False).item() if test_f1s is not None and len(test_f1s) > 1 else 0.0):.4f}' if test_f1s is not None else '',
        f'{sparsification_time_sec:.3f}' if sparsification_time_sec is not None else '',
        f'{training_time_sec:.3f}' if training_time_sec is not None else '',
        f'{total_runtime_sec:.3f}' if total_runtime_sec is not None else '',
        f'{initial_sparsification_time_sec:.3f}' if initial_sparsification_time_sec is not None else '',
        f'{resparsification_time_sec:.3f}' if resparsification_time_sec is not None else '',
    ]
    if file_exists:
        with open(filename, 'r', newline='') as f:
            existing_rows = list(csv.reader(f, skipinitialspace=True))
        if existing_rows and existing_rows[0] != header:
            old_header = existing_rows[0]
            expanded_header = old_header + [column for column in header if column not in old_header]
            padded_rows = []
            for old_row in existing_rows[1:]:
                padded_rows.append(old_row + [''] * (len(expanded_header) - len(old_row)))
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(expanded_header)
                writer.writerows(padded_rows)
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)
