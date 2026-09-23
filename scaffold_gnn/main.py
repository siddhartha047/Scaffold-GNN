import argparse
from contextlib import contextmanager, nullcontext
from concurrent.futures import ProcessPoolExecutor
import hashlib
import inspect
import json
import math
import multiprocessing as mp
import os
import random
import sys
import traceback
import functools
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, subgraph

from scaffold_gnn.parse import (
    BASIC_SINGLE_GRAPH_SPARSIFIERS,
    apply_scaffold_config_defaults,
    get_sparsifier_params,
    parse_method,
    parser_add_main_args,
    resolve_scaffold_backend,
    set_seed_for_run,
    use_large_model,
)
from scaffold_gnn.utils.data_utils import (
    average_acc_per_class,
    eval_acc,
    eval_acc_per_class,
    eval_f1_macro,
    label_count_distribution,
)
from scaffold_gnn.utils.dataset import NCDataset, canonical_split_fingerprint, canonicalize_dataset_name, load_dataset
from scaffold_gnn.utils.eval import evaluate, evaluate_cpu
from scaffold_gnn.utils.factory import get_sparsifier
from scaffold_gnn.utils.graph_utils import edge_stats, normalize_undirected
from scaffold_gnn.utils.logger import Logger, model_checkpoint_path, save_result
from scaffold_gnn.utils.seed import set_seed_from_args
from scaffold_gnn.utils.scaffold_multiview import (
    evaluation_graph_on_device, final_graph_bank_capacity, validate_final_views, run_final_views,
    evaluate_final_full_batch, evaluate_full_batch_outputs,
)
from scaffold_gnn.utils.scaffold_checkpoint import BestValidationCheckpoint
from scaffold_gnn.utils.scaffold_prediction_ensemble import (
    prediction_views, prediction_policy, prediction_ensemble_outputs, score_predictions,
)
from scaffold_gnn.utils.sparsifier_cache import (
    cache_entry,
    load_cached_graph,
    save_cached_graph,
)
from scaffold_gnn.sparsifiers.scaffold.tunedgnn_presets import TUNEDGNN_REVISION, get_preset
from scaffold_gnn.sparsifiers.scaffold.consensus import (
    ConsensusGraph,
    build_scaffold_consensus,
    canonical_original_edges,
    save_consensus_artifact,
    select_validation_candidate,
    support_id_from_original_ids,
    support_original_edge_ids,
)

try:
    from scripts.common.baseline_result_utils import (
        RunTimeBudget,
    )
except ModuleNotFoundError:
    from scripts.common.baseline_result_utils import RunTimeBudget


def get_device(args):
    if args.cpu:
        return torch.device('cpu')
    return torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')


def _class_accuracy_percentages(accuracy_by_class):
    if accuracy_by_class is None:
        return None
    return {
        int(class_id): (
            None if accuracy is None else round(100.0 * float(accuracy), 2)
        )
        for class_id, accuracy in sorted(accuracy_by_class.items())
    }


def configure_gpu_memory_profile(args, device):
    requested = str(getattr(args, 'gpu_memory_profile', 'auto')).lower()
    total_gb = 0.0
    device_name = None
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        total_gb = props.total_memory / (1024 ** 3)
        device_name = props.name
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    if requested == '80gb' or (requested == 'auto' and total_gb >= 60.0):
        resolved = '80gb'
        if abs(float(getattr(args, 'large_batch_mem_fraction', 0.85)) - 0.85) < 1e-9:
            args.large_batch_mem_fraction = 0.90
        # large_eval_cpu is a memory knob, not a protocol knob: it decides where
        # the full-batch forward runs, not what is computed. tunedgnn_strict used
        # to veto this, which pinned pokec -- the one dataset whose preset sets
        # it -- to CPU evaluation on an 80 GB card, at 61 s per eval against
        # 1.4 s per training epoch. Only an explicit --large_eval_cpu keeps it.
        if device.type == 'cuda' and '--large_eval_cpu' not in sys.argv:
            args.large_eval_cpu = False
    elif requested == 'conservative':
        resolved = 'conservative'
        if abs(float(getattr(args, 'large_batch_mem_fraction', 0.85)) - 0.85) < 1e-9:
            args.large_batch_mem_fraction = 0.70
    else:
        resolved = 'auto'

    args._gpu_memory_profile_resolved = resolved
    args._detected_gpu_memory_gb = total_gb
    return resolved, device_name, total_gb


def use_amp(args, device):
    return bool(getattr(args, 'amp', False)) and device.type == 'cuda' and torch.cuda.is_available()


def amp_autocast(args, device):
    if use_amp(args, device):
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()


@contextmanager
def evaluation_feature_decomposition(model, args, device=None):
    """Temporarily reduce GCN evaluation peak memory without changing math.

    PyG feature decomposition splits the message feature dimension before
    aggregation and concatenates the outputs. It is inference-only here, so
    tunedGNN training and every learned parameter remain unchanged.
    """

    requested = max(1, int(getattr(args, 'eval_decomposed_layers', 1)))
    if requested == 1 or str(getattr(args, 'gnn', '')).lower() != 'gcn':
        yield
        return

    modules = [
        module for module in model.modules()
        if module.__class__.__name__ == 'GCNConv'
        and hasattr(module, 'decomposed_layers')
    ]
    if not modules:
        raise RuntimeError(
            'GCN evaluation decomposition requested, but no GCNConv modules were found'
        )
    previous = [int(module.decomposed_layers) for module in modules]
    try:
        for module in modules:
            module.decomposed_layers = requested
        print(
            f'[GCNEvalMemory] device={device or "model-device"} '
            f'feature_chunks={requested} '
            f'convolutions={len(modules)} exact_message_decomposition=true',
            flush=True,
        )
        yield
    finally:
        for module, value in zip(modules, previous):
            module.decomposed_layers = value


def synchronize_timing_device(device):
    """Finish queued CUDA work before taking a wall-clock timing boundary."""

    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(device)


# Phase timing. Every boundary synchronises first: the training
# forward/backward is queued asynchronously, so without a sync its cost lands
# in whichever phase happens to read a tensor next.
_PHASE_SEC = {'train': 0.0, 'eval': 0.0, 'selection': 0.0}
_PHASE_DEVICE = None
# Names of the wrapped calls currently on the stack; see timed_phase().
_PHASE_DEPTH = []


def reset_phase_timers(device=None):
    """Start a fresh accounting period; call once per run."""

    global _PHASE_DEVICE
    _PHASE_DEVICE = device
    _PHASE_DEPTH.clear()
    for key in _PHASE_SEC:
        _PHASE_SEC[key] = 0.0


def timed_phase(name, function):
    """Wrap function so every call it receives accumulates into one phase.

    Re-entrant: only the OUTERMOST wrapped call on the stack accumulates.
    evaluate_large_full() dispatches to evaluate() or evaluate_cpu(), which are
    wrapped too, so a naive wrapper charged one evaluation to 'eval' twice and
    the phases no longer summed to the run.
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if _PHASE_DEPTH:
            return function(*args, **kwargs)
        device = _PHASE_DEVICE
        if device is not None:
            synchronize_timing_device(device)
        start = time.perf_counter()
        _PHASE_DEPTH.append(name)
        try:
            return function(*args, **kwargs)
        finally:
            _PHASE_DEPTH.pop()
            if device is not None:
                synchronize_timing_device(device)
            _PHASE_SEC[name] += time.perf_counter() - start

    return wrapper


def use_tensor_scaffold(args):
    return (
        args.sparsifier in {'scaffold_fast', 'scaffold_batch'}
        and resolve_scaffold_backend(args) == 'tensor'
    )


SCAFFOLD_METHODS = {
    'scaffold_greedy',
    'scaffold_heap',
    'scaffold_batch',
    'scaffold_fast',
    'scaffold_sample',
}
TUNEDGNN_GRAPH_METHODS = SCAFFOLD_METHODS | BASIC_SINGLE_GRAPH_SPARSIFIERS | {'full'}
SPARSE_VIEW_METHODS = SCAFFOLD_METHODS | {'random_refresh'}
ASYNC_RESPARSIFY_SPARSIFIERS = set(SCAFFOLD_METHODS)
NETWORKX_SCAFFOLD_MAX_NODES = 200_000
NETWORKX_SCAFFOLD_MAX_DIRECTED_EDGES = 2_000_000


def resolve_evaluation_graph(args):
    """Resolve the topology used by validation/test inference.

    ``--eval-graph`` is the method-independent experiment switch.  Leaving it
    unset preserves the historical contract: Scaffold follows its legacy
    option and every other sparsifier evaluates on its sparse training graph.
    """

    requested = getattr(args, 'eval_graph', None)
    if requested is not None:
        return str(requested).lower()
    if str(getattr(args, 'sparsifier', '')).lower() in SCAFFOLD_METHODS:
        return str(getattr(args, 'scaffold_eval_graph', 'sparse')).lower()
    return 'sparse'


def build_fixed_fast_maxst_evaluation_graph(source_data, args, edge_device):
    """Build the deterministic Fast-MaxST forest used only for inference.

    This matches the Fast-MaxST anchor: cosine feature similarity, the shared
    bucketed Kruskal implementation, and the caller's CPU-worker allocation.
    It is built once before training and remains fixed while Scaffold continues
    to resparsify its training graph.
    """

    evaluator = get_sparsifier(
        'fast_maxst',
        weight_method='cosine',
        bucket_count=int(getattr(args, 'joint_fast_tree_buckets', 256)),
        parallel_workers=int(getattr(args, 'joint_parallel_workers', 0)),
    )
    fixed_data = evaluator.sparsify(source_data)
    edge_index = fixed_data.edge_index.to(edge_device).contiguous()
    if uses_tunedgnn_scaffold_contract(args):
        edge_index = with_tunedgnn_self_loops(
            edge_index,
            int(source_data.num_nodes),
        )
    return edge_index


def uses_tunedgnn_scaffold_contract(args):
    return bool(getattr(args, 'tunedgnn_strict', False)) and (
        str(getattr(args, 'sparsifier', '')).lower() in TUNEDGNN_GRAPH_METHODS
    )


def print_tunedgnn_scaffold_contract(args):
    """Log the effective, auditable TunedGNN contract for graph methods."""

    if not uses_tunedgnn_scaffold_contract(args):
        return
    preset = get_preset(args.dataset, args.gnn)
    if preset is None:
        raise RuntimeError(
            f"Missing Benchmark original TunedGNN preset for {args.dataset}/{args.gnn}"
        )
    effective = {
        key: getattr(args, key)
        for key in vars(preset)
    }
    effective.update(
        {
            'dataset': canonicalize_dataset_name(args.dataset),
            'gnn': args.gnn,
            'data_dir': os.path.abspath(args.data_dir),
            'rand_split': bool(args.rand_split),
            'rand_split_class': bool(args.rand_split_class),
            'label_num_per_class': int(args.label_num_per_class),
            'valid_num': int(args.valid_num),
            'test_num': int(args.test_num),
            'large_eval_mode': getattr(args, 'large_eval_mode', 'partition'),
            'gat_eval_chunks': int(getattr(args, 'gat_eval_chunks', 1)),
            'exact_epochs': bool(getattr(args, 'exact_epochs', False)),
        }
    )
    print(
        '[TunedGNNContract] '
        f'source=Benchmark/original revision={TUNEDGNN_REVISION} '
        f'effective={json.dumps(effective, sort_keys=True)}',
        flush=True,
    )


def with_tunedgnn_self_loops(edge_index, num_nodes):
    """Match tunedGNN's remove-loops/add-one-loop-per-node preprocessing."""

    non_loop = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, non_loop]
    nodes = torch.arange(int(num_nodes), device=edge_index.device)
    loops = torch.stack((nodes, nodes), dim=0)
    return torch.cat((edge_index, loops), dim=1).contiguous()


def capture_training_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.random.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_training_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.random.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def sparse_graph_view(edge_index, edge_weight=None):
    """Return an immutable-by-convention graph view for evaluation history."""

    return edge_index, edge_weight


def sparse_graph_view_identity(view, num_nodes=None):
    """Stable identity for one canonical undirected non-self-loop support."""

    source = torch.as_tensor(view[0])
    try:
        version = source._version
    except RuntimeError:  # Inference-mode tensors have no mutation counter.
        version = None
    cache_key = (int(num_nodes) if num_nodes is not None else None, version)
    cached = getattr(source, '_scaffold_support_identity', None)
    if version is not None and cached is not None and cached[0] == cache_key:
        return cached[1]
    edge_index = torch.as_tensor(source, dtype=torch.long).detach().cpu()
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1 if edge_index.numel() else 0
    canonical = canonical_original_edges(edge_index, int(num_nodes))
    identity = hashlib.sha256(canonical.pairs.tobytes()).hexdigest()
    source._scaffold_support_identity = (cache_key, identity)
    return identity


def retain_best_sparse_graph_view(
    bank,
    view,
    validation_score,
    epoch,
    limit,
    *,
    distinct=False,
    num_nodes=None,
):
    """Keep the validation-ranked sparse training graph views seen so far."""

    item = {
        'validation_score': float(validation_score),
        'epoch': int(epoch),
        'view': view,
    }
    if len(bank) >= max(1, int(limit)) and (
        item['validation_score'], item['epoch']
    ) <= (bank[-1]['validation_score'], bank[-1]['epoch']):
        return bank
    if distinct:
        identity = sparse_graph_view_identity(view, num_nodes=num_nodes)
        item['support_id'] = identity
        for index, existing in enumerate(bank):
            if existing.get('support_id') != identity:
                continue
            if (
                item['validation_score'], item['epoch']
            ) > (
                existing['validation_score'], existing['epoch']
            ):
                bank[index] = item
            break
        else:
            bank.append(item)
    else:
        bank.append(item)
    bank.sort(
        key=lambda item: (item['validation_score'], item['epoch']),
        reverse=True,
    )
    del bank[max(1, int(limit)):]
    return bank


def use_sparse_graph_evaluation_views(ensemble_size, source):
    """Whether evaluation needs explicit sparse views instead of the active graph.

    A validation-ranked bank is meaningful even when its capacity is one: that
    is the single-best-support (k=1) protocol.  Fresh k=1 remains the legacy
    direct evaluation of the currently active sparse graph.
    """

    return int(ensemble_size) > 1 or str(source).lower() == 'best'


def resolve_sparse_graph_bank_dir(args, dataset_name, run):
    """Resolve one run's persistent best-training-graph bank in scratch."""

    requested = str(
        getattr(args, 'scaffold_eval_graph_bank_dir', 'auto') or 'auto'
    ).strip()
    if requested.lower() in {'none', 'off', 'false', 'disabled'}:
        return None
    if requested.lower() == 'auto':
        base = os.environ.get('SUPPORT_GRAPH_CACHE_DIR')
        if not base:
            scratch_root = os.environ.get(
                'SCAFFOLD_SCRATCH_ROOT',
                './results/cache',
            )
            ratio = str(getattr(args, 'target_ratio', 'na')).replace('.', 'p')
            # The variant must be in the path. It was hardcoded to
            # 'scaffold_fast', so a Fast cell and a Sample cell on the same
            # dataset, seed and ratio shared one bank directory and raced on
            # the same temp files -- persist_sparse_graph_bank then died with
            # FileNotFoundError when the other run's os.replace got there
            # first. Harmless while cells ran one at a time; fatal as soon as
            # a host runs several lanes over one dataset.
            base = os.path.join(
                scratch_root,
                'cache',
                str(dataset_name),
                str(getattr(args, 'sparsifier', 'scaffold_fast')),
                f'target_{ratio}',
            )
        requested = os.path.join(base, 'eval_graph_bank')
    return os.path.abspath(
        os.path.join(
            os.path.expanduser(requested),
            f'seed_{int(getattr(args, "seed", 0))}',
            f'run_{int(run) + 1:03d}',
        )
    )


def initialize_sparse_graph_bank(directory):
    """Start a clean generated bank directory for one training run."""

    if directory is None:
        return
    os.makedirs(directory, exist_ok=True)
    for filename in os.listdir(directory):
        if (
            filename == 'manifest.json'
            or (filename.startswith('epoch_') and filename.endswith('.pt'))
            or filename.startswith('.manifest.json.tmp.')
            or filename.startswith('.epoch_')
        ):
            path = os.path.join(directory, filename)
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)


def _atomic_torch_save(payload, destination):
    temporary = os.path.join(
        os.path.dirname(destination),
        f'.{os.path.basename(destination)}.tmp.{os.getpid()}',
    )
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def persist_sparse_graph_bank(
    bank,
    directory,
    *,
    dataset_name,
    target_ratio,
    run,
    split_fingerprint=None,
):
    """Persist exactly the current validation-best graph set and manifest."""

    if directory is None:
        return
    os.makedirs(directory, exist_ok=True)
    active_files = set()
    manifest_entries = []
    for rank, item in enumerate(bank, start=1):
        edge_index, edge_weight = item['view']
        filename = f'epoch_{int(item["epoch"]):06d}.pt'
        active_files.add(filename)
        destination = os.path.join(directory, filename)
        if not os.path.exists(destination):
            payload = {
                'edge_index': edge_index.detach().cpu().contiguous(),
                'edge_weight': (
                    None
                    if edge_weight is None
                    else edge_weight.detach().cpu().contiguous()
                ),
                'validation_score': float(item['validation_score']),
                'epoch': int(item['epoch']),
                'rank': int(rank),
                'dataset': str(dataset_name),
                'target_ratio': float(target_ratio),
                'run': int(run) + 1,
                'split_fingerprint': split_fingerprint,
                'support_id': item.get('support_id'),
            }
            _atomic_torch_save(payload, destination)
        manifest_entries.append({
            'rank': int(rank),
            'epoch': int(item['epoch']),
            'validation_score': float(item['validation_score']),
            'directed_edges': int(edge_index.size(1)),
            'support_id': item.get('support_id'),
            'file': filename,
        })

    for filename in os.listdir(directory):
        if (
            filename.startswith('epoch_')
            and filename.endswith('.pt')
            and filename not in active_files
        ):
            os.unlink(os.path.join(directory, filename))

    manifest = {
        'dataset': str(dataset_name),
        'target_ratio': float(target_ratio),
        'run': int(run) + 1,
        'split_fingerprint': split_fingerprint,
        'ensemble_source': 'best',
        'ensemble_reduce': 'union',
        'graphs': manifest_entries,
    }
    temporary = os.path.join(directory, f'.manifest.json.tmp.{os.getpid()}')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(temporary, os.path.join(directory, 'manifest.json'))


def update_best_sparse_graph_bank(
    bank,
    edge_index,
    edge_weight,
    validation_score,
    epoch,
    limit,
    directory,
    *,
    dataset_name,
    target_ratio,
    run,
    distinct=False,
    num_nodes=None,
    split_fingerprint=None,
):
    """Rank the current training graph, persist the bank, and return its views."""

    retain_best_sparse_graph_view(
        bank,
        sparse_graph_view(edge_index, edge_weight),
        validation_score,
        epoch,
        limit,
        distinct=distinct,
        num_nodes=num_nodes,
    )
    persist_sparse_graph_bank(
        bank,
        directory,
        dataset_name=dataset_name,
        target_ratio=target_ratio,
        run=run,
        split_fingerprint=split_fingerprint,
    )
    return [item['view'] for item in bank]


# Filled in once the dense graph is loaded; see 'Edges BEFORE sparsification'.
_ORIGINAL_EDGE_COUNTS = {}


def union_sparse_graph_views(graph_views, num_nodes):
    """Create the unweighted edge union used by the diagnostic union mode."""

    if not graph_views:
        raise ValueError('at least one sparse graph view is required')
    edge_index = torch.cat([view[0] for view in graph_views], dim=1)
    # Scaffold's finalized views are unweighted in the current experiments.
    # Deliberately make the diagnostic union unweighted even if a future view
    # carries weights; otherwise an edge's multiplicity silently changes the
    # definition from union to frequency weighting.
    merged = coalesce(edge_index, num_nodes=int(num_nodes))
    # Report the MEASURED union size. Coverage cannot be inferred from K: a
    # binomial model over independent draws is an upper bound, and SCAFFOLD
    # scores edges rather than sampling them uniformly, so high-scoring edges
    # recur across views and the union grows more slowly than that bound.
    # Without this line a K-ablation can only plot predicted coverage.
    try:
        per_view = int(graph_views[0][0].size(1))
        uni = int(merged.size(1))
        # Self-loops are added by the GNN input pipeline, so a view carries n
        # of them that the pre-sparsification count does not. Counting them in
        # the numerator but not the denominator put cora's coverage above 1.
        loops = int((merged[0] == merged[1]).sum())
        full = int(_ORIGINAL_EDGE_COUNTS.get('directed') or 0)
        cov = ('%.4f' % ((uni - loops) / full)) if full else 'na'
        print('[SCAFFOLD] union views=%d per_view_directed_edges=%d '
              'union_directed_edges=%d self_loops=%d full_directed_edges=%d '
              'coverage=%s growth_vs_single=%.3fx'
              % (len(graph_views), per_view, uni, loops, full, cov,
                 (uni / per_view) if per_view else float('nan')),
              flush=True)
    except Exception:
        pass
    return merged, None


@contextmanager
def temporary_dataset_graph_view(dataset, edge_index, edge_weight=None):
    """Temporarily expose an evaluation graph without changing training state."""

    training_edge_index = dataset.graph['edge_index']
    training_edge_weight = dataset.graph.get('edge_weight')
    dataset.graph['edge_index'] = edge_index
    if edge_weight is None:
        dataset.graph.pop('edge_weight', None)
    else:
        dataset.graph['edge_weight'] = edge_weight
    try:
        yield
    finally:
        dataset.graph['edge_index'] = training_edge_index
        if training_edge_weight is None:
            dataset.graph.pop('edge_weight', None)
        else:
            dataset.graph['edge_weight'] = training_edge_weight


def coalesce_union_edges(edge_index, edge_weight, num_nodes):
    """Deduplicate a k-view union before it reaches the model.

    The models convert their own input to a fused SpMM adjacency
    (models/spmm_adj.py). SpMM sums duplicate (row, col) pairs while gcn_norm
    on an edge list normalizes them differently -- the two disagree by ~0.17 --
    and the union of K overlapping views is the one graph in the pipeline that
    is not already coalesced. So dedupe here, and let the model do the rest.
    """

    if edge_index is None or edge_index.numel() == 0:
        return edge_index, edge_weight
    from torch_geometric.utils import coalesce as _coalesce
    if edge_weight is None:
        return _coalesce(edge_index, num_nodes=num_nodes), None
    return _coalesce(edge_index, edge_weight, num_nodes=num_nodes, reduce='max')


@torch.no_grad()
def predict_sparse_graph_ensemble(model, node_feat, graph_views, reduction, num_nodes):
    """Evaluate independent sparse views, or their edge union, with one model."""

    if not graph_views:
        raise ValueError('at least one sparse graph view is required')
    model.eval()
    reduction = str(reduction).lower()
    if reduction == 'union':
        edge_index, edge_weight = union_sparse_graph_views(graph_views, num_nodes)
        edge_index, edge_weight = coalesce_union_edges(edge_index, edge_weight, num_nodes)
        return model(node_feat, edge_index, edge_weight)

    outputs = [
        model(node_feat, edge_index, edge_weight)
        for edge_index, edge_weight in graph_views
    ]
    stacked = torch.stack(outputs, dim=0)
    if reduction == 'mean-logits':
        return stacked.mean(dim=0)
    if reduction == 'mean-probabilities':
        if stacked.size(-1) == 1:
            probability = torch.sigmoid(stacked).mean(dim=0)
            epsilon = torch.finfo(stacked.dtype).eps
            return torch.logit(probability.clamp(epsilon, 1.0 - epsilon))
        probabilities = torch.softmax(stacked, dim=-1).mean(dim=0)
        return probabilities.clamp_min(torch.finfo(stacked.dtype).tiny).log()
    if reduction != 'majority-vote':
        raise ValueError(f'unknown Scaffold evaluation ensemble reduction: {reduction}')

    if stacked.size(-1) == 1:
        # A vote fraction remains a useful ranking score for binary ROC-AUC.
        return (stacked > 0).to(stacked.dtype).mean(dim=0)
    classes = stacked.argmax(dim=-1)
    vote_fraction = F.one_hot(
        classes,
        num_classes=stacked.size(-1),
    ).to(stacked.dtype).mean(dim=0)
    # Break exact vote ties reproducibly using the mean softmax confidence.
    tie_break = torch.softmax(stacked, dim=-1).mean(dim=0)
    probabilities = vote_fraction + torch.finfo(stacked.dtype).eps * tie_break
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    return probabilities.clamp_min(torch.finfo(stacked.dtype).tiny).log()


def scaffold_consensus_enabled(args):
    return str(getattr(args, 'scaffold_consensus_mode', 'off')).lower() != 'off'


def invalidate_graph_dependent_model_caches(model):
    """Clear PyG-style normalized-adjacency caches before changing support."""

    for module in model.modules():
        for attribute in ('_cached_edge_index', '_cached_adj_t'):
            if hasattr(module, attribute):
                setattr(module, attribute, None)


def exact_budget_graph_view(undirected_pairs, args, edge_device, num_nodes):
    """Apply the same directed/self-loop convention as sparse training views."""

    pairs = torch.as_tensor(undirected_pairs, dtype=torch.long).detach().cpu()
    edge_index = torch.cat((pairs, pairs.flip(0)), dim=1).contiguous()
    data = Data(edge_index=edge_index, num_nodes=int(num_nodes))
    data.edge_index_is_symmetric_unique = True
    data.num_undirected_edges = int(pairs.size(1))
    finalized, edge_weight, stats = _finalize_sparsified_edge_index(
        data,
        args,
        edge_device,
        int(num_nodes),
        tensor_scaffold=True,
    )
    if int(stats[1]) != int(pairs.size(1)):
        raise AssertionError(
            f'exact-budget view changed q: before={pairs.size(1)} after={stats[1]}'
        )
    return sparse_graph_view(finalized, edge_weight)


def canonical_view_graph(
    view,
    original_edge_index,
    *,
    num_nodes,
    expected_q,
    args,
    edge_device,
):
    """Rebuild a bank view through the exact canonical q-edge path."""

    original = canonical_original_edges(original_edge_index, num_nodes)
    ids = support_original_edge_ids(view[0], original)
    if int(ids.size) != int(expected_q):
        raise ValueError(
            f'Scaffold-1 has {ids.size} canonical edges; expected q={expected_q}'
        )
    pairs = torch.from_numpy(original.pairs[:, ids].copy()).long()
    graph = ConsensusGraph(
        undirected_pairs=pairs,
        original_edge_ids=ids,
        forest_original_edge_ids=np.empty(0, dtype=np.int64),
        metadata={
            'schema_version': 1,
            'algorithm': 'scaffold-1',
            'm': original.edge_count,
            'q': int(expected_q),
            'target_ratio': (
                float(expected_q) / original.edge_count
                if original.edge_count else 0.0
            ),
            'num_nodes': int(num_nodes),
            'final_canonical_edges': int(ids.size),
            'output_support_id': support_id_from_original_ids(ids),
        },
    )
    return graph, exact_budget_graph_view(pairs, args, edge_device, num_nodes)


@torch.inference_mode()
def predict_exact_budget_view(
    model,
    node_feat,
    view,
    *,
    num_nodes,
    expected_q,
):
    """One cache-safe GNN forward, guarded by a canonical q-edge assertion."""

    actual = canonical_original_edges(view[0], num_nodes).edge_count
    if int(actual) != int(expected_q):
        raise AssertionError(
            f'GNN candidate has {actual} canonical edges; expected q={expected_q}'
        )
    invalidate_graph_dependent_model_caches(model)
    model.eval()
    return model(node_feat, view[0], view[1])


def resolve_scaffold_consensus_output_dir(args, dataset_name, run, bank_dir):
    requested = str(
        getattr(args, 'scaffold_consensus_output_dir', 'auto') or 'auto'
    ).strip()
    if requested.lower() == 'auto':
        if bank_dir is not None:
            requested = os.path.join(bank_dir, 'consensus')
            return os.path.abspath(requested)
        root = os.environ.get(
            'SUPPORT_GRAPH_CACHE_DIR',
            os.path.join(
                os.environ.get(
                    'SCAFFOLD_SCRATCH_ROOT',
                    './results/cache',
                ),
                'cache',
            ),
        )
        requested = os.path.join(root, str(dataset_name), 'consensus')
    return os.path.abspath(
        os.path.join(
            os.path.expanduser(requested),
            f'seed_{int(getattr(args, "seed", 0))}',
            f'run_{int(run) + 1:03d}',
        )
    )


def model_state_identity(model):
    """Content identity for the frozen in-memory checkpoint used to select."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def scaffold_consensus_fast_edge_weights(sparsifier, original, node_features):
    if str(getattr(sparsifier, 'support_weight_method', 'uniform')) == 'uniform':
        return None
    scores = sparsifier._compute_feature_edge_scores(
        node_features,
        torch.from_numpy(original.pairs[0]),
        torch.from_numpy(original.pairs[1]),
    )
    return scores.detach().cpu().numpy().astype(np.float64, copy=False)


def run_final_scaffold_consensus(
    *,
    args,
    model,
    dataset,
    split_for_run,
    eval_func,
    criterion,
    sparsifier,
    bank,
    bank_dir,
    original_edge_index,
    edge_device,
    num_nodes,
    dataset_name,
    split_fingerprint,
    run,
):
    """Final-only construction, validation selection, and one test forward."""

    if not bank:
        raise RuntimeError('Scaffold-Consensus requires a non-empty support bank')
    mode = str(args.scaffold_consensus_mode).lower()
    k = int(args.scaffold_consensus_k)
    original = canonical_original_edges(original_edge_index, num_nodes)
    q = (
        int(args.scaffold_consensus_q)
        if args.scaffold_consensus_q is not None
        else int(math.ceil(float(args.target_ratio) * original.edge_count - 1e-12))
    )
    fast_weights = scaffold_consensus_fast_edge_weights(
        sparsifier,
        original,
        dataset.graph['node_feat'],
    )
    records = [
        {
            **item,
            'dataset': dataset_name,
            'split_fingerprint': split_fingerprint,
            'run': int(run) + 1,
            'target_ratio': float(args.target_ratio),
        }
        for item in bank
    ]
    fallback_graph, fallback_view = canonical_view_graph(
        bank[0]['view'],
        original_edge_index,
        num_nodes=num_nodes,
        expected_q=q,
        args=args,
        edge_device=edge_device,
    )
    graphs = {'scaffold-1': fallback_graph}
    views = {'scaffold-1': fallback_view}
    lambdas = (
        [float(args.scaffold_consensus_lambda)]
        if mode == 'fixed'
        else [0.0, 0.5, 1.0]
    )

    synchronize_timing_device(edge_device)
    construction_start = time.perf_counter()
    for mixing_lambda in lambdas:
        name = f'consensus-lambda-{mixing_lambda:g}'
        graph = build_scaffold_consensus(
            original_edge_index,
            records,
            num_nodes=num_nodes,
            k=k,
            delta=float(args.target_ratio),
            q=(None if args.scaffold_consensus_q is None else q),
            mixing_lambda=mixing_lambda,
            edge_weights=fast_weights,
            alpha=float(getattr(sparsifier, 'alpha', 1.0)),
            edge_beta=float(getattr(sparsifier, 'edge_beta', 1.0)),
            node_beta=float(getattr(sparsifier, 'node_beta', 0.0)),
            edge_norm_p=float(getattr(sparsifier, 'edge_norm_p', 2.0)),
            node_norm_q=float(getattr(sparsifier, 'node_norm_q', 2.0)),
            workers=int(getattr(sparsifier, 'parallel_workers', 1)),
            expected_dataset=dataset_name,
            expected_split_fingerprint=split_fingerprint,
            expected_run=int(run) + 1,
            expected_target_ratio=float(args.target_ratio),
        )
        graph.metadata['num_nodes'] = int(num_nodes)
        graphs[name] = graph
        views[name] = exact_budget_graph_view(
            graph.undirected_pairs,
            args,
            edge_device,
            num_nodes,
        )
    synchronize_timing_device(edge_device)
    construction_time = time.perf_counter() - construction_start

    validation_metrics = {}
    validation_names = (
        [f'consensus-lambda-{float(args.scaffold_consensus_lambda):g}']
        if mode == 'fixed'
        else ['scaffold-1', 'consensus-lambda-0', 'consensus-lambda-0.5', 'consensus-lambda-1']
    )
    synchronize_timing_device(edge_device)
    validation_start = time.perf_counter()
    for name in validation_names:
        candidate_out = predict_exact_budget_view(
            model,
            dataset.graph['node_feat'],
            views[name],
            num_nodes=num_nodes,
            expected_q=q,
        )
        validation_metrics[name] = float(
            eval_func(
                dataset.label[split_for_run['valid']],
                candidate_out[split_for_run['valid']],
            )
        )
    synchronize_timing_device(edge_device)
    validation_time = time.perf_counter() - validation_start
    if mode == 'fixed':
        selected_name = validation_names[0]
    else:
        selected_name = select_validation_candidate(validation_metrics)
    selected_graph = graphs[selected_name]
    selected_view = views[selected_name]

    synchronize_timing_device(edge_device)
    inference_start = time.perf_counter()
    final_out = predict_exact_budget_view(
        model,
        dataset.graph['node_feat'],
        selected_view,
        num_nodes=num_nodes,
        expected_q=q,
    )
    synchronize_timing_device(edge_device)
    inference_time = time.perf_counter() - inference_start
    # ``result=`` guarantees evaluate() only derives metrics from this single
    # final forward and cannot accidentally invoke the GNN on another graph.
    result = evaluate(
        model,
        dataset,
        split_for_run,
        eval_func,
        criterion,
        args,
        result=final_out,
    )

    output_dir = resolve_scaffold_consensus_output_dir(
        args, dataset_name, run, bank_dir
    )
    os.makedirs(output_dir, exist_ok=True)
    final_metadata = {
        **dict(selected_graph.metadata),
        'dataset': dataset_name,
        'scaffold_family': str(args.sparsifier).replace('scaffold_', ''),
        'target_ratio_requested': float(args.target_ratio),
        'split_fingerprint': split_fingerprint,
        'run': int(run) + 1,
        'seed': int(args.seed) + int(run),
        'model_state_sha256': model_state_identity(model),
        'model_checkpoint': (
            os.path.abspath(model_checkpoint_path(args, run))
            if bool(getattr(args, 'save_model', False)) else None
        ),
        'support_bank_dir': bank_dir,
        'k_requested': k,
        'k_effective': int(
            next(iter(graphs[name].metadata.get('k_effective', 1) for name in graphs if name != 'scaffold-1'), 1)
        ),
        'lambda_values_considered': lambdas,
        'candidate_validation_metrics': validation_metrics,
        'selected_candidate': selected_name,
        'scaffold_1_fallback_selected': selected_name == 'scaffold-1',
        'graph_construction_time_sec': construction_time,
        'validation_selection_time_sec': validation_time,
        'final_inference_time_sec': inference_time,
        'final_validation_metric': float(result[1]),
        'final_test_metric': float(result[2]),
        'final_canonical_edges': q,
    }
    artifact_path = save_consensus_artifact(
        selected_graph,
        os.path.join(output_dir, 'selected_graph.pt'),
        metadata=final_metadata,
    )
    metadata_path = os.path.join(output_dir, 'metadata.json')
    temporary = f'{metadata_path}.tmp.{os.getpid()}'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(final_metadata, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(temporary, metadata_path)
    print(
        '[Scaffold-Consensus] '
        f'mode={mode} K_requested={k} '
        f'K_effective={final_metadata["k_effective"]} m={original.edge_count} q={q} '
        f'lambdas={lambdas} validation={json.dumps(validation_metrics, sort_keys=True)} '
        f'selected={selected_name} fallback={selected_name == "scaffold-1"} '
        f'construction={construction_time:.6f}s '
        f'validation_selection={validation_time:.6f}s '
        f'final_inference={inference_time:.6f}s artifact={artifact_path}',
        flush=True,
    )
    return result


def use_async_scaffold_resparsify(args, sparsifier, resparsify_every, refresh_epochs):
    return (
        bool(getattr(args, 'scaffold_async_resparsify', False))
        and int(getattr(args, 'scaffold_fast_maxst_every', 0)) <= 0
        and str(getattr(args, 'sparsifier', '')).lower() in ASYNC_RESPARSIFY_SPARSIFIERS
        and hasattr(sparsifier, 'resparsify')
        and int(resparsify_every) > 0
        and len(refresh_epochs) > 0
    )


@contextmanager
def scaffold_refresh_backbone(sparsifier, refresh_index, fast_maxst_every):
    """Temporarily select the mixed RandST/MaxST refresh backbone.

    Fast and Batch expose ``init_support``. Sample instead draws over a
    precomputed artifact and exposes ``sample_backbone``; its closest analogue
    to Fast-MaxST is the artifact's deterministic MaxST forest.
    """

    every = int(fast_maxst_every)
    if every < 0:
        raise ValueError('--scaffold_fast_maxst_every must be non-negative')

    attribute = (
        'sample_backbone'
        if hasattr(sparsifier, 'sample_backbone')
        else 'init_support'
    )
    previous = getattr(sparsifier, attribute, None)
    use_maxst = every > 0 and int(refresh_index) % every == 0
    replacement = (
        'fixed-maxst'
        if attribute == 'sample_backbone'
        else 'fast_maxst'
    )
    if use_maxst:
        setattr(sparsifier, attribute, replacement)
    try:
        yield str(getattr(sparsifier, attribute, 'unknown')).replace('_', '-')
    finally:
        if use_maxst:
            setattr(sparsifier, attribute, previous)


def use_neighbor_training(args):
    return getattr(args, 'train_mode', 'full') == 'neighbor'


def use_large_graph_training(args, dataset_name):
    if not use_large_model(args):
        return False
    return canonicalize_dataset_name(dataset_name) != 'karate'


def select_large_graph_pipeline(args, dataset_name):
    requested = str(getattr(args, 'large_graph_pipeline', 'auto')).lower()
    if requested != 'auto':
        return requested
    dataset_name = canonicalize_dataset_name(dataset_name)
    sparsifier = str(getattr(args, 'sparsifier', '')).lower()
    target_ratio = float(getattr(args, 'target_ratio', 1.0))
    if dataset_name == 'reddit' and sparsifier in {'scaffold_fast', 'scaffold_batch'} and target_ratio <= 0.25:
        return 'full'
    if dataset_name == 'ogbn-products':
        return 'random_node_loader'
    if dataset_name == 'ogbn-arxiv':
        return 'full'
    if dataset_name == 'ogbn-proteins':
        return 'neighbor'
    return 'node_batch'


def use_symmetric_scaffold_fastpath(args, graph):
    return bool(
        str(getattr(args, 'sparsifier', '')).lower()
        in (SCAFFOLD_METHODS | BASIC_SINGLE_GRAPH_SPARSIFIERS)
        and bool(graph.get('edge_index_is_symmetric_unique', False))
    )


def fast_symmetric_edge_stats(edge_index):
    src = edge_index[0]
    dst = edge_index[1]
    directed = int(edge_index.size(1))
    self_loops = int((src == dst).sum().item())
    undirected = int((src < dst).sum().item())
    return directed, undirected, self_loops


def symmetric_to_undirected_unique(edge_index):
    src = edge_index[0]
    dst = edge_index[1]
    mask = src < dst
    return edge_index[:, mask].contiguous()


def guard_large_networkx_scaffold(
    args,
    dataset_name,
    large_requested,
    tensor_scaffold,
    *,
    num_nodes=None,
    num_directed_edges=None,
):
    graph_too_large = (
        (num_nodes is not None and int(num_nodes) > NETWORKX_SCAFFOLD_MAX_NODES)
        or (
            num_directed_edges is not None
            and int(num_directed_edges) > NETWORKX_SCAFFOLD_MAX_DIRECTED_EDGES
        )
    )
    if not large_requested and not graph_too_large:
        return
    if args.sparsifier in {'scaffold_greedy', 'scaffold_heap'}:
        size_msg = ''
        if num_nodes is not None or num_directed_edges is not None:
            size_msg = f" (nodes={num_nodes}, directed_edges={num_directed_edges})"
        raise RuntimeError(
            f"{args.sparsifier} currently uses a NetworkX growth path and is disabled "
            f"for large dataset {dataset_name}{size_msg}. Use --sparsifier scaffold_fast "
            "with the tensor backend for large OGB/Reddit/Pokec runs."
        )
    if args.sparsifier in {'scaffold_fast', 'scaffold_batch'} and not tensor_scaffold:
        size_msg = ''
        if num_nodes is not None or num_directed_edges is not None:
            size_msg = f" nodes={num_nodes} directed_edges={num_directed_edges}."
        raise RuntimeError(
            f"Large {args.sparsifier} runs must use the tensor backend."
            f"{size_msg} Remove "
            "--scaffold_backend networkx or pass --scaffold_backend tensor."
        )


def should_resparsify_epoch(
    epoch,
    total_epochs,
    resparsify_every,
    include_final_interval=False,
):
    interval = int(resparsify_every)
    if interval <= 0 or epoch <= 0:
        return False
    if epoch % interval != 0:
        return False
    if include_final_interval:
        return epoch < int(total_epochs)
    return epoch + interval < int(total_epochs)


def scheduled_resparsify_epochs(
    total_epochs,
    resparsify_every,
    include_final_interval=False,
):
    return [
        epoch
        for epoch in range(int(total_epochs))
        if should_resparsify_epoch(
            epoch,
            total_epochs,
            resparsify_every,
            include_final_interval,
        )
    ]


def summarize_resparsify_schedule(
    total_epochs,
    resparsify_every,
    include_final_interval=False,
):
    refresh_epochs = scheduled_resparsify_epochs(
        total_epochs,
        resparsify_every,
        include_final_interval,
    )
    if not refresh_epochs:
        return 'initial graph only'
    if len(refresh_epochs) <= 6:
        epochs = ', '.join(str(epoch) for epoch in refresh_epochs)
    else:
        head = ', '.join(str(epoch) for epoch in refresh_epochs[:3])
        tail = ', '.join(str(epoch) for epoch in refresh_epochs[-3:])
        epochs = f'{head}, ..., {tail}'
    final_note = 'final interval included' if include_final_interval else 'final interval skipped'
    return f'before epochs {epochs} ({len(refresh_epochs)} refreshes; {final_note})'


_ASYNC_RESPARSIFY_SOURCE = None
_ASYNC_RESPARSIFY_NAME = None
_ASYNC_RESPARSIFY_PARAMS = None
_ASYNC_RESPARSIFY_SPARSIFIER = None
_ASYNC_RESPARSIFY_DATA = None


def _build_async_resparsify_source_payload(data, include_features=False):
    """Pickle the pre-sparsification graph for the background resparsifier.

    Node features are carried only when a feature-weighted support is in use.
    Without them the worker raises "support_weight_method=... requires node
    features" on the first refresh, which is what happened the first time any
    run set a non-uniform weighting: every prior run used uniform weights, so
    the omission was unreachable. They stay opt-in because the features are
    re-pickled on every refresh and are large on the big graphs.
    """
    edge_index = data.edge_index.detach().cpu().contiguous()
    payload = {
        'edge_index': edge_index,
        'num_nodes': int(data.num_nodes),
        'edge_index_is_undirected_unique': bool(getattr(data, 'edge_index_is_undirected_unique', False)),
        'edge_index_is_symmetric_unique': bool(getattr(data, 'edge_index_is_symmetric_unique', False)),
        'num_undirected_edges': getattr(data, 'num_undirected_edges', None),
    }
    if include_features:
        x = getattr(data, 'x', None)
        if x is None:
            raise ValueError(
                'a feature-weighted support was requested but the source graph '
                'carries no node features'
            )
        payload['x'] = x.detach().cpu().contiguous()
    return payload


def _data_from_async_resparsify_result(result):
    data = Data(
        edge_index=result['edge_index'],
        num_nodes=int(result['num_nodes']),
    )
    if bool(result.get('edge_index_is_undirected_unique', False)):
        data.edge_index_is_undirected_unique = True
    if bool(result.get('edge_index_is_symmetric_unique', False)):
        data.edge_index_is_symmetric_unique = True
    if result.get('num_undirected_edges') is not None:
        data.num_undirected_edges = int(result['num_undirected_edges'])
    if result.get('x') is not None:
        data.x = result['x']
    return data


def _attach_async_resparsify_support_cache(source_payload, sparsifier):
    if source_payload is None or not hasattr(sparsifier, 'export_deterministic_support_cache'):
        return None
    cache_payload = sparsifier.export_deterministic_support_cache()
    if cache_payload:
        source_payload['deterministic_support_cache'] = cache_payload
        print(
            '[AsyncResparsify] cached deterministic init support for worker '
            f"kind={cache_payload.get('kind')} init={cache_payload.get('init_support')}",
            flush=True,
        )
    return cache_payload


def _init_async_resparsify_worker(source_payload, sparsifier_name, sparsifier_params):
    global _ASYNC_RESPARSIFY_SOURCE
    global _ASYNC_RESPARSIFY_NAME
    global _ASYNC_RESPARSIFY_PARAMS
    _ASYNC_RESPARSIFY_SOURCE = source_payload
    _ASYNC_RESPARSIFY_NAME = sparsifier_name
    _ASYNC_RESPARSIFY_PARAMS = dict(sparsifier_params)
    # Keep PyTorch's intra-op pool from stealing all cores in the async child.
    # The tensor scaffold backend gets its concurrency from its own numba/thread
    # pool ``parallel_workers``; leaving torch=1 avoids fighting the parent.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _run_async_resparsify_job(source_payload, sparsifier_name, sparsifier_params, refresh_id, seed):
    """Sparsify one refresh graph.

    We keep the sparsifier instance and its input :class:`Data` cached in
    module globals so that repeat calls in the same async worker process reuse
    the tensor backend's plan cache (partition + owners + jobs) instead of
    rebuilding it from scratch. Only the RNG is reseeded per refresh.
    """
    global _ASYNC_RESPARSIFY_SPARSIFIER, _ASYNC_RESPARSIFY_DATA
    sparsifier = _ASYNC_RESPARSIFY_SPARSIFIER
    data = _ASYNC_RESPARSIFY_DATA
    if sparsifier is None:
        params = dict(sparsifier_params)
        params['seed'] = int(seed)
        sparsifier = get_sparsifier(sparsifier_name, **params)
        support_payload = source_payload.get('deterministic_support_cache')
        if support_payload and hasattr(sparsifier, 'load_deterministic_support_cache'):
            sparsifier.load_deterministic_support_cache(support_payload)
        _ASYNC_RESPARSIFY_SPARSIFIER = sparsifier
    else:
        # Reseed for this refresh so growth samples diverge from prior calls.
        try:
            import random as _random

            sparsifier._rng = _random.Random(int(seed))
            sparsifier._torch_gen = torch.Generator()
            sparsifier._torch_gen.manual_seed(int(seed))
            sparsifier.seed = int(seed)
        except Exception:
            pass

    if data is None:
        data = Data(
            edge_index=source_payload['edge_index'],
            num_nodes=int(source_payload['num_nodes']),
        )
        if bool(source_payload.get('edge_index_is_undirected_unique', False)):
            data.edge_index_is_undirected_unique = True
        if bool(source_payload.get('edge_index_is_symmetric_unique', False)):
            data.edge_index_is_symmetric_unique = True
        if source_payload.get('num_undirected_edges') is not None:
            data.num_undirected_edges = int(source_payload['num_undirected_edges'])
        # Feature-weighted supports score candidate edges from data.x, so the
        # worker's rebuilt source needs it too. The payload only carries x when
        # the weighting asks for it (see _build_async_resparsify_source_payload).
        if source_payload.get('x') is not None:
            data.x = source_payload['x']
        _ASYNC_RESPARSIFY_DATA = data

    start = time.perf_counter()
    if hasattr(sparsifier, '_draw_index'):
        # Sample.sparsify() does not advance its forest rotation; resparsify()
        # normally does. This worker calls sparsify directly on its cached data.
        sparsifier._draw_index = int(refresh_id)
    out = sparsifier.sparsify(data)
    build_time = time.perf_counter() - start
    edge_index = out.edge_index.detach().cpu().contiguous()
    try:
        edge_index.share_memory_()
    except Exception:
        pass
    return {
        'ok': True,
        'refresh_id': int(refresh_id),
        'seed': int(seed),
        'edge_index': edge_index,
        'num_nodes': int(getattr(out, 'num_nodes', source_payload['num_nodes'])),
        'edge_index_is_undirected_unique': bool(getattr(out, 'edge_index_is_undirected_unique', False)),
        'edge_index_is_symmetric_unique': bool(getattr(out, 'edge_index_is_symmetric_unique', False)),
        'num_undirected_edges': getattr(out, 'num_undirected_edges', None),
        'build_time_sec': float(build_time),
        'last_cluster_stats': getattr(sparsifier, 'last_cluster_stats', {}),
    }


def _async_resparsify_executor_job(refresh_id, seed):
    try:
        return _run_async_resparsify_job(
            _ASYNC_RESPARSIFY_SOURCE,
            _ASYNC_RESPARSIFY_NAME,
            _ASYNC_RESPARSIFY_PARAMS,
            refresh_id,
            seed,
        )
    except BaseException:
        return {
            'ok': False,
            'refresh_id': int(refresh_id),
            'seed': int(seed),
            'traceback': traceback.format_exc(),
        }


def _async_resparsify_mp_context():
    # The initial Scaffold construction can initialize GNU OpenMP through
    # numba/PyTorch before this executor is created.  Forking that process is
    # unsafe and causes libgomp to abort the async worker on its first refresh.
    # ``spawn`` starts the worker with a clean runtime while the initializer
    # below restores the cached graph and sparsifier configuration it needs.
    return mp.get_context('spawn')


def _create_async_resparsify_executor(source_payload, sparsifier_name, sparsifier_params):
    return ProcessPoolExecutor(
        max_workers=1,
        mp_context=_async_resparsify_mp_context(),
        initializer=_init_async_resparsify_worker,
        initargs=(source_payload, sparsifier_name, sparsifier_params),
    )


class AsyncResparsifyManager:
    def __init__(
        self,
        *,
        source_payload,
        sparsifier_name,
        sparsifier_params,
        total_refreshes,
        base_seed,
        scheduled_epochs,
        executor_factory=None,
        wait_for_schedule=False,
    ):
        self.source_payload = source_payload
        self.sparsifier_name = sparsifier_name
        self.sparsifier_params = dict(sparsifier_params)
        self.total_refreshes = max(0, int(total_refreshes))
        self.base_seed = int(base_seed)
        self.scheduled_epochs = set(int(epoch) for epoch in scheduled_epochs)
        self.executor_factory = executor_factory or _create_async_resparsify_executor
        self.executor = None
        self.future = None
        self.launched = 0
        self.applied = 0
        self.background_cpu_sec = 0.0
        self._pending_logged_epochs = set()
        self._closed = False
        self.wait_for_schedule = bool(wait_for_schedule)

    def start(self):
        if self.total_refreshes <= 0:
            return
        if self.executor is None:
            self.executor = self.executor_factory(
                self.source_payload,
                self.sparsifier_name,
                self.sparsifier_params,
            )
        self.launch_next()

    def launch_next(self):
        if self.executor is None or self.future is not None:
            return False
        if self.launched >= self.total_refreshes:
            return False
        refresh_id = self.launched + 1
        seed = self.base_seed + refresh_id
        self.future = self.executor.submit(_async_resparsify_executor_job, refresh_id, seed)
        self.launched = refresh_id
        print(f'[AsyncResparsify] launched refresh={refresh_id} seed={seed}', flush=True)
        return True

    def poll(self, run, epoch):
        if self.future is None:
            return None
        if self.wait_for_schedule and int(epoch) not in self.scheduled_epochs:
            return None
        if not self.future.done() and not self.wait_for_schedule:
            epoch = int(epoch)
            if epoch in self.scheduled_epochs and epoch not in self._pending_logged_epochs:
                self._pending_logged_epochs.add(epoch)
                print(
                    f'[AsyncResparsify] pending run={run} epoch={epoch} '
                    f'refresh={self.launched}',
                    flush=True,
                )
            return None

        try:
            result = self.future.result()
        except BaseException as exc:
            self.future = None
            raise RuntimeError(
                f'Async SCAFFOLD resparsification failed for refresh={self.launched}'
            ) from exc
        self.future = None
        if not result.get('ok', False):
            raise RuntimeError(
                'Async SCAFFOLD resparsification failed for '
                f"refresh={result.get('refresh_id')} seed={result.get('seed')}\n"
                f"{result.get('traceback', '')}"
            )
        return result

    def mark_applied(self, result, run, epoch, edge_count):
        self.applied += 1
        self.background_cpu_sec += float(result.get('build_time_sec', 0.0))
        print(
            f"[AsyncResparsify] applied run={run} epoch={epoch} "
            f"refresh={result.get('refresh_id')} seed={result.get('seed')} "
            f"edges={edge_count} build_time={float(result.get('build_time_sec', 0.0)):.3f}s",
            flush=True,
        )
        self.launch_next()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.executor is not None:
            processes = list((getattr(self.executor, '_processes', None) or {}).values())
            for process in processes:
                try:
                    if process.is_alive():
                        process.terminate()
                except Exception:
                    pass
            for process in processes:
                try:
                    process.join(timeout=1.0)
                except Exception:
                    pass
            for process in processes:
                try:
                    if process.is_alive() and hasattr(process, 'kill'):
                        process.kill()
                except Exception:
                    pass
            try:
                self.executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                self.executor.shutdown(wait=False)

    def print_summary(self):
        print(
            f'[AsyncResparsify] summary launched={self.launched} '
            f'applied={self.applied} background_cpu_sec={self.background_cpu_sec:.3f}',
            flush=True,
        )


def edge_stats_with_metadata(
    edge_index,
    *,
    undirected_unique=False,
    symmetric_unique=False,
    num_undirected_edges=None,
    num_self_loops=None,
):
    if undirected_unique:
        undirected = int(edge_index.size(1))
        self_loops = 0 if num_self_loops is None else int(num_self_loops)
        return 2 * undirected, undirected, self_loops
    if symmetric_unique:
        directed = int(edge_index.size(1))
        undirected = int(num_undirected_edges if num_undirected_edges is not None else directed // 2)
        self_loops = 0 if num_self_loops is None else int(num_self_loops)
        return directed, undirected, self_loops
    return edge_stats(edge_index)


def direct_symmetric_from_undirected_unique(edge_index):
    edge_index = edge_index.cpu()
    keep = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, keep]
    return torch.cat([edge_index, edge_index.flip(0)], dim=1).contiguous()


def _finalize_sparsified_edge_index(sparsified_data, args, edge_device, n, tensor_scaffold):
    """Convert a sparsifier output into device-resident edges, weights, and stats.

    Mirrors the block used after the initial sparsification so that we can
    re-run it after every ``--scaffold_resparsify_every`` window.
    """
    if bool(getattr(sparsified_data, 'edge_index_is_symmetric_unique', False)):
        sparsified_edge_index = sparsified_data.edge_index.to(edge_device)
        sparsified_edge_weight = getattr(sparsified_data, 'edge_weight', None)
        if sparsified_edge_weight is not None:
            sparsified_edge_weight = sparsified_edge_weight.to(
                edge_device, dtype=torch.float32
            ).contiguous()
        after_stats = edge_stats_with_metadata(
            sparsified_edge_index,
            symmetric_unique=True,
            num_undirected_edges=getattr(sparsified_data, 'num_undirected_edges', None),
        )
    elif args.sparsifier == 'full' and bool(getattr(sparsified_data, 'edge_index_is_undirected_unique', False)):
        num_undirected_edges = int(sparsified_data.edge_index.size(1))
        sparsified_edge_index = direct_symmetric_from_undirected_unique(sparsified_data.edge_index).to(edge_device)
        sparsified_edge_weight = None
        after_stats = edge_stats_with_metadata(
            sparsified_edge_index,
            symmetric_unique=True,
            num_undirected_edges=num_undirected_edges,
        )
    else:
        if getattr(sparsified_data, 'edge_weight', None) is not None:
            raise ValueError(
                'weighted sparsifier output must declare edge_index_is_symmetric_unique'
            )
        sparsified_edge_index = normalize_undirected(sparsified_data.edge_index.to(edge_device), n).to(edge_device)
        sparsified_edge_weight = None
        after_stats = edge_stats(sparsified_edge_index)
    if sparsified_edge_weight is not None:
        if (
            sparsified_edge_weight.dim() != 1
            or sparsified_edge_weight.numel() != sparsified_edge_index.size(1)
        ):
            raise ValueError('edge_weight must align with sparsified edge_index')
        if not bool(torch.isfinite(sparsified_edge_weight).all()) or bool(
            (sparsified_edge_weight <= 0).any()
        ):
            raise ValueError('edge_weight must be finite and strictly positive')
    if (
        uses_tunedgnn_scaffold_contract(args)
        and str(getattr(args, 'model_profile', '')).lower() != 'proteins'
    ):
        if sparsified_edge_weight is not None:
            non_loop = sparsified_edge_index[0] != sparsified_edge_index[1]
            sparsified_edge_weight = torch.cat((
                sparsified_edge_weight[non_loop],
                torch.ones(n, device=edge_device, dtype=sparsified_edge_weight.dtype),
            )).contiguous()
        sparsified_edge_index = with_tunedgnn_self_loops(sparsified_edge_index, n)
    return sparsified_edge_index, sparsified_edge_weight, after_stats


def setup_wandb(args):
    wb = None
    wb_run = None
    if getattr(args, 'wandb_mode', 'disabled') == 'disabled':
        return wb, wb_run

    try:
        import wandb as wb  # noqa: N812

        os.environ.setdefault('WANDB_SILENT', 'true')
        group = args.wandb_group if args.wandb_group else f'{args.dataset}/{args.sparsifier}'
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
    except Exception as e:
        print(f'W&B disabled: {e}')
        wb = None
        wb_run = None
    return wb, wb_run


def baseline_partition_enabled():
    mode = os.environ.get('BASELINE_PARTITION_MODE', 'none').strip().lower()
    return mode not in {'', 'none', 'false', '0', 'off'}


def load_baseline_partitioned_dataset(args, dataset_name):
    try:
        from scripts.common.baseline_dataset_bridge import load_pyg_data
    except ImportError:
        from scripts.common.baseline_dataset_bridge import load_pyg_data

    data, _source_dataset = load_pyg_data(args.data_dir, args.dataset)
    dataset = NCDataset(dataset_name)
    dataset.graph = {
        'edge_index': data.edge_index,
        'node_feat': data.x,
        'edge_feat': getattr(data, 'edge_attr', None),
        'num_nodes': int(data.num_nodes),
    }
    if getattr(data, 'edge_index_is_undirected_unique', False):
        dataset.graph['edge_index_is_undirected_unique'] = True
    dataset.label = data.y
    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]
    print(
        'Using baseline smoke partition: '
        f'nodes={dataset.graph["num_nodes"]}, edges={dataset.graph["edge_index"].size(1)}, '
        f'train={dataset.train_idx.numel()}, valid={dataset.valid_idx.numel()}, test={dataset.test_idx.numel()}'
    )
    return dataset


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
    elif uses_tunedgnn_scaffold_contract(args):
        from scaffold_gnn.sparsifiers.scaffold.tunedgnn_splits import tunedgnn_fixed_splits

        tuned_splits = tunedgnn_fixed_splits(
            args.data_dir,
            dataset,
            args.dataset,
        )
        if tuned_splits is None:
            split_idx = dataset.get_idx_split('random', args.train_prop, args.valid_prop)
        elif len(tuned_splits) == 1:
            split_idx = tuned_splits[0]
        else:
            return [
                {key: value.to(device) for key, value in split.items()}
                for split in tuned_splits
            ]
    elif hasattr(dataset, 'load_fixed_splits'):
        fixed_split = dataset.load_fixed_splits()
        if isinstance(fixed_split, list):
            split_idx = []
            for fs in fixed_split:
                split_idx.append(
                    {
                        'train': fs['train'].to(device),
                        'valid': fs['valid'].to(device),
                        'test': fs['test'].to(device),
                    }
                )
            return split_idx
        split_idx = {'train': fixed_split['train'], 'valid': fixed_split['valid'], 'test': fixed_split['test']}
    elif hasattr(dataset, 'train_idx') and dataset.train_idx is not None:
        split_idx = {
            'train': dataset.train_idx,
            'valid': dataset.valid_idx,
            'test': dataset.test_idx,
        }
    else:
        split_idx = dataset.get_idx_split('random', args.train_prop, args.valid_prop)

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


def print_split_class_distributions(labels, split_idx, num_classes):
    labels_cpu = torch.as_tensor(labels).detach().cpu()
    multi_label = labels_cpu.ndim > 1 and labels_cpu.shape[-1] > 1
    distribution_name = (
        'Positive-Label Distribution (count per task)'
        if multi_label
        else 'Class Distribution (node count per class)'
    )
    resolved_splits = split_idx if isinstance(split_idx, list) else [split_idx]
    for split_number, resolved_split in enumerate(resolved_splits):
        suffix = f', split={split_number}' if isinstance(split_idx, list) else ''
        print(f'{distribution_name}{suffix}:')
        for display_name, split_name in (
            ('Train', 'train'),
            ('Validation', 'valid'),
            ('Test', 'test'),
        ):
            indices = torch.as_tensor(resolved_split[split_name]).detach().cpu()
            if indices.dtype == torch.bool:
                indices = torch.where(indices.reshape(-1))[0]
            else:
                indices = indices.reshape(-1).long()
            distribution = label_count_distribution(
                labels_cpu[indices],
                num_classes=num_classes,
            )
            print(f'  {display_name}: {distribution}')


def get_criterion_and_eval(args):
    dataset_name = canonicalize_dataset_name(args.dataset)
    bce_datasets = {'questions', 'ogbn-proteins'}
    criterion = (
        torch.nn.BCEWithLogitsLoss()
        if dataset_name in bce_datasets
        else torch.nn.NLLLoss()
    )

    if dataset_name == 'ogbn-proteins' and args.metric != 'rocauc':
        print('Overriding metric to rocauc for ogbn-proteins (multi-label task).')
        args.metric = 'rocauc'

    if args.metric == 'rocauc':
        from scaffold_gnn.utils.data_utils import eval_rocauc

        eval_func = eval_rocauc
    else:
        eval_func = eval_acc

    return criterion, eval_func, bce_datasets


def log_wandb_graph_stats(wb, wb_run, num_nodes, before_stats, after_stats, reduction, sparsifier_params):
    if wb_run is None:
        return
    try:
        b_dir, b_undir, b_self = before_stats
        a_dir, a_undir, a_self = after_stats
        wb_run.config.update({'sparsifier_params': sparsifier_params}, allow_val_change=True)
        wb.log(
            {
                'graph/num_nodes': num_nodes,
                'graph/edges_before_directed': b_dir,
                'graph/edges_before_undirected': b_undir,
                'graph/self_loops_before': b_self,
                'graph/avg_degree_before': 2 * b_undir / num_nodes if num_nodes > 0 else 0.0,
                'graph/edges_after_directed': a_dir,
                'graph/edges_after_undirected': a_undir,
                'graph/self_loops_after': a_self,
                'graph/avg_degree_after': 2 * a_undir / num_nodes if num_nodes > 0 else 0.0,
                'graph/reduction_pct': reduction,
            }
        )
    except Exception:
        pass


def log_wandb_epoch(wb, wb_run, args, run, epoch, train_acc, valid_acc, test_acc, valid_loss, train_f1, test_f1, loss, best_val, best_test):
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
        test_acc_mean = float(test_accs.mean().item()) if hasattr(test_accs.mean(), 'item') else float(test_accs.mean())
        test_acc_std = float(test_accs.std(unbiased=False).item()) if test_accs.numel() > 1 else 0.0
        payload = {
            'final/test_acc_mean': test_acc_mean,
            'final/test_acc_std': test_acc_std,
        }
        if test_f1s is not None:
            test_f1_mean = float(test_f1s.mean().item()) if hasattr(test_f1s.mean(), 'item') else float(test_f1s.mean())
            test_f1_std = float(test_f1s.std(unbiased=False).item()) if test_f1s.numel() > 1 else 0.0
            payload.update(
                {
                    'final/test_f1_macro_mean': test_f1_mean,
                    'final/test_f1_macro_std': test_f1_std,
                }
            )
        wb.log(payload)
    except Exception:
        pass


def parse_neighbor_fanouts(value, num_layers):
    text = str(value).strip().lower()
    if text in {'all', 'full', '-1'}:
        return [-1] * max(1, int(num_layers))
    fanouts = [int(part.strip()) for part in text.replace(';', ',').split(',') if part.strip()]
    if not fanouts:
        fanouts = [10]
    if len(fanouts) < num_layers:
        fanouts.extend([fanouts[-1]] * (num_layers - len(fanouts)))
    return fanouts[: max(1, int(num_layers))]


def build_neighbor_loader_data(dataset):
    data = Data(
        x=dataset.graph['node_feat'].detach().cpu(),
        edge_index=dataset.graph['edge_index'].detach().cpu(),
        y=dataset.label.detach().cpu(),
        num_nodes=dataset.graph['num_nodes'],
    )
    if dataset.graph.get('edge_weight') is not None:
        data.edge_weight = dataset.graph['edge_weight'].detach().cpu()
    return data


_NEIGHBOR_TMP_PATH_BUDGET_BYTES = 48


def resolve_neighbor_tmp_dir(requested_dir):
    """Return a short local temp path safe for multiprocessing AF_UNIX sockets.

    Python and DGL append generated ``pymp-*``/``listener-*`` components below
    TMPDIR.  Linux limits the complete AF_UNIX address to roughly 108 bytes, so
    a valid shared-cache path can still be too long for loader workers.  Keep
    persistent graph caches on shared storage, but remap worker-only files to a
    compact node-local directory when needed.
    """

    if not requested_dir:
        return None
    requested = os.path.abspath(os.path.expanduser(str(requested_dir)))
    if len(os.fsencode(requested)) <= _NEIGHBOR_TMP_PATH_BUDGET_BYTES:
        return requested

    digest = hashlib.sha256(os.fsencode(requested)).hexdigest()[:10]
    job_value = os.environ.get('SLURM_JOB_ID') or str(os.getpid())
    job_digest = hashlib.sha256(str(job_value).encode('utf-8')).hexdigest()[:8]
    local_root = os.path.abspath(os.path.expanduser(
        os.environ.get('SCAFFOLD_LOCAL_TMP_ROOT')
        or os.environ.get('SLURM_TMPDIR')
        or '/tmp'
    ))
    candidates = (
        os.path.join(local_root, f'es_{os.getuid()}_{job_digest}_{digest}'),
        os.path.join('/tmp', f'es_{os.getuid()}_{digest}'),
    )
    for candidate in candidates:
        if len(os.fsencode(candidate)) <= _NEIGHBOR_TMP_PATH_BUDGET_BYTES:
            return candidate
    raise RuntimeError(
        'Could not construct an AF_UNIX-safe neighbor-loader temporary path; '
        f'requested={requested!r}'
    )


def configure_neighbor_tmp_dir(args):
    requested_dir = getattr(args, 'neighbor_tmp_dir', None)
    tmp_dir = resolve_neighbor_tmp_dir(requested_dir)
    if not tmp_dir:
        return None
    os.makedirs(tmp_dir, exist_ok=True)
    os.environ['TMPDIR'] = tmp_dir
    os.environ['TEMP'] = tmp_dir
    os.environ['TMP'] = tmp_dir
    try:
        import tempfile

        tempfile.tempdir = None
    except Exception:
        pass
    args.neighbor_tmp_dir = tmp_dir
    print(
        '[NeighborTmp] '
        f'requested={os.path.abspath(os.path.expanduser(str(requested_dir)))} '
        f'resolved={tmp_dir} path_bytes={len(os.fsencode(tmp_dir))} '
        'af_unix_safe=true',
        flush=True,
    )
    return tmp_dir


def make_neighbor_loader(loader_data, input_nodes, args, *, batch_size, shuffle, train):
    from torch_geometric.loader import NeighborLoader

    worker_count = max(0, int(args.neighbor_workers))
    fanout_value = args.neighbor_num_neighbors
    if not train and str(getattr(args, 'neighbor_eval_num_neighbors', '')).strip():
        fanout_value = args.neighbor_eval_num_neighbors
    return NeighborLoader(
        loader_data,
        input_nodes=input_nodes.detach().cpu(),
        num_neighbors=parse_neighbor_fanouts(fanout_value, args.local_layers),
        batch_size=max(1, int(batch_size)),
        shuffle=shuffle,
        num_workers=worker_count,
        persistent_workers=worker_count > 0,
    )


def neighbor_train_epoch(model, loader_data, train_idx, args, criterion, dataset_name, bce_datasets, optimizer, device):
    loader = make_neighbor_loader(
        loader_data,
        train_idx,
        args,
        batch_size=args.neighbor_batch_size,
        shuffle=True,
        train=True,
    )
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        with amp_autocast(args, device):
            logits = model(
                batch.x,
                batch.edge_index,
                getattr(batch, 'edge_weight', None),
            )[: batch.batch_size]
            labels = batch.y[: batch.batch_size]

            if dataset_name in bce_datasets:
                if labels.ndim == 1 or labels.shape[-1] == 1:
                    labels = F.one_hot(labels.reshape(-1), labels.max() + 1)
                loss = criterion(logits.float(), labels.to(torch.float))
            else:
                loss = criterion(F.log_softmax(logits.float(), dim=1), labels.reshape(-1))

        loss.backward()
        optimizer.step()
        examples = int(batch.batch_size)
        total_loss += float(loss.item()) * examples
        total_examples += examples
    return total_loss / max(1, total_examples)


@torch.no_grad()
def neighbor_predict(model, loader_data, input_nodes, args, device, out_channels):
    loader = make_neighbor_loader(
        loader_data,
        input_nodes,
        args,
        batch_size=args.neighbor_eval_batch_size,
        shuffle=False,
        train=False,
    )
    model.eval()
    out = torch.empty((loader_data.num_nodes, out_channels), dtype=torch.float32)
    for batch in loader:
        seed_nodes = batch.n_id[: batch.batch_size].detach().cpu()
        batch = batch.to(device)
        with amp_autocast(args, device):
            logits = model(
                batch.x,
                batch.edge_index,
                getattr(batch, 'edge_weight', None),
            )[: batch.batch_size]
        logits = logits.detach().float().cpu()
        out[seed_nodes] = logits
    return out


def evaluate_neighbor(model, loader_data, dataset, split_idx, eval_func, criterion, args, dataset_name, bce_datasets, device, out_channels):
    split_cpu = {key: value.detach().cpu() for key, value in split_idx.items()}
    eval_nodes = torch.unique(torch.cat([split_cpu['train'], split_cpu['valid'], split_cpu['test']]))
    out = neighbor_predict(model, loader_data, eval_nodes, args, device, out_channels)
    labels = dataset.label.detach().cpu()

    train_acc = eval_func(labels[split_cpu['train']], out[split_cpu['train']])
    valid_acc = eval_func(labels[split_cpu['valid']], out[split_cpu['valid']])
    test_acc = eval_func(labels[split_cpu['test']], out[split_cpu['test']])

    if dataset_name in bce_datasets:
        if labels.ndim == 1 or labels.shape[1] == 1:
            true_label = F.one_hot(labels.reshape(-1), labels.max() + 1)
        else:
            true_label = labels
        valid_loss = criterion(out[split_cpu['valid']], true_label[split_cpu['valid']].to(torch.float))
    else:
        log_out = F.log_softmax(out, dim=1)
        valid_loss = criterion(log_out[split_cpu['valid']], labels.squeeze()[split_cpu['valid']])
    return train_acc, valid_acc, test_acc, valid_loss, out


def build_tunedgnn_protein_loader_data(dataset):
    """Build the CPU DGL graph used by tunedGNN-org's Proteins entrypoint."""

    import dgl

    edge_index = dataset.graph['edge_index'].detach().cpu()
    graph = dgl.graph(
        (edge_index[0], edge_index[1]),
        num_nodes=int(dataset.graph['num_nodes']),
    )
    if dataset.graph.get('edge_weight') is not None:
        graph.edata['edge_weight'] = dataset.graph['edge_weight'].detach().cpu()
    graph.create_formats_()
    return {
        'graph': graph,
        'x': dataset.graph['node_feat'].detach().cpu(),
        'y': dataset.label.detach().cpu(),
    }


def make_tunedgnn_protein_loaders(loader_data, train_idx, split_idx, args):
    """Match tunedGNN's layerwise DGL train/evaluation samplers exactly."""

    from dgl.dataloading import DataLoader, MultiLayerNeighborSampler

    train_fanouts = parse_neighbor_fanouts(
        args.neighbor_num_neighbors,
        args.local_layers,
    )
    eval_fanouts = parse_neighbor_fanouts(
        args.neighbor_eval_num_neighbors or args.neighbor_num_neighbors,
        args.local_layers,
    )
    graph = loader_data['graph']
    workers = max(0, int(args.neighbor_workers))
    train_loader = DataLoader(
        graph,
        train_idx.detach().cpu(),
        MultiLayerNeighborSampler(train_fanouts),
        batch_size=max(1, int(args.neighbor_batch_size)),
        shuffle=True,
        num_workers=workers,
    )
    eval_loader, split_cpu = make_tunedgnn_protein_eval_loader(
        loader_data,
        split_idx,
        args,
        eval_fanouts=eval_fanouts,
    )
    return train_loader, eval_loader, split_cpu


def make_tunedgnn_protein_eval_loader(
    loader_data,
    split_idx,
    args,
    *,
    eval_fanouts=None,
):
    """Build only the DGL evaluation loader for a selected graph topology."""

    from dgl.dataloading import DataLoader, MultiLayerNeighborSampler

    if eval_fanouts is None:
        eval_fanouts = parse_neighbor_fanouts(
            args.neighbor_eval_num_neighbors or args.neighbor_num_neighbors,
            args.local_layers,
        )
    split_cpu = {key: value.detach().cpu() for key, value in split_idx.items()}
    eval_nodes = torch.cat(
        [split_cpu['train'], split_cpu['valid'], split_cpu['test']],
    )
    eval_loader = DataLoader(
        loader_data['graph'],
        eval_nodes,
        MultiLayerNeighborSampler(eval_fanouts),
        batch_size=max(1, int(args.neighbor_eval_batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.neighbor_workers)),
    )
    return eval_loader, split_cpu


def tunedgnn_protein_train_epoch(
    model,
    loader_data,
    train_loader,
    criterion,
    optimizer,
    device,
    expected_examples=None,
):
    model.train()
    total_loss = 0.0
    total_examples = 0
    total_batches = 0
    labels = loader_data['y']
    features = loader_data['x']
    for input_nodes, output_nodes, blocks in train_loader:
        blocks = [block.to(device) for block in blocks]
        batch_x = features[input_nodes].to(device)
        batch_y = labels[output_nodes].to(device).float()
        optimizer.zero_grad()
        model_out = model.forward_blocks(blocks, batch_x)
        loss = criterion(model_out, batch_y)
        loss.backward()
        optimizer.step()
        examples = int(output_nodes.numel())
        total_loss += float(loss.item()) * examples
        total_examples += examples
        total_batches += 1
    if total_batches == 0 or total_examples == 0:
        raise RuntimeError(
            'OGBN-Proteins training loader produced zero batches/examples. '
            'Check DGL worker errors and the neighbor-loader temporary path.'
        )
    if expected_examples is not None and total_examples != int(expected_examples):
        raise RuntimeError(
            'OGBN-Proteins training loader coverage mismatch: '
            f'expected={int(expected_examples)} observed={total_examples} '
            f'batches={total_batches}'
        )
    average_loss = total_loss / total_examples
    if not math.isfinite(average_loss):
        raise RuntimeError(
            f'OGBN-Proteins training produced non-finite loss: {average_loss}'
        )
    if not loader_data.get('_train_coverage_logged', False):
        print(
            '[ProteinLoaderCoverage] phase=train '
            f'examples={total_examples} expected='
            f'{int(expected_examples) if expected_examples is not None else "unknown"} '
            f'batches={total_batches}',
            flush=True,
        )
        loader_data['_train_coverage_logged'] = True
    return average_loss


@torch.no_grad()
def evaluate_tunedgnn_protein(
    model,
    loader_data,
    eval_loader,
    split_idx,
    eval_func,
    criterion,
    device,
):
    model.eval()
    labels = loader_data['y']
    features = loader_data['x']
    predictions = torch.zeros(labels.shape, dtype=torch.float32, device=device)
    seen = torch.zeros(labels.shape[0], dtype=torch.bool)
    total_batches = 0
    for input_nodes, output_nodes, blocks in eval_loader:
        blocks = [block.to(device) for block in blocks]
        batch_x = features[input_nodes].to(device)
        output_nodes_cpu = output_nodes.detach().cpu()
        predictions[output_nodes_cpu.to(device)] = model.forward_blocks(blocks, batch_x)
        seen[output_nodes_cpu] = True
        total_batches += 1

    expected_nodes = torch.unique(torch.cat([
        split_idx['train'].detach().cpu(),
        split_idx['valid'].detach().cpu(),
        split_idx['test'].detach().cpu(),
    ]))
    missing_nodes = expected_nodes[~seen[expected_nodes]]
    if total_batches == 0 or missing_nodes.numel() > 0:
        raise RuntimeError(
            'OGBN-Proteins evaluation loader coverage failure: '
            f'batches={total_batches} expected_nodes={expected_nodes.numel()} '
            f'missing_nodes={missing_nodes.numel()}. Check DGL worker errors '
            'and the neighbor-loader temporary path.'
        )
    if not bool(torch.isfinite(predictions[expected_nodes.to(device)]).all()):
        raise RuntimeError('OGBN-Proteins evaluation produced non-finite predictions')
    if not loader_data.get('_eval_coverage_logged', False):
        print(
            '[ProteinLoaderCoverage] phase=eval '
            f'nodes={expected_nodes.numel()} expected={expected_nodes.numel()} '
            f'batches={total_batches}',
            flush=True,
        )
        loader_data['_eval_coverage_logged'] = True

    labels_device = labels.to(device)
    split_device = {key: value.to(device) for key, value in split_idx.items()}
    train_acc = eval_func(
        labels_device[split_device['train']],
        predictions[split_device['train']],
    )
    valid_acc = eval_func(
        labels_device[split_device['valid']],
        predictions[split_device['valid']],
    )
    test_acc = eval_func(
        labels_device[split_device['test']],
        predictions[split_device['test']],
    )
    valid_loss = criterion(
        predictions[split_device['valid']],
        labels_device[split_device['valid']].float(),
    )
    return train_acc, valid_acc, test_acc, valid_loss, predictions.detach().cpu()


def build_large_loader_data(dataset, split_idx):
    split_cpu = {key: value.detach().cpu() for key, value in split_idx.items()}
    data = Data(
        x=dataset.graph['node_feat'].detach().cpu(),
        edge_index=dataset.graph['edge_index'].detach().cpu(),
        y=dataset.label.detach().cpu(),
        num_nodes=dataset.graph['num_nodes'],
    )
    if dataset.graph.get('edge_weight') is not None:
        data.edge_weight = dataset.graph['edge_weight'].detach().cpu()
    for split_name, indices in split_cpu.items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[indices] = True
        setattr(data, f'{split_name}_mask', mask)
    return data


def make_random_node_loader(data, *, num_parts, shuffle, workers):
    from torch_geometric.loader import RandomNodeLoader

    worker_count = max(0, int(workers))
    # RandomNodeLoader is rebuilt for train/eval passes. Persistent worker
    # teardown is brittle with PyG/PyTorch multiprocessing on some clusters.
    return RandomNodeLoader(
        data,
        num_parts=max(1, int(num_parts)),
        shuffle=shuffle,
        num_workers=worker_count,
        persistent_workers=False,
    )


def _is_auto_part_count(value):
    return isinstance(value, str) and value.strip().lower() in {'auto', 'adaptive', 'default'}


def _resolve_large_random_node_parts(args, data, device, *, split):
    split = str(split)
    if split not in {'train', 'eval'}:
        raise ValueError("split must be 'train' or 'eval'")
    attr = f'large_{split}_parts'
    requested = getattr(args, attr)
    if not _is_auto_part_count(requested):
        return max(1, int(requested))

    cache_attr = f'_resolved_{attr}'
    cached = getattr(args, cache_attr, None)
    if cached is not None:
        return int(cached)

    n = int(getattr(data, 'num_nodes', 0))
    edge_index = getattr(data, 'edge_index', None)
    directed_edges = int(edge_index.size(1)) if torch.is_tensor(edge_index) else 0
    avg_directed_degree = directed_edges / max(1, n)
    x = getattr(data, 'x', None)
    feature_dim = int(x.size(-1)) if torch.is_tensor(x) and x.dim() > 1 else 1
    scalar_bytes = int(x.element_size()) if torch.is_tensor(x) and x.is_floating_point() else 4
    hidden = max(1, int(getattr(args, 'hidden_channels', 64)))
    layers = max(1, int(getattr(args, 'local_layers', 2)))
    class_count = int(getattr(data, 'y', torch.empty(0)).max().item() + 1) if torch.is_tensor(getattr(data, 'y', None)) and data.y.numel() else 1

    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        except TypeError:
            free_bytes, total_bytes = torch.cuda.mem_get_info(
                device.index if device.index is not None else torch.cuda.current_device()
            )
        mem_fraction = min(0.98, max(0.05, float(getattr(args, 'large_batch_mem_fraction', 0.85))))
        # RandomNodeLoader materializes induced subgraphs. Keep a real safety
        # margin for activations, gradients, optimizer state, and CUDA caching.
        budget_bytes = int(free_bytes * mem_fraction * 0.60)
    else:
        free_bytes = total_bytes = 0
        mem_fraction = float(getattr(args, 'large_batch_mem_fraction', 0.85))
        budget_bytes = 8 * 1024 ** 3

    activation_scalars = feature_dim + (layers + 2) * hidden * 10 + class_count * 4
    activation_bytes_per_node = activation_scalars * scalar_bytes
    edge_bytes_per_node = avg_directed_degree * 2 * 8 * 4
    bytes_per_node = max(1.0, activation_bytes_per_node + edge_bytes_per_node)
    target_nodes_per_part = max(1, int(budget_bytes / bytes_per_node))
    resolved = max(1, int(math.ceil(n / max(1, target_nodes_per_part))))

    if split == 'eval':
        train_cached = getattr(args, '_resolved_large_train_parts', None)
        if train_cached is not None:
            resolved = max(resolved, int(train_cached))

    setattr(args, cache_attr, int(resolved))
    if device.type == 'cuda' and torch.cuda.is_available():
        free_gb = free_bytes / (1024 ** 3)
        total_gb = total_bytes / (1024 ** 3)
        budget_gb = budget_bytes / (1024 ** 3)
        print(
            f'[adaptive RandomNodeLoader] split={split} '
            f'cuda_free={free_gb:.1f}GB/{total_gb:.1f}GB '
            f'mem_fraction={mem_fraction:.2f} budget={budget_gb:.1f}GB '
            f'est_bytes_per_node={bytes_per_node:.0f} '
            f'parts={resolved} nodes_per_part~{math.ceil(n / resolved)}',
            flush=True,
        )
    else:
        print(
            f'[adaptive RandomNodeLoader] split={split} CUDA unavailable; '
            f'parts={resolved} nodes_per_part~{math.ceil(n / resolved)}',
            flush=True,
        )
    return int(resolved)


def large_random_node_train_epoch(model, loader_data, args, criterion, dataset_name, bce_datasets, optimizer, device):
    train_parts = _resolve_large_random_node_parts(args, loader_data, device, split='train')
    loader = make_random_node_loader(
        loader_data,
        num_parts=train_parts,
        shuffle=True,
        workers=getattr(args, 'large_loader_workers', 0),
    )
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        if not bool(batch.train_mask.any()):
            continue
        batch = batch.to(device)
        optimizer.zero_grad()
        with amp_autocast(args, device):
            logits = model(
                batch.x,
                batch.edge_index,
                getattr(batch, 'edge_weight', None),
            )

            if dataset_name in bce_datasets:
                labels = batch.y
                if labels.ndim == 1 or labels.shape[-1] == 1:
                    labels = F.one_hot(labels.reshape(-1), labels.max() + 1)
                loss = criterion(logits[batch.train_mask].float(), labels[batch.train_mask].to(torch.float))
            else:
                loss = criterion(
                    F.log_softmax(logits[batch.train_mask].float(), dim=1),
                    batch.y.reshape(-1)[batch.train_mask],
                )

        loss.backward()
        optimizer.step()
        examples = int(batch.train_mask.sum().item())
        total_loss += float(loss.item()) * examples
        total_examples += examples
    return total_loss / max(1, total_examples)


@torch.no_grad()
def evaluate_large_random_node(model, loader_data, args, eval_func, criterion, dataset_name, bce_datasets, device):
    if getattr(args, 'large_eval_mode', 'partition') == 'full':
        from types import SimpleNamespace

        split = {name: getattr(loader_data, f'{name}_mask').nonzero().flatten()
                 for name in ('train', 'valid', 'test')}
        dataset = SimpleNamespace(
            graph={'node_feat': loader_data.x, 'edge_index': loader_data.edge_index,
                   'edge_weight': getattr(loader_data, 'edge_weight', None),
                   'num_nodes': loader_data.num_nodes}, label=loader_data.y)
        metrics, logits = evaluate_full_batch_outputs(
            model, dataset, split, loader_data.edge_index,
            getattr(loader_data, 'edge_weight', None), eval_func, criterion, args,
            device, evaluation_feature_decomposition)
        labels = loader_data.y.detach().cpu()
        logits = logits.detach().cpu()
        truth = {name: labels[idx.cpu()] for name, idx in split.items()}
        predictions = {name: logits[idx.cpu()] for name, idx in split.items()}
        return (*metrics, (truth, predictions))
    eval_parts = _resolve_large_random_node_parts(args, loader_data, device, split='eval')
    loader = make_random_node_loader(
        loader_data,
        num_parts=eval_parts,
        shuffle=False,
        workers=getattr(args, 'large_loader_workers', 0),
    )
    model.eval()
    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    with evaluation_feature_decomposition(model, args, device):
        for batch in loader:
            batch = batch.to(device)
            with amp_autocast(args, device):
                logits = model(
                    batch.x,
                    batch.edge_index,
                    getattr(batch, 'edge_weight', None),
                )
            logits = logits.detach().float().cpu()
            labels = batch.y.detach().cpu()
            for split_name in ('train', 'valid', 'test'):
                mask = getattr(batch, f'{split_name}_mask').detach().cpu()
                if bool(mask.any()):
                    y_true[split_name].append(labels[mask])
                    y_pred[split_name].append(logits[mask])

    merged_true = {key: torch.cat(value, dim=0) for key, value in y_true.items()}
    merged_pred = {key: torch.cat(value, dim=0) for key, value in y_pred.items()}
    train_acc = eval_func(merged_true['train'], merged_pred['train'])
    valid_acc = eval_func(merged_true['valid'], merged_pred['valid'])
    test_acc = eval_func(merged_true['test'], merged_pred['test'])

    if dataset_name in bce_datasets:
        labels = merged_true['valid']
        if labels.ndim == 1 or labels.shape[-1] == 1:
            labels = F.one_hot(labels.reshape(-1), labels.max() + 1)
        valid_loss = criterion(merged_pred['valid'], labels.to(torch.float))
    else:
        valid_loss = criterion(F.log_softmax(merged_pred['valid'], dim=1), merged_true['valid'].reshape(-1))
    return train_acc, valid_acc, test_acc, valid_loss, (merged_true, merged_pred)


def _is_auto_batch_size(value):
    return isinstance(value, str) and value.strip().lower() in {'auto', 'adaptive'}


def _resolve_large_batch_size(args, dataset, device):
    requested = getattr(args, 'large_batch_size', 'auto')
    n = int(dataset.graph['num_nodes'])
    if not _is_auto_batch_size(requested):
        return min(n, max(1, int(requested)))

    cached = getattr(args, '_resolved_large_batch_size', None)
    if cached is not None:
        return int(cached)

    min_size = max(1, int(getattr(args, 'large_batch_min_size', 1024)))
    max_size = int(getattr(args, 'large_batch_max_size', 0))
    if max_size <= 0:
        max_size = n
    max_size = min(n, max(1, max_size))

    if device.type != 'cuda' or not torch.cuda.is_available():
        batch_size = min(max_size, max(min_size, min(n, 100000)))
        args._resolved_large_batch_size = int(batch_size)
        print(f'[adaptive batch] CUDA unavailable; using node_batch_size={batch_size}')
        return int(batch_size)

    torch.cuda.empty_cache()
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except TypeError:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device.index if device.index is not None else torch.cuda.current_device())

    mem_fraction = float(getattr(args, 'large_batch_mem_fraction', 0.85))
    mem_fraction = min(0.98, max(0.05, mem_fraction))
    budget_bytes = max(1, int(free_bytes * mem_fraction))

    x = dataset.graph['node_feat']
    feature_dim = int(x.size(-1)) if torch.is_tensor(x) and x.dim() > 1 else 1
    scalar_bytes = int(x.element_size()) if torch.is_tensor(x) and x.is_floating_point() else 4
    hidden = max(1, int(getattr(args, 'hidden_channels', 64)))
    layers = max(1, int(getattr(args, 'local_layers', 2)))
    num_classes = max(1, int(getattr(dataset, 'num_classes', 1)))
    edge_index = dataset.graph['edge_index']
    directed_edges = int(edge_index.size(1)) if torch.is_tensor(edge_index) else 0
    avg_directed_degree = directed_edges / max(1, n)

    # Conservative bytes-per-node estimate for GNN activations, gradients,
    # optimizer scratch, and induced edge_index materialization.
    activation_scalars = feature_dim + (layers + 2) * hidden * 8 + num_classes * 4
    activation_bytes_per_node = activation_scalars * scalar_bytes
    edge_bytes_per_node = avg_directed_degree * 2 * 8 * 4
    bytes_per_node = max(1.0, activation_bytes_per_node + edge_bytes_per_node)
    estimated = int(budget_bytes / bytes_per_node)
    batch_size = min(max_size, max(min_size, estimated))
    args._resolved_large_batch_size = int(batch_size)

    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    budget_gb = budget_bytes / (1024 ** 3)
    print(
        '[adaptive batch] '
        f'cuda_free={free_gb:.1f}GB/{total_gb:.1f}GB '
        f'mem_fraction={mem_fraction:.2f} budget={budget_gb:.1f}GB '
        f'est_bytes_per_node={bytes_per_node:.0f} '
        f'node_batch_size={batch_size}/{n}'
    )
    return int(batch_size)


def large_node_batch_train_epoch(model, dataset, train_idx, args, criterion, dataset_name, bce_datasets, optimizer, device):
    edge_index = dataset.graph['edge_index'].detach().cpu()
    edge_weight = dataset.graph.get('edge_weight')
    if edge_weight is not None:
        edge_weight = edge_weight.detach().cpu()
    x = dataset.graph['node_feat'].detach().cpu()
    labels = dataset.label.detach().cpu()
    n = dataset.graph['num_nodes']
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx.detach().cpu()] = True
    batch_size = _resolve_large_batch_size(args, dataset, device)
    total_loss = 0.0
    total_batches = 0

    for idx_i in torch.randperm(n).split(batch_size):
        train_mask_i = train_mask[idx_i]
        if not bool(train_mask_i.any()):
            continue
        edge_index_i, edge_weight_i = subgraph(
            idx_i,
            edge_index,
            edge_attr=edge_weight,
            num_nodes=n,
            relabel_nodes=True,
        )
        x_i = x[idx_i].to(device)
        y_i = labels[idx_i].to(device)
        mask_i = train_mask_i.to(device)
        edge_index_i = edge_index_i.to(device)
        if edge_weight_i is not None:
            edge_weight_i = edge_weight_i.to(device)

        optimizer.zero_grad()
        with amp_autocast(args, device):
            logits = model(x_i, edge_index_i, edge_weight_i)
            if dataset_name in bce_datasets:
                if y_i.ndim == 1 or y_i.shape[-1] == 1:
                    y_i = F.one_hot(y_i.reshape(-1), y_i.max() + 1)
                loss = criterion(logits[mask_i].float(), y_i[mask_i].to(torch.float))
            else:
                loss = criterion(F.log_softmax(logits[mask_i].float(), dim=1), y_i.reshape(-1)[mask_i])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        total_batches += 1
    return total_loss / max(1, total_batches)


def evaluate_large_full(model, dataset, split_idx, eval_func, criterion, args, device):
    def colocated():
        # Moves features, edges, labels and the split onto one device, runs the
        # exact forward, falls back to CPU on CUDA OOM, and restores the
        # training placement afterwards.
        metrics, outputs = evaluate_full_batch_outputs(
            model, dataset, split_idx, dataset.graph['edge_index'], dataset.graph.get('edge_weight'),
            eval_func, criterion, args, device, evaluation_feature_decomposition)
        return (*metrics, outputs)

    if getattr(args, 'scaffold_final_report_best_checkpoint', False):
        return colocated()
    if not bool(getattr(args, 'large_eval_cpu', True)):
        # evaluate() reads dataset.graph in place, so it is only safe once the
        # graph already sits where the model does -- true for the 'full'
        # pipeline (arxiv, reddit below the ratio cut), which is left on its
        # original path. node_batch and random_node_loader deliberately park
        # features, edges and labels on CPU while the model trains on GPU, and
        # there the in-place read would meet CPU tensors with a CUDA model.
        if dataset.graph['node_feat'].device == device:
            return evaluate(model, dataset, split_idx, eval_func, criterion, args)
        return colocated()
    split_cpu = {key: value.detach().cpu() for key, value in split_idx.items()}
    return evaluate_cpu(model, dataset, split_cpu, eval_func, criterion, args, device)


def training_epoch_count(args, tunedgnn_contract):
    if getattr(args, 'exact_epochs', False):
        return int(args.epochs)
    if getattr(args, 'scaffold_exact_epochs', False):
        if args.sparsifier not in ('scaffold_sample', 'scaffold_fast', 'scaffold_batch'):
            raise ValueError('--scaffold_exact_epochs is restricted to Scaffold campaigns')
        return int(args.epochs)
    return int(args.epochs) - int(bool(tunedgnn_contract and args.model_profile == 'products'))


def reset_model_for_run(model, args, run):
    """Independent initialization, including the ProductsGNN/Reddit profile."""
    set_seed_for_run(args, run)
    model.reset_parameters()


# Charged to 'eval': anything that scores a model rather than fitting it.
evaluate = timed_phase('eval', evaluate)
evaluate_cpu = timed_phase('eval', evaluate_cpu)
evaluate_neighbor = timed_phase('eval', evaluate_neighbor)
evaluate_tunedgnn_protein = timed_phase('eval', evaluate_tunedgnn_protein)
evaluate_large_random_node = timed_phase('eval', evaluate_large_random_node)
evaluate_large_full = timed_phase('eval', evaluate_large_full)
# Charged to 'selection': Scaffold-K's per-epoch support bookkeeping -- the
# validation-best bank (which also rewrites its manifest to disk) and the
# K-view union built for the diagnostic union evaluation.
# Charged to 'train': the mini-batch pipelines keep a whole training epoch
# inside a helper, so the in-loop bracket -- which ends at the full-batch
# optimiser.step() -- never reaches them. reddit (random_node_loader) and
# ogbn-products run here, and without this their train_only would fall back to
# the subtraction that Scaffold-K is known to break.
neighbor_train_epoch = timed_phase('train', neighbor_train_epoch)
tunedgnn_protein_train_epoch = timed_phase('train', tunedgnn_protein_train_epoch)
large_random_node_train_epoch = timed_phase('train', large_random_node_train_epoch)
large_node_batch_train_epoch = timed_phase('train', large_node_batch_train_epoch)
union_sparse_graph_views = timed_phase('selection', union_sparse_graph_views)
update_best_sparse_graph_bank = timed_phase(
    'selection', update_best_sparse_graph_bank)


def main():
    parser = argparse.ArgumentParser(description='Graph Sparsification Experiments')
    parser = parser_add_main_args(parser)
    parser = apply_scaffold_config_defaults(parser)
    args = parser.parse_args()
    final_eval_views = validate_final_views(args)
    dataset_name = canonicalize_dataset_name(args.dataset)
    print_tunedgnn_scaffold_contract(args)

    device = get_device(args)
    gpu_profile, device_name, total_gb = configure_gpu_memory_profile(args, device)
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'CUDA Device: {device_name} ({total_gb:.1f} GB)')
    print(
        f'GPU Memory Profile: requested={args.gpu_memory_profile} resolved={gpu_profile} '
        f'large_batch_mem_fraction={args.large_batch_mem_fraction:.2f} '
        f'large_eval_cpu={args.large_eval_cpu} '
        f'amp_bf16={use_amp(args, device)}'
    )
    set_seed_from_args(args)
    wb, wb_run = setup_wandb(args)

    print(f'Loading dataset: {dataset_name}...')
    print(f'[DatasetSource] dataset={dataset_name} data_root={os.path.abspath(args.data_dir)} method={args.sparsifier}')
    try:
        if baseline_partition_enabled():
            dataset = load_baseline_partitioned_dataset(args, dataset_name)
        else:
            dataset = load_dataset(args.data_dir, args.dataset)
    except FileNotFoundError as exc:
        print(f'Dataset load failed for {dataset_name}: {exc}', flush=True)
        raise SystemExit(2) from None
    print(f'Dataset {dataset_name} loaded successfully.')

    graph, label = dataset[0]
    if not hasattr(dataset, 'label'):
        dataset.label = label
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    if bool(getattr(args, 'dataset_load_only', False)):
        if not hasattr(dataset, 'num_classes'):
            dataset.num_classes = max(dataset.label.max().item() + 1, dataset.label.shape[1])
        split_idx = get_split_indices(dataset, args, torch.device('cpu'))
        print_split_stats(split_idx, args)
        print_split_class_distributions(dataset.label, split_idx, dataset.num_classes)
        split_for_hash = split_idx[0] if isinstance(split_idx, list) else split_idx
        split_hash = canonical_split_fingerprint(split_for_hash)
        print(f'[DatasetSplit] fingerprint={split_hash}', flush=True)

        num_nodes = int(graph['num_nodes'])

        def as_mask(value):
            value = torch.as_tensor(value).detach().cpu()
            if value.dtype == torch.bool:
                return value.reshape(-1).clone()
            mask = torch.zeros(num_nodes, dtype=torch.bool)
            mask[value.reshape(-1).long()] = True
            return mask

        contract_y = dataset.label.detach().cpu()
        if contract_y.dim() == 2 and contract_y.size(1) == 1:
            contract_y = contract_y.view(-1)
        contract_data = Data(
            x=torch.as_tensor(graph['node_feat']).detach().cpu(),
            edge_index=torch.as_tensor(graph['edge_index']).detach().cpu(),
            y=contract_y,
            num_nodes=num_nodes,
        )
        contract_data.train_mask = as_mask(split_for_hash['train'])
        contract_data.val_mask = as_mask(split_for_hash['valid'])
        contract_data.test_mask = as_mask(split_for_hash['test'])

        from scaffold_gnn.utils.dataset_smoke import dataset_fingerprint, exact_homophily

        fingerprint = dataset_fingerprint(contract_data)
        num_features = int(contract_data.x.size(1))
        print(
            '[DatasetLoadOnly] '
            f'status=PASS method={args.sparsifier} dataset={dataset_name} '
            f'data_root={os.path.abspath(args.data_dir)} nodes={num_nodes} '
            f'edges={int(contract_data.edge_index.size(1))} features={num_features} '
            f'classes_or_tasks={int(dataset.num_classes)} '
            f'train={int(contract_data.train_mask.sum())} '
            f'valid={int(contract_data.val_mask.sum())} '
            f'test={int(contract_data.test_mask.sum())} '
            f'fingerprint={fingerprint}',
            flush=True,
        )
        if bool(getattr(args, 'dataset_load_homophily', False)):
            node_h, edge_h, definition = exact_homophily(contract_data)
            print(
                '[DatasetHomophily] '
                f'dataset={dataset_name} node={node_h:.6f} edge={edge_h:.6f} '
                f'definition={definition}',
                flush=True,
            )
        if wb_run is not None:
            wb_run.finish()
        return

    tensor_scaffold = use_tensor_scaffold(args)
    large_requested = use_large_graph_training(args, dataset_name)
    large_pipeline = select_large_graph_pipeline(args, dataset_name) if large_requested else 'full'
    guard_large_networkx_scaffold(args, dataset_name, large_requested, tensor_scaffold)
    neighbor_requested = (large_requested and large_pipeline == 'neighbor') or (not large_requested and use_neighbor_training(args))
    large_cpu_graph = large_requested and large_pipeline in {'node_batch', 'random_node_loader', 'neighbor'}
    edge_device = torch.device('cpu') if neighbor_requested or large_cpu_graph else device
    feature_device = torch.device('cpu') if neighbor_requested or large_cpu_graph else device
    graph['node_feat'] = graph['node_feat'].to(feature_device)
    if not tensor_scaffold and not neighbor_requested and not large_cpu_graph:
        graph['edge_index'] = graph['edge_index'].to(device)
    label_device = torch.device('cpu') if large_cpu_graph else device
    label = label.to(label_device)
    dataset.label = dataset.label.to(label_device)

    if not hasattr(dataset, 'num_classes'):
        dataset.num_classes = max(dataset.label.max().item() + 1, dataset.label.shape[1])

    split_idx = get_split_indices(dataset, args, device)
    print_split_stats(split_idx, args)
    print_split_class_distributions(dataset.label, split_idx, dataset.num_classes)
    split_for_hash = split_idx[0] if isinstance(split_idx, list) else split_idx
    split_fingerprint = canonical_split_fingerprint(split_for_hash)
    print(
        f"[DatasetSplit] fingerprint={split_fingerprint}",
        flush=True,
    )

    n = dataset.graph['num_nodes']
    raw_edge_index = dataset.graph['edge_index']
    guard_large_networkx_scaffold(
        args,
        dataset_name,
        large_requested,
        tensor_scaffold,
        num_nodes=n,
        num_directed_edges=int(raw_edge_index.size(1)),
    )
    raw_undirected_unique = bool(graph.get('edge_index_is_undirected_unique', False))
    raw_symmetric_fastpath = use_symmetric_scaffold_fastpath(args, graph)
    direct_undirected_passthrough = raw_undirected_unique and (
        args.sparsifier == 'full'
        or str(getattr(args, 'sparsifier', '')).lower()
        in (SCAFFOLD_METHODS | BASIC_SINGLE_GRAPH_SPARSIFIERS)
    )
    if raw_symmetric_fastpath:
        raw_stats = fast_symmetric_edge_stats(raw_edge_index)
        print(
            '[GraphFastPath] using symmetric edge fast-path: '
            'skipping global torch.unique/coalesce before sparsification',
            flush=True,
        )
    else:
        raw_stats = edge_stats_with_metadata(
            raw_edge_index,
            undirected_unique=raw_undirected_unique,
        )
    if tensor_scaffold or direct_undirected_passthrough:
        sparsifier_edge_index = raw_edge_index.cpu()
        norm_info = {
            'self_loops_removed': 0,
            'directed_before_coalesce': raw_stats[0],
            'directed_after_coalesce': raw_stats[0],
            'coalesce_removed_directed': 0,
        }
    elif raw_symmetric_fastpath:
        sparsifier_edge_index = symmetric_to_undirected_unique(raw_edge_index.cpu()).to(edge_device)
        norm_info = {
            'self_loops_removed': raw_stats[2],
            'directed_before_coalesce': raw_stats[0],
            'directed_after_coalesce': 2 * raw_stats[1],
            'coalesce_removed_directed': 0,
        }
    else:
        raw_edge_index = raw_edge_index.to(edge_device)
        sparsifier_edge_index, norm_info = normalize_undirected(raw_edge_index, n, return_info=True)
        sparsifier_edge_index = sparsifier_edge_index.to(edge_device)
    evaluation_graph = resolve_evaluation_graph(args)
    multiview_original_edges = None
    if final_eval_views:
        if raw_symmetric_fastpath:
            multiview_original_edges = raw_edge_index.cpu()
        elif raw_undirected_unique:
            multiview_original_edges = direct_symmetric_from_undirected_unique(sparsifier_edge_index)
        else:
            multiview_original_edges = normalize_undirected(sparsifier_edge_index.cpu(), n)
        if uses_tunedgnn_scaffold_contract(args) and args.model_profile != 'proteins':
            multiview_original_edges = with_tunedgnn_self_loops(multiview_original_edges, n)
        print('[ScaffoldMultiView] enabled final_only=true views=' + ','.join(final_eval_views)
              + ' training_graph=sparse model_selection=final_epoch bank=validation_best_distinct', flush=True)
    fixed_eval_edge_index = None
    if evaluation_graph == 'original':
        fixed_eval_edge_index = normalize_undirected(
            sparsifier_edge_index.to(edge_device),
            n,
        ).to(edge_device)
        if uses_tunedgnn_scaffold_contract(args):
            fixed_eval_edge_index = with_tunedgnn_self_loops(
                fixed_eval_edge_index,
                n,
            )
        print(
            '[EvaluationGraph] topology=original-full '
            f'directed_edges={int(fixed_eval_edge_index.size(1))} '
            'edge_weight=unweighted training_topology=sparsified',
            flush=True,
        )
    elif evaluation_graph == 'fast-maxst':
        # Built below after the source PyG object has been assembled.
        pass
    else:
        print(
            '[EvaluationGraph] topology=sparse-training-graph',
            flush=True,
        )
    dataset.graph['node_feat'] = dataset.graph['node_feat'].to(feature_device)

    before_stats = edge_stats_with_metadata(
        sparsifier_edge_index,
        undirected_unique=(
            ((tensor_scaffold or direct_undirected_passthrough) and raw_undirected_unique)
            or (raw_symmetric_fastpath and not tensor_scaffold)
        ),
        symmetric_unique=raw_symmetric_fastpath and tensor_scaffold,
        num_undirected_edges=raw_stats[1] if raw_symmetric_fastpath else None,
        num_self_loops=raw_stats[2] if raw_symmetric_fastpath and tensor_scaffold else None,
    )
    r_dir, r_undir, r_self = raw_stats
    b_dir, b_undir, b_self = before_stats

    print('-' * 50)
    print(f'Sparsification: {args.sparsifier}')
    print('Raw Input Graph (before normalize_undirected):')
    print(f'  Total Directed Edges:   {r_dir}')
    print(f'  Unique Undirected Edges: {r_undir}')
    print(f'  Self Loops:             {r_self}')
    print(f'  Average Degree:         {2 * r_undir / n:.2f}')
    print('Graph AFTER normalize_undirected (input to sparsifier):')
    print('  normalize_undirected details:')
    print(f"    self_loops_removed:      {norm_info['self_loops_removed']}")
    print(f"    directed_before_coalesce:{norm_info['directed_before_coalesce']}")
    print(f"    directed_after_coalesce: {norm_info['directed_after_coalesce']}")
    print(f"    coalesce_removed_directed: {norm_info['coalesce_removed_directed']}")
    # Remembered so the diagnostic union can report coverage of the ORIGINAL
    # graph. At union time dataset.graph holds the sparse training view, so the
    # dense edge count is no longer reachable from there.
    global _ORIGINAL_EDGE_COUNTS
    _ORIGINAL_EDGE_COUNTS = {'directed': int(b_dir), 'undirected': int(b_undir)}
    print('Edges BEFORE sparsification:')
    print(f'  Total Directed Edges:   {b_dir}')
    print(f'  Unique Undirected Edges: {b_undir}')
    print(f'  Self Loops:             {b_self}')
    print(f'  Average Degree:         {2 * b_undir / n:.2f}')

    sparsifier_params = get_sparsifier_params(args, num_nodes=n, num_edges=b_undir)
    print(f'Sparsifier Config:      {sparsifier_params}')

    sparsifier = get_sparsifier(args.sparsifier, **sparsifier_params)
    if 'parallel_workers' in sparsifier_params and hasattr(sparsifier, 'parallel_workers'):
        print(
            '[Sparsifier] parallel_workers '
            f"requested={sparsifier_params.get('parallel_workers')} "
            f"resolved={getattr(sparsifier, 'parallel_workers')}",
            flush=True,
        )
    pyg_data = Data(
        x=dataset.graph['node_feat'],
        edge_index=sparsifier_edge_index,
        y=label,
        num_nodes=dataset.graph['num_nodes'],
    )
    if raw_undirected_unique and (tensor_scaffold or direct_undirected_passthrough):
        pyg_data.edge_index_is_undirected_unique = True
    elif raw_symmetric_fastpath:
        if tensor_scaffold:
            pyg_data.edge_index_is_symmetric_unique = True
            pyg_data.num_undirected_edges = b_undir
            pyg_data.num_self_loops = b_self
        else:
            pyg_data.edge_index_is_undirected_unique = True
    elif not tensor_scaffold and not direct_undirected_passthrough:
        pyg_data.edge_index_is_symmetric_unique = True
        pyg_data.num_undirected_edges = b_undir
    if evaluation_graph == 'fast-maxst':
        fixed_eval_edge_index = build_fixed_fast_maxst_evaluation_graph(
            pyg_data,
            args,
            edge_device,
        )
        fixed_non_loop_edges = int(
            (fixed_eval_edge_index[0] != fixed_eval_edge_index[1]).sum().item()
        )
        print(
            '[EvaluationGraph] topology=fixed-fast-maxst '
            f'directed_edges={int(fixed_eval_edge_index.size(1))} '
            f'undirected_non_loop_edges={fixed_non_loop_edges // 2} '
            'weight_method=cosine training_topology=resparsified',
            flush=True,
        )
    cache_target_ratio = (
        1.0
        if args.sparsifier in {'mst', 'fast_maxst'}
        else args.target_ratio
    )
    sparse_cache_entry = cache_entry(
        getattr(args, 'sparsified_graph_cache_dir', None),
        dataset=dataset_name,
        sparsifier=args.sparsifier,
        target_ratio=cache_target_ratio,
        seed=args.seed,
        split_fingerprint=split_fingerprint,
        source_edge_index=sparsifier_edge_index,
        num_nodes=n,
        num_undirected_edges=b_undir,
        sparsifier_params=sparsifier_params,
    )
    if sparse_cache_entry is not None:
        sparsifier.networkit_cache_path = str(sparse_cache_entry['networkit_path'])
    async_resparsify_source_payload = None
    if bool(getattr(args, 'scaffold_async_resparsify', False)):
        async_resparsify_source_payload = _build_async_resparsify_source_payload(
            pyg_data,
            include_features=(
                str(getattr(args, 'scaffold_support_weight_method', 'uniform')) != 'uniform'
            ),
        )
    training_rng_state = (
        capture_training_rng_state()
        if uses_tunedgnn_scaffold_contract(args)
        else None
    )
    synchronize_timing_device(device)
    sparsification_start = time.perf_counter()
    # A cached initial sparse graph bypasses sparsifier.sparsify(), but both
    # forced training refreshes and fresh evaluation ensembles still need the
    # original pre-sparsification graph. Cache it unconditionally before the
    # disk-cache branch so resparsify() works on hits as well as misses.
    if hasattr(sparsifier, 'cache_source'):
        sparsifier.cache_source(pyg_data)
    cached = None
    if not bool(getattr(args, 'recompute_sparsified_graph', False)):
        cached = load_cached_graph(sparse_cache_entry, pyg_data)
    cache_read_time_sec = 0.0
    if cached is not None:
        sparsified_data, sparse_cache_metadata = cached
        synchronize_timing_device(device)
        cache_read_time_sec = time.perf_counter() - sparsification_start
        initial_sparsification_time_sec = float(
            sparse_cache_metadata['sparsification_time_sec']
        )
        print(
            '[SparseGraphCache] hit '
            f"path={sparse_cache_entry['graph_path']} "
            f'original_sparsification_time={initial_sparsification_time_sec:.6f}s '
            f'cache_read_time={cache_read_time_sec:.6f}s',
            flush=True,
        )
    else:
        sparsified_data = sparsifier.sparsify(pyg_data)
        _attach_async_resparsify_support_cache(async_resparsify_source_payload, sparsifier)
    if training_rng_state is not None:
        restore_training_rng_state(training_rng_state)

    sparsified_edge_index, sparsified_edge_weight, after_stats = _finalize_sparsified_edge_index(
        sparsified_data, args, edge_device, n, tensor_scaffold
    )
    synchronize_timing_device(device)
    if cached is None:
        initial_sparsification_time_sec = time.perf_counter() - sparsification_start
        sparse_cache_metadata = save_cached_graph(
            sparse_cache_entry,
            sparsified_data,
            sparsification_time_sec=initial_sparsification_time_sec,
            before_stats=before_stats,
            after_stats=after_stats,
        )
        if sparse_cache_entry is not None:
            print(
                '[SparseGraphCache] saved '
                f"path={sparse_cache_entry['graph_path']} "
                f'sparsification_time={initial_sparsification_time_sec:.6f}s',
                flush=True,
            )
    a_dir, a_undir, a_self = after_stats
    args.achieved_kept_ratio = (a_undir / b_undir) if b_undir > 0 else 0.0

    reduction = (1 - (a_undir / b_undir)) * 100 if b_undir > 0 else 0
    print('Edges AFTER sparsification:')
    print(f'  Total Directed Edges:   {a_dir}')
    print(f'  Unique Undirected Edges: {a_undir}')
    print(f'  Self Loops:             {a_self}')
    print(f'  Average Degree:         {2 * a_undir / n:.2f}')
    print(f'  Reduction:              {reduction:.2f}%')
    print('-' * 50)

    log_wandb_graph_stats(wb, wb_run, n, before_stats, after_stats, reduction, sparsifier_params)

    dataset.graph['edge_index'] = sparsified_edge_index
    if sparsified_edge_weight is None:
        dataset.graph.pop('edge_weight', None)
    else:
        dataset.graph['edge_weight'] = sparsified_edge_weight
        print(
            '[WeightedGCN] enabled=true '
            f'weights={sparsified_edge_weight.numel()} '
            f'min={float(sparsified_edge_weight.min()):.6g} '
            f'max={float(sparsified_edge_weight.max()):.6g}',
            flush=True,
        )
    n = dataset.graph['num_nodes']
    c = dataset.num_classes
    d = dataset.graph['node_feat'].shape[1]
    training_directed_edges = int(sparsified_edge_index.size(1))
    training_self_loops = int(
        (sparsified_edge_index[0] == sparsified_edge_index[1]).sum().item()
    )
    print(
        f'GNN Input Graph (post-sparsify): '
        f'nodes={n}, directed_edges={training_directed_edges}, '
        f'undirected_non_loop_edges={a_undir}, self_loops={training_self_loops}, '
        f'feature_dim={d}'
    )

    tunedgnn_contract = uses_tunedgnn_scaffold_contract(args)
    tunedgnn_protein_training = (
        tunedgnn_contract
        and str(getattr(args, 'model_profile', '')).lower() == 'proteins'
        and neighbor_requested
    )
    # tunedGNN's Proteins entrypoint constructs a fresh model inside each run,
    # after reseeding and creating its DGL loaders. Other profiles construct
    # once here and follow their original reset policy below.
    model = None if tunedgnn_protein_training else parse_method(args, n, c, d, device)
    criterion, eval_func, bce_datasets = get_criterion_and_eval(args)
    neighbor_training = neighbor_requested
    if getattr(args, 'eval_only_checkpoint', None):
        if model is None or neighbor_training or large_requested:
            raise ValueError(
                '--eval_only_checkpoint currently supports full-batch models only'
            )
        eval_run = int(getattr(args, 'eval_only_run', 1))
        if eval_run < 1:
            raise ValueError('--eval_only_run is one-based and must be at least 1')
        if isinstance(split_idx, list):
            if eval_run > len(split_idx):
                raise ValueError(
                    f'--eval_only_run {eval_run} exceeds {len(split_idx)} splits'
                )
            eval_split = split_idx[eval_run - 1]
        else:
            eval_split = split_idx
        checkpoint_path = os.path.abspath(
            os.path.expanduser(str(args.eval_only_checkpoint))
        )
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True,
            )
        except TypeError:  # pragma: no cover - older torch.
            checkpoint = torch.load(checkpoint_path, map_location=device)
        state = (
            checkpoint.get('model_state_dict')
            if isinstance(checkpoint, dict) else None
        )
        if state is None:
            raise ValueError(
                f'{checkpoint_path}: missing model_state_dict checkpoint entry'
            )
        model.load_state_dict(state)
        invalidate_graph_dependent_model_caches(model)
        synchronize_timing_device(device)
        inference_start = time.perf_counter()
        eval_result = evaluate(
            model,
            dataset,
            eval_split,
            eval_func,
            criterion,
            args,
        )
        synchronize_timing_device(device)
        inference_time = time.perf_counter() - inference_start
        eval_out = eval_result[4]
        eval_f1 = eval_f1_macro(
            dataset.label[eval_split['test']],
            eval_out[eval_split['test']],
        )
        print(
            '[EvalOnly] '
            + json.dumps(
                {
                    'checkpoint': checkpoint_path,
                    'checkpoint_state_sha256': model_state_identity(model),
                    'dataset': dataset_name,
                    'split_fingerprint': split_fingerprint,
                    'run': eval_run,
                    'canonical_retained_edges': int(a_undir),
                    'target_ratio': float(a_undir / b_undir) if b_undir else 0.0,
                    'train_metric': float(eval_result[0]),
                    'validation_metric': float(eval_result[1]),
                    'test_metric': float(eval_result[2]),
                    'test_f1_macro': float(eval_f1),
                    'inference_time_sec': inference_time,
                    'gnn_forward_passes': 1,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if wb_run is not None:
            wb_run.finish()
        return
    scaffold_eval_ensemble_size = int(
        getattr(args, 'scaffold_eval_ensemble_size', 1)
    )
    scaffold_eval_ensemble_source = str(
        getattr(args, 'scaffold_eval_ensemble_source', 'fresh')
    ).lower()
    scaffold_eval_ensemble_reduce = str(
        getattr(args, 'scaffold_eval_ensemble_reduce', 'mean-logits')
    ).lower()
    consensus_enabled = scaffold_consensus_enabled(args)
    consensus_k = int(getattr(args, 'scaffold_consensus_k', 5))
    if scaffold_eval_ensemble_size < 1:
        raise ValueError('--scaffold_eval_ensemble_size must be at least 1')
    if consensus_enabled:
        if str(getattr(args, 'sparsifier', '')).lower() not in SCAFFOLD_METHODS:
            raise ValueError('Scaffold-Consensus is supported only for Scaffold methods')
        if consensus_k < 1:
            raise ValueError('--scaffold_consensus_k must be at least 1')
        if not 0.0 <= float(args.scaffold_consensus_lambda) <= 1.0:
            raise ValueError('--scaffold_consensus_lambda must lie in [0, 1]')
        if args.scaffold_consensus_q is not None and int(args.scaffold_consensus_q) < 0:
            raise ValueError('--scaffold_consensus_q must be non-negative')
        if evaluation_graph != 'sparse' or scaffold_eval_ensemble_source != 'best':
            raise ValueError(
                'Scaffold-Consensus requires --eval_graph sparse and '
                '--scaffold_eval_ensemble_source best'
            )
        if scaffold_eval_ensemble_size != 1:
            raise ValueError(
                'Scaffold-Consensus intermediate evaluation must remain the '
                'single q-edge Scaffold-1 baseline; set '
                '--scaffold_eval_ensemble_size 1'
            )
        if not bool(getattr(args, 'report_final_epoch_only', False)):
            raise ValueError(
                'Scaffold-Consensus is a final-only policy; pass '
                '--report_final_epoch_only'
            )
        if int(getattr(args, 'eval_step', 1)) <= 0:
            raise ValueError(
                'Scaffold-Consensus needs positive --eval_step to collect '
                'validation-ranked training supports'
            )
        if neighbor_training or large_requested:
            raise ValueError(
                'Scaffold-Consensus currently supports the full-batch '
                'Cora/WikiCS pilot; neighbor/large evaluation is not enabled'
            )
        print(
            '[Scaffold-Consensus] enabled final_only=true '
            f'mode={args.scaffold_consensus_mode} K={consensus_k} '
            f'lambda={float(args.scaffold_consensus_lambda):g} '
            'intermediate_evaluation=scaffold-1',
            flush=True,
        )
    sparse_eval_views_enabled = use_sparse_graph_evaluation_views(
        scaffold_eval_ensemble_size,
        scaffold_eval_ensemble_source,
    )
    best_eval_graph_bank_limit = max(
        scaffold_eval_ensemble_size,
        consensus_k if consensus_enabled else 1,
        max((int(v) for v in final_eval_views if v != 'all'), default=1),
        final_graph_bank_capacity(args, final_eval_views) if final_eval_views else 1,
    )
    if sparse_eval_views_enabled:
        if str(getattr(args, 'sparsifier', '')).lower() not in SPARSE_VIEW_METHODS:
            raise ValueError(
                'explicit sparse graph evaluation views are supported only '
                'for Scaffold methods and random_refresh'
            )
        if fixed_eval_edge_index is not None:
            raise ValueError(
                'explicit sparse graph evaluation views require '
                '--eval_graph sparse'
            )
        if (
            (neighbor_training or large_requested)
            and scaffold_eval_ensemble_source != 'best'
        ):
            raise ValueError(
                'large/neighbor sparse graph ensembles require '
                '--scaffold_eval_ensemble_source best'
            )
        if (
            (neighbor_training or large_requested)
            and scaffold_eval_ensemble_reduce != 'union'
        ):
            raise ValueError(
                'large/neighbor sparse graph ensembles currently require '
                '--scaffold_eval_ensemble_reduce union'
            )
        if (
            scaffold_eval_ensemble_size > 1
            and scaffold_eval_ensemble_source == 'fresh'
            and not hasattr(sparsifier, 'resparsify')
        ):
            raise ValueError('fresh sparse evaluation views require resparsify()')
        print(
            '[SCAFFOLD] evaluation graph=sparse '
            f'ensemble_size={scaffold_eval_ensemble_size} '
            f'source={scaffold_eval_ensemble_source} '
            f'reduction={scaffold_eval_ensemble_reduce}',
            flush=True,
        )
        if (
            scaffold_eval_ensemble_size > 1
            and scaffold_eval_ensemble_reduce == 'union'
        ):
            print(
                '[SCAFFOLD] union evaluation is diagnostic: its unique edge '
                'count may exceed the requested target ratio',
                flush=True,
            )
    neighbor_loader_data = None
    full_eval_neighbor_loader_data = None
    if neighbor_training:
        tmp_dir = configure_neighbor_tmp_dir(args)
        neighbor_loader_data = (
            build_tunedgnn_protein_loader_data(dataset)
            if tunedgnn_protein_training
            else build_neighbor_loader_data(dataset)
        )
        if fixed_eval_edge_index is not None:
            with temporary_dataset_graph_view(
                dataset,
                fixed_eval_edge_index,
                None,
            ):
                full_eval_neighbor_loader_data = (
                    build_tunedgnn_protein_loader_data(dataset)
                    if tunedgnn_protein_training
                    else build_neighbor_loader_data(dataset)
                )
            print(
                f'[EvaluationGraph] prepared {evaluation_graph} neighbor topology '
                f'directed_edges={int(fixed_eval_edge_index.size(1))}',
                flush=True,
            )
        print(
            'Training Mode: neighbor minibatch '
            f'batch_size={args.neighbor_batch_size}, '
            f'eval_batch_size={args.neighbor_eval_batch_size}, '
            f'fanouts={parse_neighbor_fanouts(args.neighbor_num_neighbors, args.local_layers)}, '
            f'workers={args.neighbor_workers}, '
            f'tmp_dir={tmp_dir if tmp_dir else "default"}'
        )
    elif large_requested:
        print(
            'Training Mode: tunedGNN large-graph '
            f'pipeline={large_pipeline}, '
            f'batch_size={args.large_batch_size}, '
            f'train_parts={args.large_train_parts}, '
            f'eval_parts={args.large_eval_parts}, '
            f'eval_feature_chunks={args.eval_decomposed_layers}, '
            f'loader_workers={getattr(args, "large_loader_workers", 0)}'
        )

    args.method = args.gnn
    logger = Logger(args.runs, args)
    if model is not None:
        model.train()
        print('MODEL:', model)
    print('-' * 50)
    print(f'Starting Training for {args.runs} runs...')
    print('Model is training...')

    if model is not None and wb_run is not None and getattr(args, 'wandb_watch', False):
        try:
            wb.watch(model, log='all', log_freq=max(1, int(getattr(args, 'wandb_log_freq', 1))))
        except Exception:
            pass

    synchronize_timing_device(device)
    training_start = time.perf_counter()
    per_run_test_class_accuracies = []
    per_run_resparsification_times = []
    per_run_training_times = []
    for run in range(args.runs):
        pilot_run_ids = getattr(args, 'scaffold_protocol_run_ids', None)
        if pilot_run_ids is not None and run + 1 not in pilot_run_ids:
            continue
        synchronize_timing_device(device)
        run_start = time.perf_counter()
        reset_phase_timers(device)
        run_budget = RunTimeBudget().start()
        run_initial_sparsification_time_sec = (
            initial_sparsification_time_sec if run == 0 or pilot_run_ids is not None else 0.0
        )
        run_resparsification_time_sec = 0.0
        run_sparsification_foreground_time_sec = 0.0
        # A preceding async run leaves its last support in dataset.graph.
        # Each repetition must start from the same declared initial support.
        dataset.graph['edge_index'] = sparsified_edge_index
        if sparsified_edge_weight is None:
            dataset.graph.pop('edge_weight', None)
        else:
            dataset.graph['edge_weight'] = sparsified_edge_weight
        if tunedgnn_protein_training:
            set_seed_for_run(args, run)
            import dgl

            dgl.seed(args.seed + run)
            dgl.random.seed(args.seed + run)
        else:
            set_seed_for_run(args, run)
        if isinstance(split_idx, list):
            split_for_run = split_idx[run % len(split_idx)]
        else:
            split_for_run = split_idx
        train_idx = split_for_run['train'].to(device)
        protein_train_loader = None
        protein_eval_loader = None
        protein_split_cpu = None
        if tunedgnn_protein_training:
            protein_train_loader, protein_eval_loader, protein_split_cpu = (
                make_tunedgnn_protein_loaders(
                    neighbor_loader_data,
                    train_idx,
                    split_for_run,
                    args,
                )
            )
            if full_eval_neighbor_loader_data is not None:
                protein_eval_loader, protein_split_cpu = (
                    make_tunedgnn_protein_eval_loader(
                        full_eval_neighbor_loader_data,
                        split_for_run,
                        args,
                    )
                )
            model = parse_method(args, n, c, d, device)
            if run == 0:
                print('MODEL:', model)
            if wb_run is not None and getattr(args, 'wandb_watch', False):
                try:
                    wb.watch(model, log='all', log_freq=max(1, int(getattr(args, 'wandb_log_freq', 1))))
                except Exception:
                    pass
        large_loader_data = None
        full_eval_large_loader_data = None
        if large_requested and large_pipeline == 'random_node_loader':
            large_loader_data = build_large_loader_data(dataset, split_for_run)
            if fixed_eval_edge_index is not None:
                with temporary_dataset_graph_view(
                    dataset,
                    fixed_eval_edge_index,
                    None,
                ):
                    full_eval_large_loader_data = build_large_loader_data(
                        dataset,
                        split_for_run,
                    )
        if not tunedgnn_protein_training:
            reset_model_for_run(model, args, run)
        optimizer_class = torch.optim.AdamW if args.optimizer == 'adamw' else torch.optim.Adam
        optimizer = optimizer_class(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
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
        best_test_class_accuracy = None
        best_eval_graph_bank = []
        checkpoint_tracker = None
        ensemble_checkpoint_trackers = {}
        prediction_checkpoint_trackers = {}
        full_checkpoint_tracker = None
        if getattr(args, 'scaffold_final_report_best_checkpoint', False):
            checkpoint_csv = args.scaffold_final_eval_csv or os.environ.get('SCAFFOLD_MULTIVIEW_CSV')
            checkpoint_tracker = BestValidationCheckpoint(
                os.path.join(os.path.dirname(os.path.abspath(checkpoint_csv)), 'checkpoints', f'run_{run + 1:02d}'))
            for k in (getattr(args, 'scaffold_final_best_ensemble_views', None) or ()):
                ensemble_checkpoint_trackers[int(k)] = BestValidationCheckpoint(
                    os.path.join(str(checkpoint_tracker.directory), f'union_{k}'))
            if getattr(args, 'scaffold_final_best_full_checkpoint', False):
                full_checkpoint_tracker = BestValidationCheckpoint(
                    os.path.join(str(checkpoint_tracker.directory), 'full'))
            if getattr(args, 'scaffold_final_prediction_ensembles', False):
                for view in prediction_views(args):
                    prediction_checkpoint_trackers[view] = BestValidationCheckpoint(
                        os.path.join(str(checkpoint_tracker.directory), view))
        best_eval_graph_bank_dir = None
        if scaffold_eval_ensemble_source == 'best':
            best_eval_graph_bank_dir = resolve_sparse_graph_bank_dir(
                args,
                dataset_name,
                run,
            )
            initialize_sparse_graph_bank(best_eval_graph_bank_dir)
            if best_eval_graph_bank_dir is not None:
                print(
                    '[SCAFFOLD] persistent best-graph bank: '
                    f'{best_eval_graph_bank_dir}',
                    flush=True,
                )

        if args.save_model:
            from scaffold_gnn.utils.logger import save_model

            save_model(args, model, optimizer, run)

        eval_step = int(getattr(args, 'eval_step', 1))
        resparsify_every = int(getattr(args, 'scaffold_resparsify_every', 0)) if hasattr(sparsifier, 'resparsify') else 0
        fast_maxst_every = int(getattr(args, 'scaffold_fast_maxst_every', 0))
        if fast_maxst_every < 0:
            raise ValueError('--scaffold_fast_maxst_every must be non-negative')
        resparsify_include_final = bool(
            getattr(args, 'scaffold_resparsify_include_final', False)
        )
        total_epochs = training_epoch_count(args, tunedgnn_contract)
        refresh_epochs = scheduled_resparsify_epochs(
            total_epochs,
            resparsify_every,
            resparsify_include_final,
        )
        async_resparsify_enabled = use_async_scaffold_resparsify(
            args,
            sparsifier,
            resparsify_every,
            refresh_epochs,
        )
        if resparsify_every > 0:
            print(
                '[Resparsify] schedule: initial graph before epoch 0; '
                f'{summarize_resparsify_schedule(args.epochs, resparsify_every, resparsify_include_final)}'
            )
            if async_resparsify_enabled:
                print(
                    '[AsyncResparsify] enabled: initial graph is blocking; '
                    f'{len(refresh_epochs)} future refresh graphs will be built in one background process',
                    flush=True,
                )
            elif (
                fast_maxst_every > 0
                and bool(getattr(args, 'scaffold_async_resparsify', False))
            ):
                print(
                    '[Resparsify] mixed-backbone schedule requires synchronous '
                    'refreshes; async resparsification is disabled',
                    flush=True,
                )
            if fast_maxst_every > 0:
                ordinary_backbone = (
                    getattr(sparsifier, 'sample_backbone', None)
                    if hasattr(sparsifier, 'sample_backbone')
                    else getattr(sparsifier, 'init_support', None)
                )
                periodic_backbone = (
                    'fixed-maxst'
                    if hasattr(sparsifier, 'sample_backbone')
                    else 'fast-maxst'
                )
                print(
                    '[Resparsify] mixed-backbone schedule: '
                    f'ordinary={str(ordinary_backbone).replace("_", "-")} '
                    f'every_{fast_maxst_every}th_refresh={periodic_backbone}',
                    flush=True,
                )
        base_resparsify_seed = int(sparsifier_params.get('seed') or 0) if isinstance(sparsifier_params, dict) else 0
        resparsify_count = 0
        async_manager = None
        try:
            if async_resparsify_enabled:
                if async_resparsify_source_payload is None:
                    raise RuntimeError('--scaffold_async_resparsify requires a cached PyG source graph.')
                async_manager = AsyncResparsifyManager(
                    source_payload=async_resparsify_source_payload,
                    sparsifier_name=args.sparsifier,
                    sparsifier_params=sparsifier_params,
                    total_refreshes=len(refresh_epochs),
                    base_seed=base_resparsify_seed,
                    scheduled_epochs=refresh_epochs,
                    wait_for_schedule=getattr(args, 'scaffold_async_wait', False),
                )
                async_manager.start()

            for epoch in range(total_epochs):
                model.train()
                synchronize_timing_device(device)
                _epoch_train_start = time.perf_counter()
                if async_manager is not None:
                    async_result = async_manager.poll(run, epoch)
                    if async_result is not None:
                        synchronize_timing_device(device)
                        apply_start = time.perf_counter()
                        new_sparsified = _data_from_async_resparsify_result(async_result)
                        new_edge_index, new_edge_weight, _ = _finalize_sparsified_edge_index(
                            new_sparsified, args, edge_device, n, tensor_scaffold
                        )
                        dataset.graph['edge_index'] = new_edge_index
                        if new_edge_weight is None:
                            dataset.graph.pop('edge_weight', None)
                        else:
                            dataset.graph['edge_weight'] = new_edge_weight
                        if neighbor_training:
                            if tunedgnn_protein_training:
                                neighbor_loader_data = build_tunedgnn_protein_loader_data(dataset)
                                protein_train_loader, protein_eval_loader, protein_split_cpu = (
                                    make_tunedgnn_protein_loaders(
                                        neighbor_loader_data,
                                        train_idx,
                                        split_for_run,
                                        args,
                                    )
                                )
                                if full_eval_neighbor_loader_data is not None:
                                    protein_eval_loader, protein_split_cpu = (
                                        make_tunedgnn_protein_eval_loader(
                                            full_eval_neighbor_loader_data,
                                            split_for_run,
                                            args,
                                        )
                                    )
                            else:
                                neighbor_loader_data = build_neighbor_loader_data(dataset)
                        if large_requested and large_pipeline == 'random_node_loader':
                            large_loader_data = build_large_loader_data(dataset, split_for_run)
                        async_manager.mark_applied(
                            async_result,
                            run,
                            epoch,
                            int(new_edge_index.size(1)),
                        )
                        synchronize_timing_device(device)
                        apply_time_sec = time.perf_counter() - apply_start
                        run_resparsification_time_sec += apply_time_sec
                        run_sparsification_foreground_time_sec += apply_time_sec
                elif should_resparsify_epoch(
                    epoch,
                    total_epochs,
                    resparsify_every,
                    resparsify_include_final,
                ):
                    resparsify_count += 1
                    new_seed = base_resparsify_seed + resparsify_count
                    synchronize_timing_device(device)
                    t0 = time.perf_counter()
                    with scaffold_refresh_backbone(
                        sparsifier,
                        resparsify_count,
                        fast_maxst_every,
                    ) as refresh_backbone:
                        new_sparsified = sparsifier.resparsify(seed=new_seed)
                    new_edge_index, new_edge_weight, _ = _finalize_sparsified_edge_index(
                        new_sparsified, args, edge_device, n, tensor_scaffold
                    )
                    dataset.graph['edge_index'] = new_edge_index
                    if new_edge_weight is None:
                        dataset.graph.pop('edge_weight', None)
                    else:
                        dataset.graph['edge_weight'] = new_edge_weight
                    if neighbor_training:
                        if tunedgnn_protein_training:
                            neighbor_loader_data = build_tunedgnn_protein_loader_data(dataset)
                            protein_train_loader, protein_eval_loader, protein_split_cpu = (
                                make_tunedgnn_protein_loaders(
                                    neighbor_loader_data,
                                    train_idx,
                                    split_for_run,
                                    args,
                                )
                            )
                            if full_eval_neighbor_loader_data is not None:
                                protein_eval_loader, protein_split_cpu = (
                                    make_tunedgnn_protein_eval_loader(
                                        full_eval_neighbor_loader_data,
                                        split_for_run,
                                        args,
                                    )
                                )
                        else:
                            neighbor_loader_data = build_neighbor_loader_data(dataset)
                    if large_requested and large_pipeline == 'random_node_loader':
                        large_loader_data = build_large_loader_data(dataset, split_for_run)
                    synchronize_timing_device(device)
                    refresh_time_sec = time.perf_counter() - t0
                    run_resparsification_time_sec += refresh_time_sec
                    run_sparsification_foreground_time_sec += refresh_time_sec
                    print(
                        f'[Resparsify] run={run} epoch={epoch} seed={new_seed} '
                        f'backbone={refresh_backbone} '
                        f'edges={new_edge_index.size(1)} '
                        f'time={refresh_time_sec:.3f}s'
                    )
                human_epoch = epoch + 1
                eligible = human_epoch >= max(1, int(args.eval_start_epoch))
                if eval_step <= 0:
                    should_eval_epoch = eligible and epoch == total_epochs - 1
                elif tunedgnn_contract and args.model_profile == 'proteins':
                    should_eval_epoch = eligible and (
                        human_epoch % eval_step == 0 or epoch == total_epochs - 1
                    )
                elif (
                    tunedgnn_contract
                    and args.model_profile == 'large'
                    and dataset_name == 'pokec'
                ):
                    should_eval_epoch = eligible and epoch % eval_step == 0
                else:
                    should_eval_epoch = eligible and (
                        epoch % eval_step == 0 or epoch == total_epochs - 1
                    )
                # A graceful wall-time stop is also the final epoch actually
                # reached by this run. Evaluate it before stopping so
                # final-only policies can construct/report their result.
                terminal_eval_epoch = (
                    epoch == total_epochs - 1 or run_budget.expired
                )
                if run_budget.expired:
                    should_eval_epoch = True
                if neighbor_training:
                    if tunedgnn_protein_training:
                        loss = tunedgnn_protein_train_epoch(
                            model,
                            neighbor_loader_data,
                            protein_train_loader,
                            criterion,
                            optimizer,
                            device,
                            expected_examples=int(train_idx.numel()),
                        )
                    else:
                        loss = neighbor_train_epoch(
                            model,
                            neighbor_loader_data,
                            train_idx,
                            args,
                            criterion,
                            dataset_name,
                            bce_datasets,
                            optimizer,
                            device,
                        )
                    if not should_eval_epoch:
                        if epoch % args.display_step == 0:
                            eval_text = 'final epoch only' if eval_step <= 0 else f'every {eval_step} epochs'
                            print(
                                f'Epoch: {epoch:02d}, Loss: {loss:.4f}, '
                                f'Eval: skipped ({eval_text})'
                            )
                        if scheduler is not None:
                            scheduler.step(scheduler_metric)
                        continue
                    if tunedgnn_protein_training:
                        result = evaluate_tunedgnn_protein(
                            model,
                            (
                                full_eval_neighbor_loader_data
                                if full_eval_neighbor_loader_data is not None
                                else neighbor_loader_data
                            ),
                            protein_eval_loader,
                            protein_split_cpu,
                            eval_func,
                            criterion,
                            device,
                        )
                    else:
                        result = evaluate_neighbor(
                            model,
                            (
                                full_eval_neighbor_loader_data
                                if full_eval_neighbor_loader_data is not None
                                else neighbor_loader_data
                            ),
                            dataset,
                            split_for_run,
                            eval_func,
                            criterion,
                            args,
                            dataset_name,
                            bce_datasets,
                            device,
                            c,
                        )
                    if sparse_eval_views_enabled and checkpoint_tracker is None:
                        eval_graph_views = update_best_sparse_graph_bank(
                            best_eval_graph_bank,
                            dataset.graph['edge_index'],
                            dataset.graph.get('edge_weight'),
                            result[1],
                            epoch,
                            best_eval_graph_bank_limit,
                            best_eval_graph_bank_dir,
                            dataset_name=dataset_name,
                            target_ratio=args.target_ratio,
                            run=run,
                            distinct=True,
                            num_nodes=n,
                            split_fingerprint=split_fingerprint,
                        )
                        if not final_eval_views and len(eval_graph_views) >= scaffold_eval_ensemble_size:
                            union_edge_index, union_edge_weight = union_sparse_graph_views(
                                eval_graph_views,
                                n,
                            )
                            with temporary_dataset_graph_view(
                                dataset,
                                union_edge_index,
                                union_edge_weight,
                            ):
                                if tunedgnn_protein_training:
                                    union_loader_data = build_tunedgnn_protein_loader_data(
                                        dataset
                                    )
                                    (
                                        _union_train_loader,
                                        union_eval_loader,
                                        union_split_cpu,
                                    ) = make_tunedgnn_protein_loaders(
                                        union_loader_data,
                                        train_idx,
                                        split_for_run,
                                        args,
                                    )
                                    result = evaluate_tunedgnn_protein(
                                        model,
                                        union_loader_data,
                                        union_eval_loader,
                                        union_split_cpu,
                                        eval_func,
                                        criterion,
                                        device,
                                    )
                                else:
                                    union_loader_data = build_neighbor_loader_data(
                                        dataset
                                    )
                                    result = evaluate_neighbor(
                                        model,
                                        union_loader_data,
                                        dataset,
                                        split_for_run,
                                        eval_func,
                                        criterion,
                                        args,
                                        dataset_name,
                                        bce_datasets,
                                        device,
                                        c,
                                    )
                    train_acc, valid_acc, test_acc, valid_loss, out = result

                    split_cpu = {key: value.detach().cpu() for key, value in split_for_run.items()}
                    labels_cpu = dataset.label.detach().cpu()
                    train_f1 = eval_f1_macro(labels_cpu[split_cpu['train']], out[split_cpu['train']])
                    test_f1 = eval_f1_macro(labels_cpu[split_cpu['test']], out[split_cpu['test']])
                    if dataset_name == 'ogbn-proteins':
                        test_class_accuracy = None
                    else:
                        test_class_accuracy = eval_acc_per_class(
                            labels_cpu[split_cpu['test']],
                            out[split_cpu['test']],
                            num_classes=c,
                        )
                elif large_requested and large_pipeline == 'random_node_loader':
                    loss = large_random_node_train_epoch(
                        model,
                        large_loader_data,
                        args,
                        criterion,
                        dataset_name,
                        bce_datasets,
                        optimizer,
                        device,
                    )
                    if not should_eval_epoch:
                        if epoch % args.display_step == 0:
                            eval_text = 'final epoch only' if eval_step <= 0 else f'every {eval_step} epochs'
                            print(
                                f'Epoch: {epoch:02d}, Loss: {loss:.4f}, '
                                f'Eval: skipped ({eval_text})'
                            )
                        if scheduler is not None:
                            scheduler.step(scheduler_metric)
                        continue
                    result = evaluate_large_random_node(
                        model,
                        (
                            full_eval_large_loader_data
                            if full_eval_large_loader_data is not None
                            else large_loader_data
                        ),
                        args,
                        eval_func,
                        criterion,
                        dataset_name,
                        bce_datasets,
                        device,
                    )
                    if sparse_eval_views_enabled and checkpoint_tracker is None:
                        eval_graph_views = update_best_sparse_graph_bank(
                            best_eval_graph_bank,
                            dataset.graph['edge_index'],
                            dataset.graph.get('edge_weight'),
                            result[1],
                            epoch,
                            best_eval_graph_bank_limit,
                            best_eval_graph_bank_dir,
                            dataset_name=dataset_name,
                            target_ratio=args.target_ratio,
                            run=run,
                            distinct=True,
                            num_nodes=n,
                            split_fingerprint=split_fingerprint,
                        )
                        if not final_eval_views and len(eval_graph_views) >= scaffold_eval_ensemble_size:
                            union_edge_index, union_edge_weight = union_sparse_graph_views(
                                eval_graph_views,
                                n,
                            )
                            with temporary_dataset_graph_view(
                                dataset,
                                union_edge_index,
                                union_edge_weight,
                            ):
                                union_loader_data = build_large_loader_data(
                                    dataset,
                                    split_for_run,
                                )
                                result = evaluate_large_random_node(
                                    model,
                                    union_loader_data,
                                    args,
                                    eval_func,
                                    criterion,
                                    dataset_name,
                                    bce_datasets,
                                    device,
                                )
                    train_acc, valid_acc, test_acc, valid_loss, out = result

                    y_true_parts, y_pred_parts = out
                    train_f1 = eval_f1_macro(y_true_parts['train'], y_pred_parts['train'])
                    test_f1 = eval_f1_macro(y_true_parts['test'], y_pred_parts['test'])
                    if dataset_name == 'ogbn-proteins':
                        test_class_accuracy = None
                    else:
                        test_class_accuracy = eval_acc_per_class(
                            y_true_parts['test'],
                            y_pred_parts['test'],
                            num_classes=c,
                        )
                elif large_requested and large_pipeline == 'node_batch':
                    loss = large_node_batch_train_epoch(
                        model,
                        dataset,
                        train_idx,
                        args,
                        criterion,
                        dataset_name,
                        bce_datasets,
                        optimizer,
                        device,
                    )
                    if not should_eval_epoch:
                        if epoch % args.display_step == 0:
                            eval_text = 'final epoch only' if eval_step <= 0 else f'every {eval_step} epochs'
                            print(
                                f'Epoch: {epoch:02d}, Loss: {loss:.4f}, '
                                f'Eval: skipped ({eval_text})'
                            )
                        if scheduler is not None:
                            scheduler.step(scheduler_metric)
                        continue
                    if fixed_eval_edge_index is not None:
                        with temporary_dataset_graph_view(
                            dataset,
                            fixed_eval_edge_index,
                            None,
                        ):
                            result = evaluate_large_full(
                                model,
                                dataset,
                                split_for_run,
                                eval_func,
                                criterion,
                                args,
                                device,
                            )
                    else:
                        result = evaluate_large_full(
                            model,
                            dataset,
                            split_for_run,
                            eval_func,
                            criterion,
                            args,
                            device,
                        )
                    if sparse_eval_views_enabled and checkpoint_tracker is None:
                        eval_graph_views = update_best_sparse_graph_bank(
                            best_eval_graph_bank,
                            dataset.graph['edge_index'],
                            dataset.graph.get('edge_weight'),
                            result[1],
                            epoch,
                            best_eval_graph_bank_limit,
                            best_eval_graph_bank_dir,
                            dataset_name=dataset_name,
                            target_ratio=args.target_ratio,
                            run=run,
                            distinct=True,
                            num_nodes=n,
                            split_fingerprint=split_fingerprint,
                        )
                        if not final_eval_views and len(eval_graph_views) >= scaffold_eval_ensemble_size:
                            union_edge_index, union_edge_weight = union_sparse_graph_views(
                                eval_graph_views,
                                n,
                            )
                            with temporary_dataset_graph_view(
                                dataset,
                                union_edge_index,
                                union_edge_weight,
                            ):
                                result = evaluate_large_full(
                                    model,
                                    dataset,
                                    split_for_run,
                                    eval_func,
                                    criterion,
                                    args,
                                    device,
                                )
                    train_acc, valid_acc, test_acc, valid_loss, out = result

                    split_cpu = {key: value.detach().cpu() for key, value in split_for_run.items()}
                    labels_cpu = dataset.label.detach().cpu()
                    train_f1 = eval_f1_macro(labels_cpu[split_cpu['train']], out[split_cpu['train']])
                    test_f1 = eval_f1_macro(labels_cpu[split_cpu['test']], out[split_cpu['test']])
                    if dataset_name == 'ogbn-proteins':
                        test_class_accuracy = None
                    else:
                        test_class_accuracy = eval_acc_per_class(
                            labels_cpu[split_cpu['test']],
                            out[split_cpu['test']],
                            num_classes=c,
                        )
                else:
                    synchronize_timing_device(device)
                    _epoch_train_start = time.perf_counter()
                    optimizer.zero_grad()
                    with amp_autocast(args, device):
                        out = model(
                            dataset.graph['node_feat'],
                            dataset.graph['edge_index'],
                            dataset.graph.get('edge_weight'),
                        )

                        if dataset_name in bce_datasets:
                            if dataset.label.shape[1] == 1:
                                true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
                            else:
                                true_label = dataset.label
                            loss = criterion(out[train_idx].float(), true_label.squeeze(1)[train_idx].to(torch.float))
                        else:
                            out = F.log_softmax(out.float(), dim=1)
                            loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])

                    loss.backward()
                    optimizer.step()
                    synchronize_timing_device(device)
                    _PHASE_SEC['train'] += time.perf_counter() - _epoch_train_start

                    if not should_eval_epoch:
                        if epoch % args.display_step == 0:
                            eval_text = 'final epoch only' if eval_step <= 0 else f'every {eval_step} epochs'
                            print(
                                f'Epoch: {epoch:02d}, Loss: {loss:.4f}, '
                                f'Eval: skipped ({eval_text})'
                            )
                        if scheduler is not None:
                            scheduler.step(scheduler_metric)
                        continue
                    training_edge_index = dataset.graph['edge_index']
                    training_edge_weight = dataset.graph.get('edge_weight')
                    if sparse_eval_views_enabled:
                        current_view = sparse_graph_view(
                            training_edge_index,
                            training_edge_weight,
                        )
                        if scaffold_eval_ensemble_source == 'best':
                            with torch.no_grad():
                                model.eval()
                                current_out = model(
                                    dataset.graph['node_feat'],
                                    training_edge_index,
                                    training_edge_weight,
                                )
                            current_valid = eval_func(
                                dataset.label[split_for_run['valid']],
                                current_out[split_for_run['valid']],
                            )
                            eval_graph_views = update_best_sparse_graph_bank(
                                best_eval_graph_bank,
                                current_view[0],
                                current_view[1],
                                current_valid,
                                epoch,
                                best_eval_graph_bank_limit,
                                best_eval_graph_bank_dir,
                                dataset_name=dataset_name,
                                target_ratio=args.target_ratio,
                                run=run,
                                distinct=True,
                                num_nodes=n,
                                split_fingerprint=split_fingerprint,
                            )
                        else:
                            eval_graph_views = []
                            rng_state = capture_training_rng_state()
                            synchronize_timing_device(device)
                            eval_sparsify_start = time.perf_counter()
                            try:
                                eval_seed_base = (
                                    base_resparsify_seed
                                    + 1_000_000_000
                                    + run * 10_000_000
                                    + epoch * scaffold_eval_ensemble_size
                                )
                                for member in range(scaffold_eval_ensemble_size):
                                    eval_sparsified = sparsifier.resparsify(
                                        seed=eval_seed_base + member
                                    )
                                    eval_edge_index, eval_edge_weight, _ = (
                                        _finalize_sparsified_edge_index(
                                            eval_sparsified,
                                            args,
                                            edge_device,
                                            n,
                                            tensor_scaffold,
                                        )
                                    )
                                    eval_graph_views.append(
                                        sparse_graph_view(
                                            eval_edge_index,
                                            eval_edge_weight,
                                        )
                                    )
                            finally:
                                restore_training_rng_state(rng_state)
                            synchronize_timing_device(device)
                            eval_sparsify_time = (
                                time.perf_counter() - eval_sparsify_start
                            )
                            run_resparsification_time_sec += eval_sparsify_time
                            run_sparsification_foreground_time_sec += eval_sparsify_time

                        if final_eval_views and checkpoint_tracker is not None:
                            # Match the old Sample-1 validation behavior: score
                            # the historical best single support with today's
                            # model, rather than just today's training support.
                            checkpoint_out = predict_sparse_graph_ensemble(
                                model, dataset.graph['node_feat'], eval_graph_views[:1],
                                scaffold_eval_ensemble_reduce, n)
                            result = evaluate(model, dataset, split_for_run, eval_func,
                                              criterion, args, result=checkpoint_out)
                        elif final_eval_views:
                            result = evaluate(
                                model, dataset, split_for_run, eval_func,
                                criterion, args, result=current_out,
                            )
                        elif consensus_enabled and terminal_eval_epoch:
                            result = run_final_scaffold_consensus(
                                args=args,
                                model=model,
                                dataset=dataset,
                                split_for_run=split_for_run,
                                eval_func=eval_func,
                                criterion=criterion,
                                sparsifier=sparsifier,
                                bank=best_eval_graph_bank,
                                bank_dir=best_eval_graph_bank_dir,
                                original_edge_index=sparsifier_edge_index,
                                edge_device=edge_device,
                                num_nodes=n,
                                dataset_name=dataset_name,
                                split_fingerprint=split_fingerprint,
                                run=run,
                            )
                        else:
                            # A Consensus bank can retain K supports, but the
                            # training-time scheduler remains matched to the
                            # existing Scaffold-1 protocol. Never union the K
                            # construction supports during training.
                            inference_views = (
                                eval_graph_views[:scaffold_eval_ensemble_size]
                                if scaffold_eval_ensemble_source == 'best'
                                else eval_graph_views
                            )
                            ensemble_out = predict_sparse_graph_ensemble(
                                model,
                                dataset.graph['node_feat'],
                                inference_views,
                                scaffold_eval_ensemble_reduce,
                                n,
                            )
                            if consensus_enabled:
                                # Consensus support collection is validation
                                # only. Do not inspect train/test labels or
                                # metrics before the terminal selection.
                                scheduler_metric = float(
                                    eval_func(
                                        dataset.label[split_for_run['valid']],
                                        ensemble_out[split_for_run['valid']],
                                    )
                                )
                                if scheduler is not None:
                                    scheduler.step(scheduler_metric)
                                continue
                            result = evaluate(
                                model,
                                dataset,
                                split_for_run,
                                eval_func,
                                criterion,
                                args,
                                result=ensemble_out,
                            )
                    else:
                        if fixed_eval_edge_index is not None:
                            dataset.graph['edge_index'] = fixed_eval_edge_index
                            dataset.graph.pop('edge_weight', None)
                        try:
                            result = evaluate(
                                model,
                                dataset,
                                split_for_run,
                                eval_func,
                                criterion,
                                args,
                            )
                        finally:
                            dataset.graph['edge_index'] = training_edge_index
                            if training_edge_weight is None:
                                dataset.graph.pop('edge_weight', None)
                            else:
                                dataset.graph['edge_weight'] = training_edge_weight
                    train_acc, valid_acc, test_acc, valid_loss, out = result

                    train_f1 = eval_f1_macro(dataset.label[split_for_run['train']], out[split_for_run['train']])
                    test_f1 = eval_f1_macro(dataset.label[split_for_run['test']], out[split_for_run['test']])
                    if dataset_name == 'ogbn-proteins':
                        test_class_accuracy = None
                    else:
                        test_class_accuracy = eval_acc_per_class(
                            dataset.label[split_for_run['test']],
                            out[split_for_run['test']],
                            num_classes=c,
                        )

                if final_eval_views:
                    if checkpoint_tracker is not None:
                        if neighbor_training or large_requested:
                            # Rank large-graph supports with exact whole-topology
                            # inference, not the sampled/partitioned training evaluator.
                            # Training itself retains its original pipeline.
                            checkpoint_rng = capture_training_rng_state()
                            try:
                                current_metrics, current_f1 = evaluate_final_full_batch(
                                    model, dataset, split_for_run, dataset.graph['edge_index'],
                                    dataset.graph.get('edge_weight'), eval_func, criterion,
                                    args, device, evaluation_feature_decomposition)
                                update_best_sparse_graph_bank(
                                    best_eval_graph_bank, dataset.graph['edge_index'],
                                    dataset.graph.get('edge_weight'), current_metrics[1], epoch,
                                    best_eval_graph_bank_limit, best_eval_graph_bank_dir,
                                    dataset_name=dataset_name, target_ratio=args.target_ratio,
                                    run=run, distinct=True, num_nodes=n,
                                    split_fingerprint=split_fingerprint)
                                selected_graph = best_eval_graph_bank[0]
                                if selected_graph['epoch'] == epoch:
                                    single_metrics, single_f1 = current_metrics, current_f1
                                else:
                                    single_metrics, single_f1 = evaluate_final_full_batch(
                                        model, dataset, split_for_run, *selected_graph['view'],
                                        eval_func, criterion, args, device, evaluation_feature_decomposition)
                                train_acc, valid_acc, test_acc, valid_loss = single_metrics
                                train_f1, test_f1 = single_f1['train'], single_f1['test']
                                checkpoint_accuracy = single_f1.get('test_accuracy')
                                test_class_accuracy = None
                            finally:
                                restore_training_rng_state(checkpoint_rng)
                                invalidate_graph_dependent_model_caches(model)
                        else:
                            checkpoint_accuracy = float(eval_acc(
                                dataset.label[split_for_run['test']], out[split_for_run['test']]))
                        selected_graph = best_eval_graph_bank[0]
                        checkpoint_updated = checkpoint_tracker.observe(
                            model=model, view=selected_graph['view'], graph_epoch=selected_graph['epoch'],
                            model_epoch=epoch + 1, valid_metric=valid_acc, test_metric=test_acc,
                            test_accuracy=checkpoint_accuracy)
                        if checkpoint_updated:
                            best_val, best_test = float(valid_acc), float(test_acc)
                            best_test_class_accuracy = test_class_accuracy
                        if epoch % args.display_step == 0:
                            diagnostic = checkpoint_tracker.diagnostics()
                            print(f'[ScaffoldCheckpoint] run={run + 1} epoch={epoch + 1}/{total_epochs} '
                                  f'metric={args.metric} '
                                  f'valid={100 * float(valid_acc):.2f}% test={100 * float(test_acc):.2f}% '
                                  f'best_valid={100 * checkpoint_tracker.best["valid_metric"]:.2f}% '
                                  f'best_model_epoch={diagnostic["best_validation_model_epoch"]} '
                                  f'highest_test_seen={100 * diagnostic["highest_test_metric"]:.2f}% '
                                  f'(logging only)', flush=True)
                        evaluation_trackers = list(ensemble_checkpoint_trackers.items())
                        if full_checkpoint_tracker is not None:
                            evaluation_trackers.append((0, full_checkpoint_tracker))
                        for k, evaluation_tracker in evaluation_trackers:
                            if k == 0:
                                evaluation_edges, evaluation_weights = multiview_original_edges, None
                                graph_epoch, graph_epochs, historical_scores = -1, [], []
                            else:
                                if len(best_eval_graph_bank) < k:
                                    continue  # Match the old k-view warmup policy.
                                selected = best_eval_graph_bank[:k]
                                evaluation_edges, evaluation_weights = union_sparse_graph_views(
                                    [entry['view'] for entry in selected], n)
                                graph_epochs = [entry['epoch'] for entry in selected]
                                graph_epoch = max(graph_epochs)
                                historical_scores = [entry['validation_score'] for entry in selected]
                            evaluation_edges, evaluation_weights = evaluation_graph_on_device(
                                dataset.graph['node_feat'], evaluation_edges, evaluation_weights)
                            evaluation_rng = capture_training_rng_state()
                            was_training = model.training
                            try:
                                invalidate_graph_dependent_model_caches(model)
                                if neighbor_training or large_requested:
                                    evaluation_metrics, evaluation_f1 = evaluate_final_full_batch(
                                        model, dataset, split_for_run, evaluation_edges, evaluation_weights,
                                        eval_func, criterion, args, device, evaluation_feature_decomposition)
                                    evaluation_accuracy = evaluation_f1.get('test_accuracy')
                                else:
                                    with torch.no_grad():
                                        model.eval()
                                        evaluation_out = model(dataset.graph['node_feat'], evaluation_edges, evaluation_weights)
                                        evaluation_metrics = evaluate(model, dataset, split_for_run, eval_func,
                                                                      criterion, args, result=evaluation_out)
                                        evaluation_accuracy = float(eval_acc(dataset.label[split_for_run['test']],
                                                                             evaluation_out[split_for_run['test']]))
                                evaluation_tracker.observe(
                                    model=model, view=(evaluation_edges, evaluation_weights), model_epoch=epoch + 1,
                                    graph_epoch=graph_epoch, graph_epochs=graph_epochs,
                                    historical_validation_scores=historical_scores,
                                    k_effective=k, valid_metric=evaluation_metrics[1], test_metric=evaluation_metrics[2],
                                    test_accuracy=evaluation_accuracy,
                                    support_records=selected if k and getattr(args, 'scaffold_final_budgeted_union', False) else None)
                            finally:
                                restore_training_rng_state(evaluation_rng)
                                model.train(was_training)
                                invalidate_graph_dependent_model_caches(model)
                            if epoch % args.display_step == 0:
                                checkpoint_tag = 'ScaffoldFullCheckpoint' if k == 0 else 'ScaffoldEnsembleCheckpoint'
                                print(f'[{checkpoint_tag}] run={run + 1} k={k} epoch={epoch + 1}/{total_epochs} '
                                      f'metric={args.metric} valid={100 * float(evaluation_metrics[1]):.2f}% '
                                      f'test={100 * float(evaluation_metrics[2]):.2f}% '
                                      f'best_valid={100 * evaluation_tracker.best["valid_metric"]:.2f}% '
                                      f'best_model_epoch={evaluation_tracker.best["model_epoch"]} '
                                      f'highest_test_seen={100 * evaluation_tracker.highest_test_metric:.2f}% '
                                      f'(logging only)', flush=True)
                        if prediction_checkpoint_trackers:
                            from scaffold_gnn.utils.scaffold_multiview import _consume_final_full_batch
                            predictors = prediction_views(args)
                            def sparse_prediction_forward(edges, weights):
                                return _consume_final_full_batch(
                                    model, dataset, split_for_run, edges, weights, args, device,
                                    evaluation_feature_decomposition,
                                    lambda outputs, unused: outputs.detach().cpu())
                            prediction_outputs = prediction_ensemble_outputs(
                                model, dataset.graph['node_feat'], best_eval_graph_bank,
                                ks=sorted({k for k, _ in predictors.values()}),
                                reducers=tuple(dict.fromkeys(mode for _, mode in predictors.values())),
                                forward=sparse_prediction_forward)
                            for view, prediction_out in prediction_outputs.items():
                                k, mode = predictors[view]
                                selected = best_eval_graph_bank[:k]
                                metrics, f1 = score_predictions(
                                    prediction_out, dataset.label, split_for_run, eval_func, criterion, args)
                                tracker = prediction_checkpoint_trackers[view]
                                tracker.observe(
                                    model=model, view=selected[0]['view'], model_epoch=epoch + 1,
                                    graph_epoch=max(s['epoch'] for s in selected),
                                    graph_epochs=[s['epoch'] for s in selected],
                                    historical_validation_scores=[s['validation_score'] for s in selected],
                                    k_effective=k, valid_metric=metrics[1], test_metric=metrics[2],
                                    test_accuracy=f1.get('test_accuracy'), support_records=selected,
                                    selection_policy=prediction_policy(mode))
                                if epoch % args.display_step == 0:
                                    print(f'[ScaffoldPredictionCheckpoint] run={run + 1} view={view} '
                                          f'epoch={epoch + 1}/{total_epochs} valid={100 * metrics[1]:.2f}% '
                                          f'test={100 * metrics[2]:.2f}% best_valid={100 * tracker.best["valid_metric"]:.2f}% '
                                          f'best_model_epoch={tracker.best["model_epoch"]}', flush=True)
                            del prediction_outputs
                    # Intermediate metrics rank supports and optionally save
                    # validation-best single/union/full model-support pairs.
                    # Any explicitly retained final-model controls use the same
                    # last model, including after a graceful one-hour stop.
                    scheduler_metric = float(valid_acc)
                    if scheduler is not None:
                        scheduler.step(scheduler_metric)
                    if epoch % args.display_step == 0:
                        print(f'[ScaffoldMultiView] run={run + 1} epoch={epoch + 1}/{total_epochs} '
                              f'validation={100 * float(valid_acc):.2f}% distinct_bank={len(best_eval_graph_bank)}/{best_eval_graph_bank_limit}', flush=True)
                    if run_budget.exhausted(run + 1, epoch + 1, total_epochs):
                        break
                    continue

                if (
                    sparse_eval_views_enabled
                    and scaffold_eval_ensemble_source == 'best'
                    and len(best_eval_graph_bank) < scaffold_eval_ensemble_size
                ):
                    scheduler_metric = float(valid_acc)
                    if scheduler is not None:
                        scheduler.step(scheduler_metric)
                    if epoch % args.display_step == 0:
                        print(
                            '[SCAFFOLD] evaluation ensemble warmup: '
                            f'views={len(best_eval_graph_bank)}/'
                            f'{scaffold_eval_ensemble_size}; metric not retained'
                    )
                    continue

                # A best-k sparse evaluation bank must inspect intermediate
                # training supports, but this ablation asks for the final
                # trained model rather than checkpoint selection over those
                # inspections.  Keep the bank update above, then discard the
                # intermediate model metrics.  --eval_step 0 cannot implement
                # this policy because a bank observed only at the final epoch
                # would contain a single graph.
                if (
                    bool(getattr(args, 'report_final_epoch_only', False))
                    and not terminal_eval_epoch
                ):
                    scheduler_metric = float(valid_acc)
                    if scheduler is not None:
                        scheduler.step(scheduler_metric)
                    continue

                logger.add_result(run, (train_acc, valid_acc, test_acc, valid_loss, train_f1, test_f1), epoch=epoch)
                scheduler_metric = float(valid_acc)
                if scheduler is not None:
                    scheduler.step(scheduler_metric)

                if valid_acc > best_val:
                    best_val = valid_acc
                    best_test = test_acc
                    best_test_class_accuracy = test_class_accuracy
                    if args.save_model:
                        from scaffold_gnn.utils.logger import save_model

                        save_model(args, model, optimizer, run)

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
                        f'Epoch: {epoch:02d}, Loss: {loss:.4f}, '
                        f'Train: {100 * train_acc:.2f}%, Valid: {100 * valid_acc:.2f}%, '
                        f'Test: {100 * test_acc:.2f}%, Test F1: {100 * test_f1:.2f}%, '
                        f'Best Valid: {100 * best_val:.2f}%, Best Test: {100 * best_test:.2f}%'
                    )

                # The epoch's full-graph validation/test result has already
                # been added to the logger.  End gracefully here so the normal
                # per-run finalizer records the best completed epoch instead
                # of losing the run to an external timeout.
                if run_budget.exhausted(run + 1, epoch + 1, total_epochs):
                    break

        finally:
            if async_manager is not None:
                async_manager.close()
                async_manager.print_summary()
                run_resparsification_time_sec += async_manager.background_cpu_sec

        synchronize_timing_device(device)
        run_wall_time_sec = time.perf_counter() - run_start
        run_training_time_sec = max(
            0.0,
            run_wall_time_sec - run_sparsification_foreground_time_sec,
        )
        run_sparsification_time_sec = (
            run_initial_sparsification_time_sec + run_resparsification_time_sec
        )
        run_total_runtime_sec = run_initial_sparsification_time_sec + run_wall_time_sec
        run_eval_time_sec = _PHASE_SEC['eval']
        run_selection_time_sec = _PHASE_SEC['selection']
        # What is left once scoring and support bookkeeping are taken out: the
        # forward/backward/step that actually fits the model.
        # The bracket is authoritative where it runs (the full-batch epoch
        # loop). The subtraction is the fallback for the mini-batch branches,
        # whose training step sits inside a helper the bracket does not reach.
        run_train_only_time_sec = _PHASE_SEC['train'] or max(
            0.0,
            run_training_time_sec - run_eval_time_sec - run_selection_time_sec,
        )
        if final_eval_views:
            final_records = run_final_views(
                args=args, model=model, dataset=dataset, split=split_for_run,
                bank=best_eval_graph_bank, original_edges=multiview_original_edges,
                original_edge_count=b_undir, requested=final_eval_views,
                eval_func=eval_func, criterion=criterion, device=device,
                decomposition_context=evaluation_feature_decomposition,
                model_identity=model_state_identity,
                metadata={
                    'dataset': dataset_name, 'metric': args.metric,
                    'target_ratio': args.target_ratio,
                    'method': args.sparsifier, 'seed': args.seed, 'run': run + 1,
                    'epochs_requested': args.epochs, 'epochs_completed': epoch + 1,
                    'epoch_count_policy': 'exact' if args.scaffold_exact_epochs else 'legacy-tunedgnn',
                    'split_fingerprint': canonical_split_fingerprint(split_for_run),
                    'sparsification_time_sec': run_sparsification_time_sec,
                    'training_time_sec': run_training_time_sec,
                    'total_runtime_sec': run_total_runtime_sec,
                    'stopped_early': epoch + 1 < total_epochs,
                    'effective_seed': args.seed + run if pilot_run_ids is not None else args.seed,
                    'sample_weight_mode': args.scaffold_sample_weight_mode if args.sparsifier == 'scaffold_sample' else '',
                    'sample_mix_alpha': args.scaffold_sample_mix_alpha if args.sparsifier == 'scaffold_sample' and args.scaffold_sample_weight_mode != 'legacy' else '',
                },
                checkpoint_tracker=checkpoint_tracker,
                ensemble_checkpoint_trackers=ensemble_checkpoint_trackers,
                full_checkpoint_tracker=full_checkpoint_tracker,
                prediction_checkpoint_trackers=prediction_checkpoint_trackers,
            )
            first_valid = next((r for r in final_records if r['status'] == 'OK'
                                and (checkpoint_tracker is None or r['view'] == 'best')), None)
            if first_valid is not None:
                logger.add_result(run, tuple(first_valid[k] for k in (
                    'train_metric', 'valid_metric', 'test_metric', 'valid_loss', 'train_f1', 'test_f1',
                )))
        per_run_resparsification_times.append(run_resparsification_time_sec)
        per_run_training_times.append(run_training_time_sec)
        print(
            f'[RunTiming] run={run + 1} '
            f'initial_sparsification={run_initial_sparsification_time_sec:.6f}s '
            f'resparsification={run_resparsification_time_sec:.6f}s '
            f'sparsification_total={run_sparsification_time_sec:.6f}s '
            f'training={run_training_time_sec:.6f}s '
            f'train_only={run_train_only_time_sec:.6f}s '
            f'eval={run_eval_time_sec:.6f}s '
            f'selection={run_selection_time_sec:.6f}s '
            f'run_wall={run_total_runtime_sec:.6f}s',
            flush=True,
        )
        logger.print_statistics(
            run,
            initial_sparsification_time_sec=run_initial_sparsification_time_sec,
            resparsification_time_sec=run_resparsification_time_sec,
            sparsification_time_sec=run_sparsification_time_sec,
            training_time_sec=run_training_time_sec,
            total_runtime_sec=run_total_runtime_sec,
            train_only_time_sec=run_train_only_time_sec,
            eval_time_sec=run_eval_time_sec,
            selection_time_sec=run_selection_time_sec,
        )
        per_run_test_class_accuracies.append(best_test_class_accuracy)
        if best_test_class_accuracy is None:
            print(
                f'Test Accuracy by Class (%) run={run + 1}: '
                'not applicable for multi-label targets'
            )
        else:
            print(
                f'Test Accuracy by Class (%) run={run + 1}: '
                f'{_class_accuracy_percentages(best_test_class_accuracy)}'
            )
    training_phase_wall_time_sec = time.perf_counter() - training_start
    resparsification_time_sec = sum(per_run_resparsification_times)
    sparsification_time_sec = (
        initial_sparsification_time_sec + resparsification_time_sec
    )
    training_time_sec = sum(per_run_training_times)
    total_model_time_sec = time.perf_counter() - sparsification_start

    results = logger.print_statistics()
    average_test_class_accuracy = average_acc_per_class(per_run_test_class_accuracies)
    if average_test_class_accuracy:
        print(
            'Average Test Accuracy by Class (%) across '
            f'{len(per_run_test_class_accuracies)} run(s): '
            f'{_class_accuracy_percentages(average_test_class_accuracy)}'
        )
    elif per_run_test_class_accuracies:
        print(
            'Average Test Accuracy by Class (%): '
            'not applicable for multi-label targets'
        )
    if results is not None:
        if isinstance(results, tuple):
            test_accs, test_f1s = results
        else:
            test_accs = results
            test_f1s = None

        if test_accs is not None:
            print('-' * 70)
            print(f'Initial Sparsification Time: {initial_sparsification_time_sec:.3f}s')
            print(f'Resparsification Time:       {resparsification_time_sec:.3f}s')
            print(f'Sparsification Time: {sparsification_time_sec:.3f}s')
            print(f'Training Time:       {training_time_sec:.3f}s')
            print(f'Total Run Time:      {total_model_time_sec:.3f}s')
            print(f'Training Phase Wall: {training_phase_wall_time_sec:.3f}s')
            print('-' * 70)
            save_result(
                args,
                test_accs,
                test_f1s,
                sparsification_time_sec=sparsification_time_sec,
                training_time_sec=training_time_sec,
                total_runtime_sec=total_model_time_sec,
                initial_sparsification_time_sec=initial_sparsification_time_sec,
                resparsification_time_sec=resparsification_time_sec,
            )
            results_root = os.environ.get('RESULTS_ROOT', 'results')
            print(f'Results saved to: {results_root}/{args.dataset}/{args.sparsifier}/results.csv')
            log_wandb_final(wb, wb_run, test_accs, test_f1s)

    if wb_run is not None:
        try:
            wb.finish()
        except Exception:
            pass


if __name__ == '__main__':
    main()
