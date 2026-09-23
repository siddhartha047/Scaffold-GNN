from scaffold_gnn.models.model import MPNNs as SmallMPNNs
from scaffold_gnn.models.model_large import LargeMPNNs
from scaffold_gnn.utils.defaults import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR
from scaffold_gnn.utils.seed import set_seed
from scaffold_gnn.sparsifiers.scaffold.tunedgnn_presets import (
    DEFAULT_SCAFFOLD_SCRATCH_ROOT,
    argparse_defaults as tunedgnn_scaffold_defaults,
)
import argparse
import json
import math
import os


SCAFFOLD_CONFIG_DIR = os.path.join(
    os.path.dirname(__file__),
    'sparsifiers',
    'scaffold',
)
SCAFFOLD_FAST_CONFIG = os.path.join(SCAFFOLD_CONFIG_DIR, 'scaffold_fast_config.json')
SCAFFOLD_BATCH_CONFIG = os.path.join(SCAFFOLD_CONFIG_DIR, 'scaffold_batch_config.json')
SCAFFOLD_HEAP_CONFIG = os.path.join(SCAFFOLD_CONFIG_DIR, 'scaffold_heap_config.json')
SCAFFOLD_GREEDY_CONFIG = os.path.join(SCAFFOLD_CONFIG_DIR, 'scaffold_greedy_config.json')
SCAFFOLD_SAMPLE_CONFIG = os.path.join(SCAFFOLD_CONFIG_DIR, 'scaffold_sample_config.json')
DEFAULT_SCAFFOLD_CONFIG = SCAFFOLD_FAST_CONFIG
LARGE_DATASETS = {'reddit', 'reddit2', 'ogbn-products', 'ogbn-arxiv', 'ogbn-proteins', 'pokec'}
AUTO_CLUSTER_EDGE_BUDGET = 50000
LARGE_DEFAULT_CLUSTER_COUNT = 250
SCAFFOLD_SPARSIFIERS = {
    'scaffold_fast', 'scaffold_batch', 'scaffold_heap', 'scaffold_greedy',
    'scaffold_sample',
}
BASIC_SINGLE_GRAPH_SPARSIFIERS = {
    'fixed_support',
    'support_sequence',
    'mst',
    'maxst',
    'fast_maxst',
    'effective_resistance',
    'effective_resistance_fixed',
    'las_vegas_spanner',
    'forest_fire',
    'gspar',
    'local_degree',
    'lsim',
    'lspar',
    'random',
    'random_refresh',
    'rank_degree',
    'scan',
}
TUNEDGNN_GRAPH_SPARSIFIERS = SCAFFOLD_SPARSIFIERS | BASIC_SINGLE_GRAPH_SPARSIFIERS | {'full'}
ADAPTIVE_RUNTIME_DEFAULT_KEYS = {'large_train_parts', 'large_eval_parts'}
TRAINING_DEFAULT_KEYS = {
    'epochs',
    'runs',
    'seed',
    'wandb_mode',
    'display_step',
    'eval_step',
    'train_prop',
    'valid_prop',
    'rand_split',
    'rand_split_class',
    'label_num_per_class',
    'valid_num',
    'test_num',
    'metric',
    'model',
    'model_profile',
    'gnn',
    'hidden_channels',
    'local_layers',
    'num_heads',
    'pre_ln',
    'pre_linear',
    'ln',
    'bn',
    'res',
    'jk',
    'lr',
    'weight_decay',
    'dropout',
    'in_dropout',
    'gpu_memory_profile',
    'train_mode',
    'large_graph_pipeline',
    'large_batch_size',
    'large_batch_mem_fraction',
    'large_batch_min_size',
    'large_batch_max_size',
    'large_train_parts',
    'large_eval_parts',
    'large_eval_cpu',
    'large_loader_workers',
    'neighbor_batch_size',
    'neighbor_eval_batch_size',
    'neighbor_num_neighbors',
    'neighbor_eval_num_neighbors',
    'neighbor_workers',
    'neighbor_tmp_dir',
    'optimizer',
    'lr_scheduler',
    'lr_scheduler_factor',
    'lr_scheduler_patience',
    'eval_start_epoch',
    'log_every',
    'tunedgnn_strict',
}


def _dataset_key(name):
    key = str(name).strip().lower().replace('_', '-')
    if key in {'karate-balance', 'karate-balanced'}:
        return 'karate'
    return key


def _support_budget_mode(value):
    mode = str(value).strip().lower().replace('-', '_')
    aliases = {
        'early': 'early_stop',
        'stop_early': 'early_stop',
        'full_then_trim': 'full_then_random_trim',
        'benchmark': 'full_then_random_trim',
    }
    return aliases.get(mode, mode)


def _canonical_sparsifier_name(value):
    """Normalize CLI spellings and retain the pre-rename Greedy alias."""
    normalized = str(value).strip().lower().replace('-', '_')
    if normalized == 'scaffold_exact':
        return 'scaffold_greedy'
    return normalized


def _canonical_dataset_key(name):
    aliases = {
        'arxiv': 'ogbn-arxiv',
        'ogb-arxiv': 'ogbn-arxiv',
        'product': 'ogbn-products',
        'products': 'ogbn-products',
        'ogb-product': 'ogbn-products',
        'ogb-products': 'ogbn-products',
        'protein': 'ogbn-proteins',
        'proteins': 'ogbn-proteins',
        'ogb-protein': 'ogbn-proteins',
        'ogb-proteins': 'ogbn-proteins',
        'ogbn-protein': 'ogbn-proteins',
        'ogbn-product': 'ogbn-products',
        'karateclub': 'karate',
    }
    key = _dataset_key(name)
    return aliases.get(key, key)


def use_large_model(args):
    profile = str(getattr(args, 'model_profile', 'auto')).lower()
    if profile in {'large', 'products', 'proteins'}:
        return True
    if profile in {'small', 'medium'}:
        return False
    dataset_key = _canonical_dataset_key(getattr(args, 'dataset', ''))
    sparsifier = str(getattr(args, 'sparsifier', '')).lower()
    return dataset_key in LARGE_DATASETS and sparsifier in TUNEDGNN_GRAPH_SPARSIFIERS


def resolve_scaffold_backend(args):
    backend = str(getattr(args, 'scaffold_backend', 'auto')).lower()
    if backend == 'auto':
        dataset_key = _canonical_dataset_key(getattr(args, 'dataset', ''))
        sparsifier = str(getattr(args, 'sparsifier', '')).lower()
        if sparsifier in {'scaffold_fast', 'scaffold_batch'}:
            return 'tensor'
        return 'tensor' if dataset_key in LARGE_DATASETS else 'networkx'
    return backend


def resolve_scaffold_fast_score(args, backend=None):
    score = str(getattr(args, 'scaffold_fast_score', 'auto')).lower()
    if score == 'auto':
        backend = backend or resolve_scaffold_backend(args)
        sparsifier = str(getattr(args, 'sparsifier', 'scaffold_fast')).lower()
        if backend != 'tensor':
            return 'exact'
        return 'tree_exact_loop' if sparsifier == 'scaffold_batch' else 'tree_exact'
    return score


def resolve_joint_cluster_count(args, num_edges=None):
    requested = getattr(args, 'joint_cluster_count', 'auto')
    if requested is None:
        requested = 'auto'
    if isinstance(requested, str):
        requested_text = requested.strip().lower()
        if requested_text and requested_text not in ('auto', 'default', 'edge_budget', 'edge-budget'):
            try:
                count = int(requested_text)
            except ValueError as exc:
                raise ValueError("--joint_cluster_count must be a positive integer or 'auto'") from exc
            if count < 1:
                raise ValueError('--joint_cluster_count must be at least 1')
            return count
    else:
        count = int(requested)
        if count < 1:
            raise ValueError('--joint_cluster_count must be at least 1')
        return count

    dataset_key = _canonical_dataset_key(getattr(args, 'dataset', ''))
    if dataset_key in LARGE_DATASETS or dataset_key.startswith('ogb'):
        return LARGE_DEFAULT_CLUSTER_COUNT
    if num_edges is None:
        return 1
    return max(1, int(math.ceil(float(num_edges) / AUTO_CLUSTER_EDGE_BUDGET)))


def _default_tensor_scaffold_cache_dir(args):
    sparsifier = str(getattr(args, 'sparsifier', 'scaffold_fast')).lower()
    cache_root = os.environ.get('SUPPORT_GRAPH_CACHE_DIR') or os.environ.get('CACHE_ROOT')
    if not cache_root:
        data_dir = os.path.abspath(str(getattr(args, 'data_dir', '') or ''))
        if os.path.basename(data_dir) == 'data':
            cache_root = os.path.join(os.path.dirname(data_dir), 'cache')
    if cache_root:
        return os.path.join(cache_root, sparsifier, 'clusterloader')

    if sparsifier in SCAFFOLD_SPARSIFIERS:
        scaffold_root = os.environ.get(
            'SCAFFOLD_SCRATCH_ROOT',
            DEFAULT_SCAFFOLD_SCRATCH_ROOT,
        )
        return os.path.join(scaffold_root, 'cache', sparsifier, 'clusterloader')

    scratch_root = os.environ.get('SCRATCH') or os.environ.get('TMPDIR')
    if scratch_root:
        return os.path.join(scratch_root, 'support_graph_baselines', sparsifier, 'clusterloader')

    return os.path.join(DEFAULT_CACHE_DIR, sparsifier, 'clusterloader')


def default_cluster_cache_dir(args, backend=None):
    sparsifier = str(getattr(args, 'sparsifier', '')).lower()
    backend = backend or resolve_scaffold_backend(args)
    if sparsifier in {'scaffold_fast', 'scaffold_batch'} and backend == 'tensor':
        return _default_tensor_scaffold_cache_dir(args)
    if sparsifier in SCAFFOLD_SPARSIFIERS:
        scaffold_root = os.environ.get(
            'SCAFFOLD_SCRATCH_ROOT',
            DEFAULT_SCAFFOLD_SCRATCH_ROOT,
        )
        return os.path.join(scaffold_root, 'cache', sparsifier, 'clusterloader')
    return os.path.join(
        getattr(args, 'data_dir', DEFAULT_DATA_DIR),
        str(getattr(args, 'dataset', '')).lower(),
        'cluster_cache',
    )


def _load_scaffold_config(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _select_scaffold_config(path, sparsifier=None, cluster_strategy=None):
    if path:
        return path
    sparsifier_lc = _canonical_sparsifier_name(sparsifier) if sparsifier is not None else ''
    if sparsifier_lc == 'scaffold_sample':
        return SCAFFOLD_SAMPLE_CONFIG
    if sparsifier_lc == 'scaffold_batch':
        return SCAFFOLD_BATCH_CONFIG
    if sparsifier_lc == 'scaffold_greedy':
        return SCAFFOLD_GREEDY_CONFIG
    if sparsifier_lc == 'scaffold_heap' or str(cluster_strategy).lower() == 'heap':
        return SCAFFOLD_HEAP_CONFIG
    return SCAFFOLD_FAST_CONFIG


def _filtered_training_defaults(config, dataset):
    defaults = dict(config.get('default', {}))
    dataset_defaults = config.get('datasets', {})
    defaults.update(dataset_defaults.get(_dataset_key(dataset), {}))
    return {
        key: value
        for key, value in defaults.items()
        if key in TRAINING_DEFAULT_KEYS and key not in ADAPTIVE_RUNTIME_DEFAULT_KEYS
    }


def apply_scaffold_config_defaults(parser, argv=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--dataset', type=str, default='roman-empire')
    pre_parser.add_argument('--sparsifier', type=_canonical_sparsifier_name, default=None)
    pre_parser.add_argument('--joint_cluster_strategy', type=str, default=None)
    pre_parser.add_argument('--joint_swap_refine', action='store_true')
    pre_parser.add_argument('--scaffold_config', type=str, default=None)
    pre_parser.add_argument('--gnn', type=str, default='gcn')
    known_args, _ = pre_parser.parse_known_args(argv)

    config_path = _select_scaffold_config(
        known_args.scaffold_config,
        sparsifier=known_args.sparsifier,
        cluster_strategy=known_args.joint_cluster_strategy,
    )
    sparsifier = known_args.sparsifier
    dataset_key = _canonical_dataset_key(known_args.dataset)
    explicit_non_scaffold = known_args.sparsifier is not None and sparsifier not in SCAFFOLD_SPARSIFIERS
    full_or_omitted = known_args.sparsifier is None or sparsifier == 'full'

    if explicit_non_scaffold and sparsifier not in BASIC_SINGLE_GRAPH_SPARSIFIERS and sparsifier != 'full':
        parser.set_defaults(scaffold_config=config_path)
        return parser

    if sparsifier in BASIC_SINGLE_GRAPH_SPARSIFIERS:
        defaults = tunedgnn_scaffold_defaults(known_args.dataset, known_args.gnn)
        defaults['sparsifier'] = sparsifier
        defaults['scaffold_config'] = config_path
        if defaults:
            parser.set_defaults(**defaults)
        return parser

    config = _load_scaffold_config(config_path)
    if sparsifier == 'full' or (full_or_omitted and dataset_key in LARGE_DATASETS and sparsifier not in SCAFFOLD_SPARSIFIERS):
        defaults = _filtered_training_defaults(config, known_args.dataset)
        defaults.update(tunedgnn_scaffold_defaults(known_args.dataset, known_args.gnn))
        defaults['sparsifier'] = 'full'
        defaults['scaffold_config'] = config_path
        if defaults:
            parser.set_defaults(**defaults)
        return parser

    if known_args.sparsifier is None:
        parser.set_defaults(scaffold_config=config_path)
        return parser

    defaults = dict(config.get('default', {}))
    dataset_defaults = config.get('datasets', {})
    defaults.update(dataset_defaults.get(_dataset_key(known_args.dataset), {}))
    for key in ADAPTIVE_RUNTIME_DEFAULT_KEYS:
        defaults.pop(key, None)
    defaults.update(tunedgnn_scaffold_defaults(known_args.dataset, known_args.gnn))
    configured_backend = str(defaults.get('scaffold_backend', 'auto')).lower()
    if (
        sparsifier in {'scaffold_fast', 'scaffold_batch'}
        and known_args.joint_swap_refine
        and configured_backend != 'networkx'
    ):
        defaults.update({
            'scaffold_backend': 'tensor',
            'scaffold_fast_score': (
                'tree_exact_loop'
                if sparsifier == 'scaffold_batch'
                else 'tree_exact'
            ),
            'joint_cluster_method': 'metis',
            'joint_swap_sampling_mode': 'random',
            'joint_swap_cycle_sample_size': 1,
        })
    defaults['scaffold_config'] = config_path
    if defaults:
        parser.set_defaults(**defaults)
    return parser

def parse_method(args, n, c, d, device):
    profile = str(getattr(args, 'model_profile', 'auto')).lower()
    if profile == 'products':
        from scaffold_gnn.models.model_products import ProductsGNN

        model = ProductsGNN(
            d,
            args.hidden_channels,
            c,
            args.local_layers,
            args.dropout,
            args.ln,
            args.gnn,
            args.jk,
            args.res,
        ).to(device)
        model.gat_eval_chunks = max(1, int(getattr(args, 'gat_eval_chunks', 1)))
    elif profile == 'proteins':
        from scaffold_gnn.models.model_proteins import ProteinGNN

        model = ProteinGNN(
            d,
            c,
            n_layers=args.local_layers,
            n_heads=args.num_heads,
            n_hidden=args.hidden_channels,
            dropout=args.dropout,
            input_drop=args.in_dropout,
            mpnn=args.gnn,
            jumping_knowledge=args.jk,
        ).to(device)
    elif use_large_model(args):
        model = LargeMPNNs(
            d,
            args.hidden_channels,
            c,
            local_layers=args.local_layers,
            in_dropout=args.in_dropout,
            dropout=args.dropout,
            heads=args.num_heads,
            pre_ln=args.pre_ln,
            bn=args.bn,
            local_attn=args.gnn == 'gat',
            res=args.res,
            ln=args.ln,
            jk=args.jk,
            sage=args.gnn == 'sage',
        ).to(device)
    else:
        model = SmallMPNNs(d, args.hidden_channels, c, local_layers=args.local_layers, dropout=args.dropout, heads=args.num_heads, pre_ln=args.pre_ln, pre_linear=args.pre_linear, res=args.res, ln=args.ln, bn=args.bn, jk=args.jk, gnn=args.gnn).to(device)
    return model

def parser_add_main_args(parser):
    parser.add_argument('--exact_epochs', action='store_true',
                        help='Execute exactly --epochs optimizer epochs for every model profile.')
    parser.add_argument('--large_eval_mode', choices=['partition', 'full'], default='partition',
                        help='full preserves every edge during large-graph validation/test; partition uses induced subgraphs.')
    parser.add_argument('--gat_eval_chunks', type=int, default=1,
                        help='Exact GAT inference destination batches; all source neighbors are retained.')
    parser.add_argument('--scaffold_async_wait', action='store_true',
                        help='Wait for each scheduled refresh if needed, preserving the epoch-to-support schedule.')
    parser.add_argument('--scaffold_config', type=str, default=None, help='JSON file with default SCAFFOLD experiment settings and dataset overrides.')
    parser.add_argument('--dataset', type=str, default='roman-empire')
    parser.add_argument(
        '--data_dir', '--data-root', '--data_root',
        dest='data_dir',
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f'Dataset root (default: {DEFAULT_DATA_DIR}).',
    )
    parser.add_argument(
        '--dataset-load-only', '--dataset_load_only',
        dest='dataset_load_only',
        action='store_true',
        help='Load and verify the dataset contract, then exit before sparsification or training.',
    )
    parser.add_argument(
        '--dataset-load-homophily', '--dataset_load_homophily',
        dest='dataset_load_homophily',
        action='store_true',
        help='With --dataset-load-only, compute exact node/edge homophily in bounded-memory chunks.',
    )
    parser.add_argument('--gpu', '--device', dest='gpu', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--valid_prop', type=float, default=0.25)
    parser.add_argument('--rand_split', action='store_true')
    parser.add_argument('--rand_split_class', action='store_true')
    parser.add_argument('--label_num_per_class', type=int, default=20)
    parser.add_argument('--valid_num', type=int, default=500)
    parser.add_argument('--test_num', type=int, default=1000)
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc'])
    parser.add_argument('--model', type=str, default='MPNN')
    parser.add_argument('--model_profile', type=str, default='auto', choices=['auto', 'small', 'medium', 'large', 'products', 'proteins'], help='Model family. Scaffold/full presets select the exact tunedGNN implementation family for each dataset.')
    parser.add_argument('--sparsifier', type=_canonical_sparsifier_name, default='full', choices=['full', 'fixed_support', 'support_sequence', 'mst', 'maxst', 'fast_maxst', 'effective_resistance', 'effective_resistance_fixed', 'las_vegas_spanner', 'tspanner_greedy', 'tspanner_halperin', 'k_random_neighbor', 'random', 'random_refresh', 'er', 'support', 'support_dilation', 'joint_dilation_congestion', 'clustered_joint_dilation_congestion', 'parallel_clustered_joint_dilation_congestion', 'scaffold_heap', 'scaffold_batch', 'scaffold_fast', 'scaffold_greedy', 'scaffold_sample', 'slst', 'randspt', 'llst', 'local_degree', 'rank_degree', 'gspar', 'lspar', 'lsim', 'scan', 'forest_fire', 'cut'])
    parser.add_argument(
        '--fixed_support_path', '--fixed-support-path',
        dest='fixed_support_path', type=str, default=None,
        help='Torch artifact containing the fixed undirected support to train on.',
    )
    parser.add_argument(
        '--support_sequence_path', '--support-sequence-path',
        dest='support_sequence_path', type=str, default=None,
        help='Torch artifact holding a list of supports to cycle through during '
             'training, one per resparsification step.',
    )
    parser.add_argument('--mst_algorithm', type=str, default='kruskal', choices=['kruskal', 'prim'])
    parser.add_argument('--mst_weight', type=str, default=None)
    parser.add_argument('--spanner_stretch', type=int, default=2)
    parser.add_argument('--spanner_weight', type=str, default=None)
    parser.add_argument('--effective_resistance_path', type=str, default=None)
    parser.add_argument('--las_vegas_spanner_stretch', type=int, default=3)
    parser.add_argument('--las_vegas_spanner_device', type=str, default='auto')
    parser.add_argument(
        '--las_vegas_spanner_tuning_cache', type=str, default=None,
        help=(
            'Directory for persistent per-dataset/ratio adaptive stretch choices. '
            'The final sparse graph remains in --sparsified_graph_cache_dir.'
        ),
    )
    parser.add_argument(
        '--las_vegas_spanner_networkx_max_edges', type=int, default=1_000_000,
        help=(
            'Maximum undirected edge count for adaptive multi-stretch '
            'NetworkX Baswana-Sen tuning; larger graphs use the CUDA t=3 path.'
        ),
    )
    parser.add_argument('--k_neighbors', type=int, default=5)
    parser.add_argument('--dropout_force_undirected', action='store_true')
    parser.add_argument('--er_epsilon', type=float, default=0.5)
    parser.add_argument('--er_cache_path', type=str, default=None, help='Path to cache/reuse ER values across runs.')
    parser.add_argument('--support_init', type=str, default='mst', choices=['mst', 'maxst', 'spanner', 'slst', 'randspt'])
    parser.add_argument('--support_congestion', type=str, default='node', choices=['node', 'edge'], help='Congestion objective for support sparsifier.')
    parser.add_argument('--support_sampling_mode', type=str, default='full', choices=['full', 'random', 'weighted'])
    parser.add_argument('--support_sample_size', type=int, default=1000)
    parser.add_argument('--support_verbose', action='store_true')
    parser.add_argument('--support_batch_mode', type=str, default='single', choices=['single', 'all', 'topk'], help='Batch mode for support graph: single (original), all (add all edges through congested node), topk (add top-k by dilation)')
    parser.add_argument('--support_batch_size', type=int, default=1, help='k for topk batch mode')
    parser.add_argument(
        '--joint_init_support', '--joint-init-support', '--init-support', '--init_support',
        dest='joint_init_support',
        type=str,
        default='glst',
        # Both spellings are accepted for every backbone: the historical
        # "-st" (tree) names and the "-sf" (forest) names the paper uses.
        # spanning_tree.canonical_support_name() folds them together, so a
        # forest spelling produces a byte-identical run. Bare 'sf'/'st'
        # resolve to the default deterministic forest, maxst.
        choices=[
            'mst', 'minst', 'maxst',
            'fast-mst', 'fast_mst', 'fast-minst', 'fast_minst',
            'fast-maxst', 'fast_maxst',
            'randst', 'fast-randst', 'fast_randst',
            'glst', 'slst', 'randspt', 'llst',
            # forest spellings
            'sf', 'st', 'msf', 'minsf', 'maxsf',
            'fast-msf', 'fast_msf', 'fast-minsf', 'fast_minsf',
            'fast-maxsf', 'fast_maxsf',
            'randsf', 'fast-randsf', 'fast_randsf',
            'glsf', 'slsf', 'randspf', 'llsf',
        ],
        help='Initial Scaffold support. fast-maxst/fast-mst use the parallel Benchmark bucketed Kruskal kernel.',
    )
    parser.add_argument(
        '--joint_fast_tree_buckets', '--joint-fast-tree-buckets',
        '--fast_tree_buckets', '--fast-tree-buckets',
        dest='joint_fast_tree_buckets',
        type=int,
        default=256,
        help='Priority buckets for fast-maxst/fast-mst (2..65536). More buckets improve weighted-order fidelity; default: 256.',
    )
    parser.add_argument('--joint_alpha', type=float, default=1.0, help='Exponent on dilation term for joint dilation-congestion sparsifier.')
    parser.add_argument('--joint_beta', type=float, default=1.0, help='Legacy shared congestion exponent for joint dilation-congestion; used as the fallback for both edge and node congestion when the more specific values are not provided.')
    parser.add_argument('--joint_edge_beta', type=float, default=None, help='Exponent on edge-congestion term for joint dilation-congestion sparsifier.')
    parser.add_argument('--joint_node_beta', type=float, default=None, help='Exponent on node-congestion term for joint dilation-congestion sparsifier.')
    parser.add_argument('--joint_edge_norm_p', type=float, default=2.0, help='p-norm order for normalized edge-congestion aggregation in joint dilation-congestion.')
    parser.add_argument('--joint_node_norm_q', type=float, default=2.0, help='q-norm order for normalized node-congestion aggregation in joint dilation-congestion.')
    parser.add_argument('--joint_glst_alpha', type=float, default=1.0, help='Stretch penalty exponent when joint init_support=glst.')
    parser.add_argument('--joint_glst_eta', type=float, default=1.0, help='Cut-size reward exponent when joint init_support=glst.')
    parser.add_argument('--joint_sampling_mode', type=str, default='weighted', choices=['full', 'random', 'weighted'], help='Candidate-edge sampling mode for joint dilation-congestion sparsifier.')
    parser.add_argument('--joint_sample_size', type=int, default=256, help='Number of missing edges sampled per joint dilation-congestion iteration.')
    parser.add_argument('--joint_batch_mode', type=str, default='topk', choices=['single', 'topk', 'all'], help='Batch edge-addition mode for joint dilation-congestion sparsifier.')
    parser.add_argument('--joint_batch_size', type=int, default=32, help='Edges to add per iteration when joint_batch_mode=topk.')
    parser.add_argument('--joint_growth_mode', type=str, default='exact', choices=['exact', 'lazy_topk'], help='Candidate-growth loop for joint dilation-congestion: exact sampled recomputation or lazy top-k heap refresh.')
    parser.add_argument('--joint_lazy_top_k', type=int, default=16, help='Number of stale heap candidates to refresh exactly per step when joint_growth_mode=lazy_topk.')
    parser.add_argument('--joint_lazy_rebuild_interval', type=int, default=0, help='Rebuild the sampled lazy candidate heap after this many accepted additions; 0 means rebuild only when the heap is exhausted.')
    parser.add_argument('--joint_verbose', action='store_true')
    parser.add_argument('--joint_seed', type=int, default=None, help='Optional random seed for joint sparsifier initialization/search; defaults to --seed.')
    parser.add_argument('--joint_swap_refine', action='store_true', help='After edge-addition growth, run fixed-budget add/remove swap refinement using the joint global objective.')
    parser.add_argument('--joint_swap_max_passes', type=int, default=5, help='Maximum number of improving swap passes for joint swap refinement.')
    parser.add_argument('--joint_swap_sampling_mode', type=str, default='full', choices=['full', 'random', 'weighted'], help='Candidate-edge sampling mode for joint swap refinement.')
    parser.add_argument('--joint_swap_sample_size', type=int, default=256, help='Number of missing edges sampled when evaluating swap candidates in the joint swap-refinement phase.')
    parser.add_argument('--joint_swap_cycle_sample_size', type=int, default=0, help='Maximum number of removable cycle edges evaluated per candidate add-edge during joint swap refinement; when positive, choose the highest current edge-congestion cycle edges when available; 0 means evaluate the full cycle in NetworkX mode, while tensor scaffold uses top-1 for feasibility.')
    parser.add_argument('--joint_swap_no_improve_patience', type=int, default=1, help='How many sampled swap-refinement rounds without improvement to tolerate before stopping.')
    parser.add_argument('--joint_swap_search_mode', type=str, default='restart', choices=['restart', 'lazy_topk'], help='Swap-refinement loop: restart sampled search each pass or maintain a persistent lazy top-k heap of swap add-candidates.')
    parser.add_argument('--joint_swap_lazy_top_k', type=int, default=16, help='Number of swap add-candidates to refresh exactly per pass when joint_swap_search_mode=lazy_topk.')
    parser.add_argument('--joint_swap_lazy_rebuild_interval', type=int, default=0, help='Rebuild the sampled lazy swap heap after this many accepted swaps; 0 means rebuild only when the heap is exhausted.')
    parser.add_argument('--joint_cluster_count', type=str, default='auto', help='Number of graph clusters for clustered joint sparsifiers. Use auto for ceil(undirected_edges/50000) on non-large datasets and 250 on Reddit/Pokec/OGB defaults.')
    parser.add_argument('--joint_cluster_method', type=str, default='metis', choices=['bfs', 'louvain', 'greedy', 'metis', 'label', 'node_prefix'], help='Node clustering method for clustered joint sparsifiers.')
    parser.add_argument('--joint_cluster_strategy', type=str, default='sample', choices=['heap', 'sample'], help='SCAFFOLD strategy for parallel_clustered_joint_dilation_congestion.')
    parser.add_argument('--joint_parallel_clusters', type=int, default=0, help='Maximum non-conflicting cluster proposals added per clustered lazy round; 0 means no explicit cap.')
    parser.add_argument('--joint_parallel_workers', type=int, default=0, help='Maximum worker threads for clustered SCAFFOLD; positive values are capped by available CPUs, 0 means min(available CPUs, 8).')
    parser.add_argument('--joint_cluster_add_per_round', type=int, default=16, help='Maximum edges added per active cluster per round in parallel_clustered_joint_dilation_congestion.')
    parser.add_argument('--joint_cluster_cache_dir', type=str, default=None, help='Directory for cached METIS partition assignments; tensor scaffold_fast defaults to the shared scratch-style cache derived from CACHE_ROOT or data_dir.')
    parser.add_argument('--joint_local_update_radius', type=int, default=1, help='Support-graph radius around an accepted edge path used to refresh dirty cached candidates.')
    parser.add_argument('--scaffold_sample_backbone', type=str, default='fixed-maxst', choices=['fixed-maxst', 'fixed-slst', 'rotate-randst',
                                 'fixed-maxsf', 'fixed-slsf', 'rotate-randsf',
                                 'fixed-sf', 'fixed-st', 'fixed-msf'], help='SCAFFOLD-Sample backbone unioned into every draw. fixed-maxst reuses one deterministic MaxST forest (comparable to scaffold-fast); rotate-randst cycles the precomputed random spanning forests (comparable to scaffold-fast-randst). Either way a complete spanning forest is always included, so every sample has exactly the components of the input graph.')
    parser.add_argument('--scaffold_sample_scheme', type=str, default='systematic', choices=['systematic', 'alias'], help='SCAFFOLD-Sample draw for the non-backbone budget. systematic is pi-ps over a tree-locality ordering (exact budget, exact marginal inclusion probabilities, tree-local spread); alias is with-replacement leverage-style sampling for the effective-resistance-comparable ablation.')
    parser.add_argument('--scaffold_sample_tree_count', type=int, default=8, help='Number of random spanning forests aggregated during the SCAFFOLD-Sample precompute, and the rotation length for --scaffold_sample_backbone rotate-randst.')
    parser.add_argument('--scaffold_sample_lambda', type=float, default=1.0, help='Weight of the mean normalized SCAFFOLD score relative to spanning-forest membership frequency when forming the sampling weight pi.')
    parser.add_argument('--scaffold_sample_weight_mode', choices=['legacy', 'normalized-mixture'], default='normalized-mixture', help='Sample weights: independently normalize tree frequency and support on the eligible non-backbone pool, then mix (default). Use legacy to reproduce the original distribution.')
    parser.add_argument('--scaffold_sample_mix_alpha', type=float, default=0.5, help='Support-score mixture mass in [0,1], used only by normalized-mixture. Distinct from joint_alpha (dilation).')
    parser.add_argument('--scaffold_sample_assert_connectivity', type=str, default='auto', choices=['auto', 'true', 'false'], help='Verify every draw has the same component count as the input graph. auto enables it below 5M undirected edges.')
    parser.add_argument('--scaffold_sample_allow_build', type=str, default='true', choices=['true', 'false'], help='Build and cache the SCAFFOLD-Sample artifact in-process when it is missing or stale. Set false to require scripts/precompute_scaffold_weights.py to have run first.')
    parser.add_argument('--scaffold_sample_artifact_path', type=str, default=None, help='Explicit SCAFFOLD-Sample .npz artifact path; defaults to a config-hashed file under SCAFFOLD_SCRATCH_ROOT/cache/<dataset>/scaffold_sample/.')
    parser.add_argument('--joint_dirty_limit', type=int, default=0, help='Optional cap on dirty candidates refreshed after a clustered lazy round; 0 means refresh all dirty candidates.')
    parser.add_argument('--scaffold_fast_mode', type=str, default='fast', choices=['fast', 'quality'], help='SCAFFOLD-Fast scoring mode: fast=dilation only, quality=dilation plus support load.')
    parser.add_argument('--scaffold_load_lambda', type=float, default=1.0, help='Support-load weight for SCAFFOLD-Fast quality mode.')
    parser.add_argument('--scaffold_backend', type=str, default='auto', choices=['auto', 'networkx', 'tensor'], help='SCAFFOLD Fast/Batch backend: auto uses the tensor ClusterData path; pass networkx explicitly for the small/debug path.')
    parser.add_argument('--scaffold_fast_score', type=str, default='auto', choices=['auto', 'exact', 'tree_distance', 'tree_exact', 'tree_exact_loop'], help="Scaffold Fast/Batch score mode. auto uses tree_exact (one LCA pass plus global top-k) for tensor Fast, tree_exact_loop (sample a batch, compute its dilation/congestion score, add top-r) for tensor Batch, and exact for NetworkX. tree_distance remains available only for reproducing the legacy dilation-only tensor ablation.")
    parser.add_argument('--scaffold_crossing_policy', type=str, default='balanced_owner', choices=['balanced_owner', 'drop'], help='Tensor Scaffold Fast/Batch crossing-edge policy: balanced_owner assigns each crossing edge to one endpoint partition; drop removes crossings for debugging.')
    parser.add_argument('--scaffold_support_bridge', type=str, default='auto', choices=['auto', 'true', 'false'], help='Tensor Scaffold Fast/Batch support bridge: auto adds owned crossing edges only for small target-infeasible local-support cases; true forces it; false disables it.')
    parser.add_argument('--scaffold_support_bridge_max_candidates', type=int, default=1000000, help='Maximum owned crossing-edge candidates allowed for --scaffold_support_bridge auto. Keeps Reddit-scale bridge scans from adding time; 0 disables the cap.')
    parser.add_argument(
        '--scaffold_support_budget_mode', '--scaffold-support-budget-mode',
        dest='scaffold_support_budget_mode',
        type=_support_budget_mode,
        default='full_then_random_trim',
        choices=['early_stop', 'full_then_random_trim'],
        help=(
            'How Scaffold fits its initial support to the edge budget: '
            'full_then_random_trim (default) builds the complete support and '
            'uniformly trims it to the target, matching Benchmark layer '
            'sampling; early_stop stops support construction at the target.'
        ),
    )
    parser.add_argument(
        '--scaffold_weighted_paths', '--scaffold-weighted-paths',
        dest='scaffold_weighted_paths',
        action='store_true',
        help=(
            'Measure dilation as the sum of tree edge weights along the support '
            'path instead of the hop count. The denominator w_G(e) is weighted '
            'either way; this selects the numerator. Only meaningful together '
            'with --scaffold_support_weight_method other than uniform, since '
            'under uniform weights every edge weighs 1 and the two agree.'
        ),
    )
    parser.add_argument(
        '--scaffold_support_weight_method', '--scaffold-support-weight-method',
        dest='scaffold_support_weight_method',
        type=str,
        default='uniform',
        choices=['uniform', 'cosine', 'euclidean', 'dot'],
        help=(
            'Edge quality used by weighted Scaffold MST/MaxST initializers. '
            'uniform preserves the existing default; cosine matches the '
            'default Benchmark fast-maxst weighting.'
        ),
    )
    parser.add_argument(
        '--eval_graph', '--eval-graph',
        dest='eval_graph',
        type=str,
        default=None,
        choices=['sparse', 'original', 'fast-maxst'],
        help=(
            'Graph used for validation/test inference for every graph method. '
            'original evaluates the sparsely trained model on the original '
            'full topology; fast-maxst evaluates on one fixed deterministic '
            'cosine-weighted Fast-MaxST forest; sparse preserves the training-graph evaluation '
            'protocol. If omitted, the legacy --scaffold_eval_graph setting '
            'controls Scaffold and all other methods use sparse evaluation.'
        ),
    )
    parser.add_argument(
        '--scaffold_eval_graph', '--scaffold-eval-graph',
        dest='scaffold_eval_graph',
        type=str,
        default='sparse',
        choices=['sparse', 'original'],
        help=(
            'Graph used for Scaffold validation/test inference. sparse keeps '
            'the existing default; original matches Benchmark evaluation. '
            'Original-graph evaluation currently applies to full-batch mode.'
        ),
    )
    parser.add_argument(
        '--scaffold_eval_ensemble_size',
        type=int,
        default=1,
        help=(
            'Number of sparse graph views used for Scaffold evaluation. '
            'Scaffold-Fast defaults to the five validation-best training '
            'graphs. With source=fresh, 1 preserves direct evaluation on the '
            'active graph; with source=best, 1 evaluates on the single '
            'validation-best training graph. Explicit graph views require '
            '--scaffold_eval_graph sparse.'
        ),
    )
    parser.add_argument(
        '--scaffold_eval_ensemble_source',
        type=str,
        default='fresh',
        choices=['fresh', 'best'],
        help=(
            'Sparse views for an evaluation ensemble: fresh generates '
            'evaluation-only resparsifications with a disjoint seed stream; '
            'best retains training graphs with the highest validation score '
            'observed when each graph was active.'
        ),
    )
    parser.add_argument(
        '--scaffold_eval_ensemble_reduce',
        type=str,
        default='mean-logits',
        choices=['mean-logits', 'mean-probabilities', 'majority-vote', 'union'],
        help=(
            'Combine sparse evaluation views by averaging independent logits, '
            'soft-voting with mean class probabilities, hard class majority '
            'vote, or one forward pass over their unweighted edge union. '
            'Union can exceed the target edge ratio.'
        ),
    )
    parser.add_argument(
        '--scaffold_eval_graph_bank_dir', '--scaffold-eval-graph-bank-dir',
        dest='scaffold_eval_graph_bank_dir',
        type=str,
        default='auto',
        help=(
            'Scratch directory for the retained validation-best sparse '
            'training graphs. auto uses SUPPORT_GRAPH_CACHE_DIR when set, '
            'otherwise SCAFFOLD_SCRATCH_ROOT/cache/<dataset>/eval_graph_bank. '
            'Use none to disable persistence.'
        ),
    )
    parser.add_argument(
        '--scaffold_final_eval_views', nargs='+', choices=['1', '3', '5', '10', 'all'],
        default=None,
        help='Opt-in: train once, rank distinct training supports on validation, then evaluate the final model on nested best-1/3/5 unions and/or the original full graph.',
    )
    parser.add_argument('--scaffold_final_eval_csv', default=None,
                        help='Isolated per-run CSV for the final-only multiview results.')
    parser.add_argument('--scaffold_final_graph_bank_size', type=int, default=None,
                        help='Opt-in final evaluation bank capacity; otherwise inferred from requested views.')
    parser.add_argument('--scaffold_final_reselect_best1', action='store_true',
                        help='Re-score every retained support on validation with the frozen final model, then choose best-1. Off by default; requires final multiview evaluation.')
    parser.add_argument('--scaffold_final_report_historical1', action='store_true',
                        help='Also report historical-best-1 with that same final model as a control; requires final best-1 re-selection.')
    parser.add_argument('--scaffold_final_report_best_checkpoint', action='store_true',
                        help='Opt-in: also save/evaluate the validation-best model and its matching single support. Test maxima are diagnostics only; final-model controls remain unless explicitly suppressed.')
    parser.add_argument('--scaffold_exact_epochs', action='store_true',
                        help='Opt-in Scaffold campaigns: execute exactly --epochs iterations, including Reddit/Products. Legacy tunedGNN epoch conventions remain unchanged otherwise.')
    parser.add_argument('--scaffold_final_best_single_only', action='store_true',
                        help='Opt-in: suppress last-model single-graph inference and report only its validation-best model/support checkpoint. Requires best-checkpoint reporting; full evaluation has its own independent policy.')
    parser.add_argument('--scaffold_final_best_full_checkpoint', action='store_true',
                        help='Opt-in: evaluate the original full graph during training, save its own validation-best model, and report that checkpoint instead of last-model full evaluation. Requires best-checkpoint reporting and final view all.')
    parser.add_argument('--scaffold_final_best_ensemble_views', nargs='+', choices=['3', '5', '10'], default=None,
                        help='Opt-in: track a separate validation-best model and exact graph-union snapshot for each requested k; report these in addition to last-model controls. Requires best-checkpoint reporting.')
    parser.add_argument('--scaffold_final_prediction_ensembles', action='store_true',
                        help='Opt-in: separately forward retained sparse supports and track validation-best prediction ensembles.')
    parser.add_argument('--scaffold_prediction_ks', type=int, nargs='+', choices=[3, 5, 10], default=[3, 5])
    parser.add_argument('--scaffold_prediction_reducers', nargs='+', choices=['mean-logits', 'majority-vote'],
                        default=['mean-logits'], help='Majority voting is disabled unless explicitly requested (Questions only).')
    parser.add_argument('--scaffold_final_skip_last_controls', action='store_true',
                        help='Report only validation-best checkpoints; skip redundant last-model unions.')
    parser.add_argument('--scaffold_final_budgeted_union', action='store_true',
                        help='Opt-in final-only experiment: snapshot constituents of each winning k-union, then frequency-first prune to q using original-graph Scaffold support scores. Requires best-ensemble checkpoints; training and ordinary union/full outputs are unchanged.')
    parser.add_argument('--scaffold_protocol_run_ids', type=int, nargs='+', default=None,
                        help='Opt-in distributed checkpoint pilot: execute only these one-based run IDs, preserving their fixed split indices. Uses independent seed+run-index replicas; normal run policy is unchanged.')
    parser.add_argument(
        '--scaffold_consensus_mode', '--scaffold-consensus-mode',
        dest='scaffold_consensus_mode',
        type=str,
        default='off',
        choices=['off', 'fixed', 'validation'],
        help=(
            'Final-only exact-budget Scaffold inference. fixed constructs one '
            'candidate with --scaffold_consensus_lambda; validation constructs '
            'lambda={0,0.5,1}, includes Scaffold-1, selects using validation '
            'only, and performs one final test-bearing forward on the winner.'
        ),
    )
    parser.add_argument(
        '--scaffold_consensus_k', '--scaffold-consensus-k',
        dest='scaffold_consensus_k',
        type=int,
        default=5,
        help='Requested number of distinct validation-ranked training supports.',
    )
    parser.add_argument(
        '--scaffold_consensus_lambda', '--scaffold-consensus-lambda',
        dest='scaffold_consensus_lambda',
        type=float,
        default=0.5,
        help='Frequency/Fast percentile mixing weight for fixed Consensus mode.',
    )
    parser.add_argument(
        '--scaffold_consensus_q', '--scaffold-consensus-q',
        dest='scaffold_consensus_q',
        type=int,
        default=None,
        help=(
            'Authoritative canonical undirected non-self-loop edge budget. '
            'When supplied it must agree with ceil(target_ratio*m) under the '
            'repository budget convention.'
        ),
    )
    parser.add_argument(
        '--scaffold_consensus_output_dir', '--scaffold-consensus-output-dir',
        dest='scaffold_consensus_output_dir',
        type=str,
        default='auto',
        help=(
            'Directory for selected q-edge graph artifacts and metadata. auto '
            'places them next to the persistent best-support bank.'
        ),
    )
    parser.add_argument(
        '--report_final_epoch_only', '--report-final-epoch-only',
        dest='report_final_epoch_only',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Report validation/test metrics only for the final trained model. '
            'Unlike --eval_step 0, intermediate evaluations may still run; '
            'this is needed to rank training supports for a validation-best '
            'sparse-graph ensemble before evaluating that ensemble once at '
            'the final epoch.'
        ),
    )
    parser.add_argument('--scaffold_progress', type=str, default='auto', choices=['auto', 'true', 'false'], help='Show efficient tqdm progress bars for tensor Scaffold Fast/Batch support-tree and growth stages.')
    parser.add_argument('--metis_recompute', action='store_true', help='For tensor scaffold_fast, delete the matching METIS ClusterData and edge-owner assignment caches before recomputing them.')
    parser.add_argument('--scaffold_metis_edge_sample_size', type=int, default=0, help='Deprecated for tensor SCAFFOLD-Fast ClusterData backend; retained for old commands and ignored.')
    parser.add_argument('--scaffold_resparsify_every', type=int, default=1, help='Rebuild the sparse graph every K training epochs (Scaffold Greedy/Heap/Fast), starting after the initial before-epoch-0 graph and skipping the final K-epoch window. 1 (default) selects a freshly seeded sparse graph every scheduled epoch; 0 precomputes a single sparse graph at the start of the run.')
    parser.add_argument(
        '--scaffold_fast_maxst_every', '--scaffold-fast-maxst-every',
        dest='scaffold_fast_maxst_every',
        type=int,
        default=0,
        help=(
            'Use a Fast-MaxST initializer on every Nth Scaffold refresh while '
            'retaining the configured initializer on all other refreshes. '
            'For Scaffold-Sample, the analogous schedule uses its fixed '
            'deterministic MaxST backbone every Nth draw and its configured '
            'backbone otherwise. 0 (default) disables the mixed-backbone '
            'schedule. This schedule is synchronous.'
        ),
    )
    parser.add_argument(
        '--scaffold_resparsify_include_final',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Also rebuild in the final partial K-epoch window. With '
            '--scaffold_resparsify_every 1, this gives every training epoch '
            'its own graph (the epoch-0 graph is the initial blocking build). '
            'Scaffold-Fast enables this through its default configuration; '
            'use --no-scaffold_resparsify_include_final to disable it.'
        ),
    )
    parser.add_argument(
        '--scaffold_async_resparsify',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Build future SCAFFOLD graphs in one background CPU process and '
            'swap them in when ready. Enabled by default; use '
            '--no-scaffold_async_resparsify for blocking rebuilds, which keep '
            'every epoch on the graph built for it but spend foreground time '
            'that grows sharply with edge density.'
        ),
    )
    parser.add_argument('--scaffold_resparsify_reseed', type=str, default='true', choices=['true', 'false'], help='When rebuilding the sparse graph mid-training, reseed the sparsifier RNGs from base_seed + epoch // resparsify_every. Set to false to keep the original seed and rely on other stochastic sources.')
    parser.add_argument('--llst_max_passes', type=int, default=10, help='Maximum number of exact improving edge-swap passes for the local-search low-stretch tree.')
    parser.add_argument(
        '--llst_init_support',
        type=str,
        default='glst',
        choices=[
            'glst', 'maxst', 'mst', 'minst',
            'fast-maxst', 'fast_maxst',
            'fast-mst', 'fast_mst', 'fast-minst', 'fast_minst',
            'randst', 'randspt',
            # forest spellings, folded by canonical_support_name()
            'glsf', 'maxsf', 'msf', 'minsf', 'sf', 'st',
            'fast-maxsf', 'fast_maxsf',
            'fast-msf', 'fast_msf', 'fast-minsf', 'fast_minsf',
            'randsf', 'randspf',
        ],
        help='Initializer tree used before LLST local swaps.',
    )
    parser.add_argument('--llst_glst_alpha', type=float, default=1.0, help='Stretch exponent for the GLST initializer used inside LLST.')
    parser.add_argument('--llst_glst_eta', type=float, default=1.0, help='Cut-size exponent for the GLST initializer used inside LLST.')
    parser.add_argument('--llst_candidate_strategy', type=str, default='random', choices=['random', 'tree_distance'], help='How LLST picks add-edge candidates before exact sampled evaluation.')
    parser.add_argument('--llst_candidate_sample_size', type=int, default=0, help='Sample at most this many non-tree candidate add-edges per LLST pass; 0 means evaluate all candidates.')
    parser.add_argument('--llst_eval_sample_size', type=int, default=0, help='Approximate LLST total stretch on a sampled subset of original graph edges; 0 means evaluate all edges exactly.')
    parser.add_argument('--llst_cycle_sample_size', type=int, default=0, help='Sample at most this many removable cycle edges for each LLST add-edge candidate; 0 means evaluate the full cycle.')
    parser.add_argument('--llst_resample_eval_each_pass', action='store_true', help='Refresh the sampled LLST evaluation edge set at every pass so longer local-search runs do not overfit one fixed sample.')
    parser.add_argument('--llst_verbose', action='store_true', help='Verbose logging for the local-search low-stretch tree builder.')
    parser.add_argument('--slst_num_roots', type=int, default=8, help='Number of candidate roots for the scalable low-stretch tree heuristic.')
    parser.add_argument('--slst_eval_sample_size', type=int, default=8192, help='Number of original edges sampled to estimate average stretch for each candidate root; use 0 for exact-on-all-edges.')
    parser.add_argument('--slst_exact_eval_threshold', type=int, default=20000, help='Use exact evaluation when a connected component has at most this many edges.')
    parser.add_argument('--slst_fast', action='store_true', help='Skip the Tarjan-LCA stretch scoring in SLST (uses the top max-degree root only). ~10x faster on medium graphs; sensible when SLST is only an init for downstream sparsifiers.')
    parser.add_argument('--cut_epsilon', type=float, default=10.0)
    parser.add_argument('--cut_d', type=int, default=1)
    parser.add_argument('--cut_connectivity', type=int, default=10)
    parser.add_argument('--cut_seed', type=int, default=None)
    parser.add_argument('--rank_degree_rho', type=float, default=0.1)
    parser.add_argument('--lspar_e', type=float, default=0.6, help='L-Spar degree exponent (default: 0.6)')
    parser.add_argument('--gnn', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'])
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--local_layers', type=int, default=7)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--pre_ln', action='store_true')
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
    parser.add_argument('--jk', action='store_true')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--in_dropout', type=float, default=0.15, help='Input dropout used by the tunedGNN large-graph model.')
    parser.add_argument('--gpu_memory_profile', type=str, default='auto', choices=['auto', '80gb', 'conservative'], help='GPU memory tuning profile. auto treats CUDA devices with at least 60GB as the 80gb profile.')
    parser.add_argument('--display_step', type=int, default=100)
    parser.add_argument('--eval_step', type=int, default=1, help='Evaluate every N epochs; final epoch is always evaluated. Use 0 for final-only evaluation.')
    parser.add_argument('--eval_start_epoch', type=int, default=1, help='First one-based epoch eligible for evaluation (tunedGNN Pokec uses 1002).')
    parser.add_argument('--log_every', type=int, default=10, help='Logging cadence used by tunedGNN large-graph presets.')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'])
    parser.add_argument('--lr_scheduler', type=str, default='none', choices=['none', 'plateau'])
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.75)
    parser.add_argument('--lr_scheduler_patience', type=int, default=50)
    parser.add_argument('--tunedgnn_strict', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--train_mode', type=str, default='full', choices=['full', 'neighbor'], help='Training loop: full uses full-batch message passing; neighbor uses PyG NeighborLoader minibatches.')
    parser.add_argument('--large_graph_pipeline', type=str, default='auto', choices=['auto', 'full', 'node_batch', 'random_node_loader', 'neighbor'], help='Large-graph training loop. auto matches tunedGNN large_graph conventions per dataset.')
    parser.add_argument('--large_batch_size', type=str, default='auto', help='Node batch size for tunedGNN-style large_graph/main-batch training. Use auto/adaptive to estimate from available GPU memory.')
    parser.add_argument('--large_batch_mem_fraction', type=float, default=0.85, help='Fraction of currently free CUDA memory to target when --large_batch_size auto is used.')
    parser.add_argument('--large_batch_min_size', type=int, default=1024, help='Minimum node batch size when --large_batch_size auto is used.')
    parser.add_argument('--large_batch_max_size', type=int, default=0, help='Optional maximum node batch size when --large_batch_size auto is used; 0 means no cap beyond num_nodes.')
    parser.add_argument('--large_train_parts', type=str, default='auto', help='RandomNodeLoader train partitions for ogbn-products-style training. Use auto/adaptive to estimate from available GPU memory, or pass an integer.')
    parser.add_argument('--large_eval_parts', type=str, default='auto', help='RandomNodeLoader evaluation partitions for ogbn-products-style evaluation. Use auto/adaptive to estimate from available GPU memory, or pass an integer.')
    parser.add_argument('--large_loader_workers', type=int, default=0, help='Worker processes for large-graph RandomNodeLoader. Default 0 avoids multiprocessing cleanup assertions on clusters.')
    parser.add_argument(
        '--eval_decomposed_layers', '--eval-decomposed-layers',
        dest='eval_decomposed_layers', type=int, default=1,
        help=(
            'Exact GCN evaluation feature decomposition. Values above 1 split '
            'message features into this many GPU chunks, reducing peak memory '
            'without changing the model, graph, weights, or predictions.'
        ),
    )
    parser.add_argument('--large_eval_cpu', dest='large_eval_cpu', action='store_true', help='Evaluate large node-batch runs on CPU, matching tunedGNN main-batch.')
    parser.add_argument('--no_large_eval_cpu', dest='large_eval_cpu', action='store_false')
    parser.set_defaults(large_eval_cpu=True)
    parser.add_argument('--amp', action='store_true', help='Use CUDA BF16 autocast for model forward/loss. Useful on H100/A100 large-graph runs.')
    parser.add_argument('--neighbor_batch_size', type=int, default=1024, help='Seed-node batch size for neighbor minibatch training.')
    parser.add_argument('--neighbor_eval_batch_size', type=int, default=2048, help='Seed-node batch size for neighbor minibatch evaluation.')
    parser.add_argument('--neighbor_num_neighbors', type=str, default='10,10,5', help='Comma-separated per-layer neighbor fanouts for NeighborLoader, or -1/all for full neighborhoods.')
    parser.add_argument('--neighbor_eval_num_neighbors', type=str, default='', help='Optional evaluation fanouts; empty reuses --neighbor_num_neighbors.')
    parser.add_argument('--neighbor_workers', type=int, default=0, help='NeighborLoader worker processes.')
    parser.add_argument('--neighbor_tmp_dir', type=str, default=None, help='Temporary directory for neighbor-loader worker files; use scratch on NFS-backed clusters.')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--model_dir', type=str, default='./model/')
    parser.add_argument(
        '--eval_only_checkpoint', '--eval-only-checkpoint',
        dest='eval_only_checkpoint',
        type=str,
        default=None,
        help=(
            'Skip training, load this model_state_dict checkpoint, and perform '
            'one evaluation forward on the configured graph. Intended for '
            'reloading a saved Scaffold-Consensus fixed-support artifact.'
        ),
    )
    parser.add_argument(
        '--eval_only_run', '--eval-only-run',
        dest='eval_only_run',
        type=int,
        default=1,
        help='One-based split/run identity used by --eval_only_checkpoint.',
    )
    parser.add_argument('--target_ratio', type=float, default=0.2, help='Target edge keep ratio after sparsification.')
    parser.add_argument(
        '--sparsified_graph_cache_dir', '--sparsified-graph-cache-dir',
        dest='sparsified_graph_cache_dir',
        type=str,
        default=None,
        help=(
            'Persistent cache root for a single sparsified training graph. '
            'Entries are keyed by dataset, method, ratio, seed, split, source '
            'graph signature, and sparsifier parameters.'
        ),
    )
    parser.add_argument(
        '--recompute_sparsified_graph', '--recompute-sparsified-graph',
        dest='recompute_sparsified_graph',
        action='store_true',
        help='Ignore and replace a matching persistent sparse-graph cache entry.',
    )
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

def get_sparsifier_params(args, num_nodes=None, num_edges=None):
    main_seed = args.seed
    params = {}
    if num_nodes is None or num_edges is None:
        raise ValueError('num_nodes and num_edges must be provided for dynamic sparsification calculation')
    keep_ratio = max(0.0, min(1.0, args.target_ratio))
    if args.sparsifier == 'mst':
        # A minimum spanning forest has |V|-components edges and therefore no
        # target-ratio budget. Its achieved ratio is measured after creation.
        params = {'algorithm': args.mst_algorithm, 'weight': args.mst_weight}
    elif args.sparsifier == 'fixed_support':
        import hashlib
        from pathlib import Path

        if not args.fixed_support_path:
            raise ValueError('--fixed_support_path is required for fixed_support')
        support_path = Path(args.fixed_support_path).expanduser().resolve()
        if not support_path.is_file():
            raise FileNotFoundError(f'fixed support does not exist: {support_path}')
        digest = hashlib.sha256(support_path.read_bytes()).hexdigest()
        params = {
            'support_path': str(support_path),
            'support_sha256': digest,
            'target_ratio': keep_ratio,
        }
    elif args.sparsifier == 'support_sequence':
        import hashlib
        from pathlib import Path

        if not args.support_sequence_path:
            raise ValueError(
                '--support_sequence_path is required for support_sequence'
            )
        sequence_path = Path(args.support_sequence_path).expanduser().resolve()
        if not sequence_path.is_file():
            raise FileNotFoundError(
                f'support sequence does not exist: {sequence_path}'
            )
        # 'seed' is the base main.py adds its per-run refresh counter to, so the
        # sparsifier can turn a refresh seed back into a sequence index. Pinning
        # it to 0 makes index == refresh number.
        params = {
            'sequence_path': str(sequence_path),
            'sequence_sha256': hashlib.sha256(sequence_path.read_bytes()).hexdigest(),
            'target_ratio': keep_ratio,
            'seed': 0,
        }
    elif args.sparsifier == 'fast_maxst':
        params = {
            'weight_method': 'cosine',
            'bucket_count': args.joint_fast_tree_buckets,
            'parallel_workers': args.joint_parallel_workers,
        }
    elif args.sparsifier in {'effective_resistance', 'effective_resistance_fixed'}:
        from scaffold_gnn.sparsifiers.effective_resistance import (
            effective_resistance_artifact,
            effective_resistance_artifact_signature,
        )

        artifact = args.effective_resistance_path or effective_resistance_artifact(
            args.data_dir, args.dataset
        )
        params = {
            'target_ratio': keep_ratio,
            'seed': main_seed,
            'artifact_path': str(artifact),
            'artifact_signature': effective_resistance_artifact_signature(artifact),
            'parallel_workers': args.joint_parallel_workers,
        }
    elif args.sparsifier == 'las_vegas_spanner':
        params = {
            'target_ratio': keep_ratio,
            'stretch': args.las_vegas_spanner_stretch,
            'seed': main_seed,
            'parallel_workers': args.joint_parallel_workers,
            'device': args.las_vegas_spanner_device,
            'dataset': args.dataset,
            'tuning_cache_path': args.las_vegas_spanner_tuning_cache,
            'networkx_max_edges': args.las_vegas_spanner_networkx_max_edges,
        }
    elif args.sparsifier == 'maxst':
        params = {'algorithm': args.mst_algorithm, 'weight': args.mst_weight, 'target_ratio': keep_ratio}
    elif args.sparsifier == 'tspanner_greedy':
        params = {'stretch': args.spanner_stretch, 'seed': main_seed}
    elif args.sparsifier == 'tspanner_halperin':
        params = {'stretch': args.spanner_stretch, 'seed': main_seed}
    elif args.sparsifier == 'k_random_neighbor':
        params = {'k': args.k_neighbors, 'seed': main_seed}
    elif args.sparsifier in {'random', 'random_refresh'}:
        params = {'drop_prob': 1.0 - keep_ratio, 'target_ratio': keep_ratio, 'seed': main_seed}
    elif args.sparsifier == 'er':
        params = {'epsilon': args.er_epsilon, 'er_cache_path': args.er_cache_path}
    elif args.sparsifier == 'support':
        params = {
            'target_ratio': keep_ratio,
            'init_support': args.support_init,
            'node_congestion': args.support_congestion == 'node',
            'sampling_mode': args.support_sampling_mode,
            'sample_size': args.support_sample_size,
            'verbose': args.support_verbose,
            'seed': main_seed,
            'batch_mode': args.support_batch_mode,
            'batch_size': args.support_batch_size,
            'slst_num_roots': args.slst_num_roots,
            'slst_eval_sample_size': args.slst_eval_sample_size,
            'slst_exact_eval_threshold': args.slst_exact_eval_threshold,
        }
    elif args.sparsifier == 'support_dilation':
        params = {
            'target_ratio': keep_ratio,
            'init_support': args.support_init,
            'sampling_mode': args.support_sampling_mode,
            'sample_size': args.support_sample_size,
            'verbose': args.support_verbose,
            'seed': main_seed,
            'slst_num_roots': args.slst_num_roots,
            'slst_eval_sample_size': args.slst_eval_sample_size,
            'slst_exact_eval_threshold': args.slst_exact_eval_threshold,
        }
    elif args.sparsifier in ('joint_dilation_congestion', 'clustered_joint_dilation_congestion', 'parallel_clustered_joint_dilation_congestion', 'scaffold_heap', 'scaffold_batch', 'scaffold_fast', 'scaffold_greedy', 'scaffold_sample'):
        joint_seed = main_seed if args.joint_seed is None else args.joint_seed
        resolved_backend = resolve_scaffold_backend(args) if args.sparsifier in {'scaffold_fast', 'scaffold_batch'} else None
        resolved_fast_score = (
            resolve_scaffold_fast_score(args, resolved_backend)
            if args.sparsifier in {'scaffold_fast', 'scaffold_batch'}
            else None
        )
        params = {
            'target_ratio': keep_ratio,
            'init_support': args.joint_init_support,
            'fast_tree_buckets': args.joint_fast_tree_buckets,
            'alpha': args.joint_alpha,
            'beta': args.joint_beta,
            'edge_beta': args.joint_edge_beta,
            'node_beta': args.joint_node_beta,
            'edge_norm_p': args.joint_edge_norm_p,
            'node_norm_q': args.joint_node_norm_q,
            'glst_alpha': args.joint_glst_alpha,
            'glst_eta': args.joint_glst_eta,
            'sampling_mode': args.joint_sampling_mode,
            'sample_size': args.joint_sample_size,
            'batch_mode': args.joint_batch_mode,
            'batch_size': args.joint_batch_size,
            'growth_mode': args.joint_growth_mode,
            'lazy_top_k': args.joint_lazy_top_k,
            'lazy_rebuild_interval': args.joint_lazy_rebuild_interval,
            'swap_refine': args.joint_swap_refine,
            'swap_max_passes': args.joint_swap_max_passes,
            'swap_sampling_mode': args.joint_swap_sampling_mode,
            'swap_sample_size': args.joint_swap_sample_size,
            'swap_cycle_sample_size': args.joint_swap_cycle_sample_size,
            'swap_no_improve_patience': args.joint_swap_no_improve_patience,
            'swap_search_mode': args.joint_swap_search_mode,
            'swap_lazy_top_k': args.joint_swap_lazy_top_k,
            'swap_lazy_rebuild_interval': args.joint_swap_lazy_rebuild_interval,
            'slst_num_roots': args.slst_num_roots,
            'slst_eval_sample_size': args.slst_eval_sample_size,
            'slst_exact_eval_threshold': args.slst_exact_eval_threshold,
            'slst_fast': bool(args.slst_fast),
            'llst_max_passes': args.llst_max_passes,
            'llst_init_support': args.llst_init_support,
            'llst_glst_alpha': args.llst_glst_alpha,
            'llst_glst_eta': args.llst_glst_eta,
            'llst_candidate_strategy': args.llst_candidate_strategy,
            'llst_candidate_sample_size': args.llst_candidate_sample_size,
            'llst_eval_sample_size': args.llst_eval_sample_size,
            'llst_cycle_sample_size': args.llst_cycle_sample_size,
            'llst_resample_eval_each_pass': args.llst_resample_eval_each_pass,
            'llst_verbose': args.llst_verbose,
            'verbose': args.joint_verbose,
            'seed': joint_seed,
        }
        if args.sparsifier in ('clustered_joint_dilation_congestion', 'parallel_clustered_joint_dilation_congestion', 'scaffold_heap', 'scaffold_batch', 'scaffold_fast'):
            dataset_cache_dir = args.joint_cluster_cache_dir
            if dataset_cache_dir is None:
                dataset_cache_dir = default_cluster_cache_dir(args, backend=resolved_backend)
            cluster_method = args.joint_cluster_method
            if args.sparsifier in {'scaffold_fast', 'scaffold_batch'} and resolved_backend == 'tensor':
                cluster_method = 'metis'
            cluster_count = resolve_joint_cluster_count(args, num_edges=num_edges)
            params.update({
                'cluster_count': cluster_count,
                'cluster_method': cluster_method,
                'parallel_clusters': args.joint_parallel_clusters,
                'cluster_cache_dir': dataset_cache_dir,
                'local_update_radius': args.joint_local_update_radius,
                'dirty_limit': args.joint_dirty_limit,
            })
        if args.sparsifier in ('scaffold_greedy', 'scaffold_heap', 'scaffold_batch', 'scaffold_fast', 'scaffold_sample', 'parallel_clustered_joint_dilation_congestion'):
            params.update({
                'support_budget_mode': args.scaffold_support_budget_mode,
                'support_weight_method': args.scaffold_support_weight_method,
                'weighted_paths': bool(args.scaffold_weighted_paths),
                'resparsify_every': int(args.scaffold_resparsify_every),
                'resparsify_reseed': str(args.scaffold_resparsify_reseed).lower() == 'true',
            })
        if args.sparsifier == 'scaffold_sample':
            params.update({
                'parallel_workers': args.joint_parallel_workers,
                'sample_backbone': args.scaffold_sample_backbone,
                'sample_scheme': args.scaffold_sample_scheme,
                'sample_tree_count': int(args.scaffold_sample_tree_count),
                'sample_lambda': float(args.scaffold_sample_lambda),
                'sample_assert_connectivity': args.scaffold_sample_assert_connectivity,
                'sample_allow_build': str(args.scaffold_sample_allow_build).lower() == 'true',
                'sample_artifact_path': args.scaffold_sample_artifact_path,
                # Canonical, not raw: the artifact lives under the
                # canonical key (cache/ogbn-arxiv/...), and a wrapper that
                # passes "OGB-arxiv" was looking in cache/ogb-arxiv/ and
                # failing with "artifact is missing" on four of five graphs.
                'sample_dataset': _canonical_dataset_key(args.dataset),
            })
            # Always pass the mode: explicit legacy must override the new class default.
            params['sample_weight_mode'] = args.scaffold_sample_weight_mode
            if args.scaffold_sample_weight_mode != 'legacy':
                params['sample_mix_alpha'] = float(args.scaffold_sample_mix_alpha)
        if args.sparsifier in ('parallel_clustered_joint_dilation_congestion', 'scaffold_heap', 'scaffold_batch', 'scaffold_fast'):
            cluster_strategy = args.joint_cluster_strategy
            if args.sparsifier == 'scaffold_heap':
                cluster_strategy = 'heap'
            elif args.sparsifier in {'scaffold_fast', 'scaffold_batch'}:
                cluster_strategy = 'sample'
            params.update({
                'parallel_workers': args.joint_parallel_workers,
                'cluster_add_per_round': args.joint_cluster_add_per_round,
            })
            if args.sparsifier == 'parallel_clustered_joint_dilation_congestion':
                params['cluster_strategy'] = cluster_strategy
            if cluster_strategy == 'sample':
                params.update({
                    'fast_mode': args.scaffold_fast_mode,
                    'fast_load_lambda': args.scaffold_load_lambda,
                    'backend': resolved_backend,
                    'fast_score': resolved_fast_score,
                    'metis_edge_sample_size': args.scaffold_metis_edge_sample_size,
                    'crossing_policy': args.scaffold_crossing_policy,
                    'support_bridge': args.scaffold_support_bridge,
                    'support_bridge_max_candidates': args.scaffold_support_bridge_max_candidates,
                    'progress': args.scaffold_progress,
                    'metis_recompute': args.metis_recompute,
                })
    elif args.sparsifier == 'slst':
        params = {
            'target_ratio': keep_ratio,
            'num_roots': args.slst_num_roots,
            'eval_sample_size': args.slst_eval_sample_size,
            'exact_eval_threshold': args.slst_exact_eval_threshold,
            'seed': main_seed,
            'verbose': args.support_verbose,
            'fast_mode': bool(args.slst_fast),
        }
    elif args.sparsifier == 'llst':
        params = {
            'target_ratio': keep_ratio,
            'max_passes': args.llst_max_passes,
            'init_support': args.llst_init_support,
            'glst_alpha': args.llst_glst_alpha,
            'glst_eta': args.llst_glst_eta,
            'candidate_strategy': args.llst_candidate_strategy,
            'candidate_sample_size': args.llst_candidate_sample_size,
            'eval_sample_size': args.llst_eval_sample_size,
            'cycle_sample_size': args.llst_cycle_sample_size,
            'resample_eval_each_pass': args.llst_resample_eval_each_pass,
            'seed': main_seed,
            'verbose': args.llst_verbose,
        }
    elif args.sparsifier == 'randspt':
        params = {'seed': main_seed, 'target_ratio': keep_ratio}
    elif args.sparsifier == 'local_degree':
        params = {'target_ratio': keep_ratio}
    elif args.sparsifier == 'rank_degree':
        params = {'rho': args.rank_degree_rho, 'seed': main_seed, 'target_ratio': keep_ratio}
    elif args.sparsifier == 'gspar':
        params = {'target_ratio': keep_ratio}
    elif args.sparsifier == 'lspar':
        params = {'target_ratio': keep_ratio, 'e': args.lspar_e}
    elif args.sparsifier == 'lsim':
        params = {'target_ratio': keep_ratio}
    elif args.sparsifier == 'scan':
        params = {'target_ratio': keep_ratio}
    elif args.sparsifier == 'forest_fire':
        params = {'target_ratio': keep_ratio, 'seed': main_seed}
    elif args.sparsifier == 'cut':
        cut_seed = args.cut_seed if args.cut_seed is not None else main_seed
        params = {'epsilon': args.cut_epsilon, 'd': args.cut_d, 'connectivity': args.cut_connectivity, 'seed': cut_seed}
    return params

def set_seed_for_run(args, run):
    run_seed = args.seed + run
    set_seed(run_seed)
