import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Coauthor, HeterophilousGraphDataset, WikiCS, Planetoid, KarateClub
import numpy as np
import scipy.io
import scipy.sparse as sp
import os
import hashlib
from os import path
from pathlib import Path
import shutil
import urllib.request
import zipfile
from .data_utils import (
    balanced_train_valid_test_idx,
    dataset_drive_url,
    rand_train_test_idx,
)


def canonical_split_fingerprint(split):
    """Hash split membership exactly like Benchmark/tunedGNN comparison runs."""
    hasher = hashlib.sha256()
    for key in ('train', 'valid', 'test'):
        value = torch.as_tensor(split[key]).detach().cpu()
        if value.dtype == torch.bool:
            value = torch.where(value.reshape(-1))[0]
        else:
            value = value.reshape(-1).long()
        value = value.sort().values.contiguous()
        hasher.update(key.encode('utf-8'))
        hasher.update(value.numpy().tobytes())
    return hasher.hexdigest()

class NCDataset(object):

    def __init__(self, name):
        self.name = name
        self.data_dir = None
        self.graph = {}
        self.label = None
        self.train_idx = None
        self.valid_idx = None
        self.test_idx = None
        self.split_strategy = 'random'
        self.train_nodes_per_class = None
        self.balanced_valid_num = None

    def get_idx_split(self, split_type='random', train_prop=0.5, valid_prop=0.25):
        if self.train_idx is not None:
            return {'train': self.train_idx, 'valid': self.valid_idx, 'test': self.test_idx}
        ignore_negative = False if self.name == 'ogbn-proteins' else True
        split_seed = int(os.environ.get('SUPPORT_GRAPH_SPLIT_SEED', '123'))
        cache_path = None
        if self.data_dir:
            safe_name = canonicalize_dataset_name(self.name).replace('/', '-')
            ratio_tag = f'train{train_prop:.6f}_valid{valid_prop:.6f}'.replace('.', 'p')
            if self.split_strategy == 'balanced_train':
                strategy_tag = (
                    f'_balanced_train{int(self.train_nodes_per_class)}'
                    f'_valid{int(self.balanced_valid_num)}'
                )
            else:
                strategy_tag = ''
            cache_path = (
                Path(self.data_dir)
                / 'splits'
                / f'{safe_name}{strategy_tag}_{ratio_tag}_seed{split_seed}.pt'
            )
            if cache_path.exists():
                split = torch.load(cache_path, map_location='cpu', weights_only=True)
                if all(key in split for key in ('train', 'valid', 'test')):
                    return {key: torch.as_tensor(split[key]).long() for key in ('train', 'valid', 'test')}
        generator = torch.Generator(device='cpu')
        generator.manual_seed(split_seed)
        if self.split_strategy == 'balanced_train':
            train_idx, valid_idx, test_idx = balanced_train_valid_test_idx(
                self.label,
                train_nodes_per_class=int(self.train_nodes_per_class),
                valid_num=int(self.balanced_valid_num),
                ignore_negative=ignore_negative,
                generator=generator,
            )
        else:
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label,
                train_prop=train_prop,
                valid_prop=valid_prop,
                ignore_negative=ignore_negative,
                generator=generator,
            )
        split = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = cache_path.with_suffix(cache_path.suffix + '.tmp')
            torch.save(split, temp_path)
            temp_path.replace(cache_path)
        return split

    def __getitem__(self, idx):
        return (self.graph, self.label)

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def _mark_symmetric_edge_index(graph, data_obj=None, *, assume=False, max_check_edges=5000000):
    """Annotate COO edge_index when it is already symmetric and coalesced.

    The annotation lets SCAFFOLD skip global ``torch.unique``/coalesce passes.
    For large graphs we only set it for datasets whose storage format is known.
    For small PyG datasets we let PyG verify undirectedness.
    """
    if graph.get('edge_index_is_undirected_unique') or graph.get('edge_index_is_symmetric_unique'):
        return graph
    if assume:
        graph['edge_index_is_symmetric_unique'] = True
        return graph
    edge_index = graph.get('edge_index')
    if not torch.is_tensor(edge_index) or int(edge_index.size(1)) > int(max_check_edges):
        return graph
    if data_obj is None or not hasattr(data_obj, 'is_undirected'):
        return graph
    try:
        if bool(data_obj.is_undirected()):
            graph['edge_index_is_symmetric_unique'] = True
    except Exception:
        pass
    return graph


def _graph_from_pyg_data(data_obj, *, assume_symmetric=False):
    graph = {
        'edge_index': data_obj.edge_index,
        'node_feat': data_obj.x,
        'edge_feat': None,
        'num_nodes': data_obj.num_nodes,
    }
    return _mark_symmetric_edge_index(graph, data_obj, assume=assume_symmetric)

def canonicalize_dataset_name(dataname):
    dataname_lower = dataname.lower()
    ogb_aliases = {
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
        'karateclub': 'karate',
        'karate-balanced': 'karate-balance',
        'karate_balance': 'karate-balance',
    }
    return ogb_aliases.get(dataname_lower, dataname_lower)

def _load_legacy_dataset(data_dir, dataname, sub_dataname=''):
    data_dir = str(Path(data_dir).expanduser())
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    dataname_lower = canonicalize_dataset_name(dataname)
    if dataname_lower in ('amazon-photo', 'amazon-computer', 'amazon-computers'):
        name = 'amazon-computer' if 'computer' in dataname_lower else 'amazon-photo'
        dataset = load_amazon_dataset(data_dir, name)
    elif dataname_lower in ('coauthor-cs', 'coauthor-physics'):
        dataset = load_coauthor_dataset(data_dir, dataname_lower)
    elif dataname_lower in ('roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions'):
        dataset = load_hetero_dataset(data_dir, dataname_lower)
    elif dataname_lower == 'wikics':
        dataset = load_wikics_dataset(data_dir)
    elif dataname_lower in ('ogbn-arxiv', 'ogbn-products', 'ogbn-proteins'):
        dataset = load_ogb_dataset(data_dir, dataname_lower)
    elif dataname_lower in ('pokec', 'pokec-regions', 'graphland-pokec'):
        dataset = load_pokec_dataset(data_dir)
    elif dataname_lower in ('cora', 'citeseer', 'pubmed'):
        dataset = load_planetoid_dataset(data_dir, dataname_lower)
    elif dataname_lower == 'karate':
        dataset = load_karate_dataset()
    elif dataname_lower == 'karate-balance':
        dataset = load_karate_balanced_dataset()
    elif dataname_lower in ('chameleon', 'squirrel'):
        dataset = load_wiki_new(data_dir, dataname_lower)
    elif dataname_lower in ('reed98', 'amherst41', 'penn94', 'johnshopkins55', 'cornell5', 'amherest41', 'johnhopkings55'):
        if dataname_lower == 'amherest41':
            dataname_lower = 'amherst41'
        if dataname_lower == 'johnhopkings55':
            dataname_lower = 'johnshopkins55'
        dataset = load_facebook100_dataset(data_dir, dataname_lower)
    elif dataname_lower in ('cornell', 'texas', 'wisconsin'):
        dataset = load_webkb_dataset(data_dir, dataname_lower)
    elif dataname_lower == 'reddit':
        dataset = load_reddit_dataset(data_dir)
    elif dataname_lower == 'reddit2':
        dataset = load_reddit2_dataset(data_dir)
    elif dataname_lower in ('cora-full', 'corafull'):
        dataset = load_cora_full_dataset(data_dir)
    elif dataname_lower.startswith(('synth-', 'synthetic-')):
        dataset = load_synthetic_dataset(dataname_lower)
    else:
        raise ValueError(f'Invalid dataname: {dataname}')
    dataset.data_dir = data_dir
    return dataset

def load_planetoid_dataset(data_dir, name, no_feat_norm=True):
    if not no_feat_norm:
        transform = T.NormalizeFeatures()
        data_obj = Planetoid(root=f'{data_dir}/Planetoid', name=name, transform=transform)[0]
    else:
        data_obj = Planetoid(root=f'{data_dir}/Planetoid', name=name)[0]
    dataset = NCDataset(name)
    dataset.train_idx = torch.where(data_obj.train_mask)[0]
    dataset.valid_idx = torch.where(data_obj.val_mask)[0]
    dataset.test_idx = torch.where(data_obj.test_mask)[0]
    dataset.graph = _graph_from_pyg_data(data_obj)
    dataset.label = data_obj.y
    return dataset

def load_synthetic_dataset(dataname, marker_scale=None, noise_dim=0):
    """A named synthetic family instance as a node-classification dataset.

    ``synth-grid16-s3`` is the 16x16 lattice, task instance 3. See
    ``utils/synthetic_graphs.py`` for the registry, for why these families
    exist (they are route-poor, so congestion can bind where it cannot on the
    real benchmarks), and for the two constraints the task imposes: the model
    must be at least ``required_depth`` layers deep, and anchor nodes carry
    label ``-1`` so they are excluded from every split.
    """
    from .synthetic_graphs import (
        anchor_voronoi_task, build_graph, parse_spec, required_depth,
    )

    # A unit marker is not learnable at these depths: the anchor signal is
    # divided by the degree product along every hop, so by hop 7 a one-hot of
    # amplitude 1 is below the noise floor of the initial linear layer and the
    # *full* graph trains to chance. 10.0 is the amplitude the bottleneck task
    # needed for the same reason. Override with SCAFFOLD_SYNTH_MARKER_SCALE.
    if marker_scale is None:
        marker_scale = float(os.environ.get('SCAFFOLD_SYNTH_MARKER_SCALE', '10.0'))

    family, param, seed = parse_spec(dataname)
    G = build_graph(dataname)
    x, y, anchors = anchor_voronoi_task(
        G, family.anchors, seed=seed,
        marker_scale=marker_scale, noise_dim=noise_dim,
    )
    nodes = sorted(G.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    pairs = [(index[u], index[v]) for u, v in G.edges()]
    if pairs:
        undirected = torch.tensor(pairs, dtype=torch.long).t()
        edge_index = torch.cat([undirected, undirected.flip(0)], dim=1)
        order = torch.argsort(edge_index[0] * len(nodes) + edge_index[1])
        edge_index = edge_index[:, order].contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    dataset = NCDataset(dataname)
    dataset.graph = {
        'edge_index': edge_index,
        'node_feat': torch.from_numpy(x),
        'edge_feat': None,
        'num_nodes': len(nodes),
    }
    # Built here as a sorted, deduplicated, both-directions COO, so SCAFFOLD may
    # skip its global unique/coalesce pass.
    _mark_symmetric_edge_index(dataset.graph, assume=True)
    dataset.label = torch.from_numpy(y)
    dataset.synthetic = {
        'family': family.name, family.param: param, 'seed': seed,
        'n_classes': int(len(anchors)), 'anchors': list(anchors),
        'required_depth': required_depth(G, anchors),
    }
    return dataset


def load_karate_dataset():
    data_obj = KarateClub()[0]
    dataset = NCDataset('karate')
    dataset.graph = _graph_from_pyg_data(data_obj)
    dataset.label = data_obj.y
    return dataset


def load_karate_balanced_dataset():
    data_obj = KarateClub()[0]
    dataset = NCDataset('karate-balance')
    dataset.split_strategy = 'balanced_train'
    dataset.train_nodes_per_class = 2
    dataset.balanced_valid_num = 8
    dataset.graph = _graph_from_pyg_data(data_obj)
    dataset.label = data_obj.y
    return dataset


def load_wiki_new(data_dir, name):
    data = np.load(f'{data_dir}/geom-gcn/{name}/{name}_filtered.npz')
    node_feat = torch.as_tensor(data['node_features'])
    labels = torch.as_tensor(data['node_labels'])
    edges = torch.as_tensor(data['edges'], dtype=torch.long).T
    dataset = NCDataset(name)
    dataset.graph = {'edge_index': edges, 'node_feat': node_feat, 'edge_feat': None, 'num_nodes': node_feat.shape[0]}
    dataset.label = labels
    # Expose canonical geom-gcn fixed splits (multiple masks).
    from .data_utils import load_fixed_splits
    dataset.load_fixed_splits = lambda: load_fixed_splits(data_dir, dataset, name)
    return dataset

def load_wikics_dataset(data_dir):
    data_obj = WikiCS(root=f'{data_dir}/wikics/')[0]
    dataset = NCDataset('wikics')
    dataset.graph = _graph_from_pyg_data(data_obj)
    dataset.label = data_obj.y
    dataset.train_idx = torch.where(data_obj.train_mask[:, 0])[0]
    dataset.valid_idx = torch.where(data_obj.val_mask[:, 0])[0]
    dataset.test_idx = torch.where(data_obj.test_mask)[0]
    from .data_utils import load_fixed_splits
    dataset.load_fixed_splits = lambda: load_fixed_splits(data_dir, dataset, 'wikics')
    return dataset

def load_hetero_dataset(data_dir, name):
    # PyG expects one of:
    # roman-empire, amazon-ratings, minesweeper, tolokers, questions
    data_obj = HeterophilousGraphDataset(name=name, root=data_dir)[0]
    dataset = NCDataset(name)
    dataset.graph = _graph_from_pyg_data(data_obj)
    dataset.label = data_obj.y
    # Use fixed splits when provided by the dataset (first split column).
    if hasattr(data_obj, 'train_mask') and data_obj.train_mask is not None:
        train_mask = data_obj.train_mask
        if train_mask.dim() == 2:
            train_mask = train_mask[:, 0]
        dataset.train_idx = torch.where(train_mask)[0]
    if hasattr(data_obj, 'val_mask') and data_obj.val_mask is not None:
        val_mask = data_obj.val_mask
        if val_mask.dim() == 2:
            val_mask = val_mask[:, 0]
        dataset.valid_idx = torch.where(val_mask)[0]
    if hasattr(data_obj, 'test_mask') and data_obj.test_mask is not None:
        test_mask = data_obj.test_mask
        if test_mask.dim() == 2:
            test_mask = test_mask[:, 0]
        dataset.test_idx = torch.where(test_mask)[0]
    return dataset

def load_amazon_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    if name == 'amazon-photo':
        data_obj = Amazon(root=f'{data_dir}/Amazon', name='Photo', transform=transform)[0]
    else:
        data_obj = Amazon(root=f'{data_dir}/Amazon', name='Computers', transform=transform)[0]
    dataset = NCDataset(name)
    dataset.graph = _graph_from_pyg_data(data_obj)
    dataset.label = data_obj.y
    split_path = f'{data_dir}/{name}_split.npz'
    if path.exists(split_path):
        idx = np.load(split_path)
        dataset.train_idx = torch.from_numpy(idx['train']).long()
        dataset.valid_idx = torch.from_numpy(idx['valid']).long()
        dataset.test_idx = torch.from_numpy(idx['test']).long()
    return dataset

def load_coauthor_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    if name == 'coauthor-cs':
        data_obj = Coauthor(root=f'{data_dir}/Coauthor', name='CS', transform=transform)[0]
    else:
        data_obj = Coauthor(root=f'{data_dir}/Coauthor', name='Physics', transform=transform)[0]
    dataset = NCDataset(name)
    dataset.graph = _graph_from_pyg_data(data_obj)
    dataset.label = data_obj.y
    split_path = f'{data_dir}/{name}_split.npz'
    if path.exists(split_path):
        idx = np.load(split_path)
        dataset.train_idx = torch.from_numpy(idx['train']).long()
        dataset.valid_idx = torch.from_numpy(idx['valid']).long()
        dataset.test_idx = torch.from_numpy(idx['test']).long()
    return dataset

def load_ogb_dataset(data_dir, name):
    try:
        import ogb.nodeproppred.dataset as ogb_dataset_module
        from ogb.nodeproppred import NodePropPredDataset
    except ImportError as exc:
        raise ImportError(
            f"Dataset '{name}' requires the optional 'ogb' package. "
            "Install it with `pip install ogb`, or choose a non-OGB dataset."
        ) from exc

    dataset = NCDataset(name)
    auto_confirm = __import__('os').environ.get('SUPPORT_GRAPH_AUTO_CONFIRM_DOWNLOAD', '1')
    if str(auto_confirm).lower() not in ('0', 'false', 'no', 'off'):
        ogb_dataset_module.decide_download = lambda url: True
    ogb_root = Path(data_dir) / 'ogb' / name.replace('-', '_')
    processed_path = ogb_root / 'processed' / 'data_processed'
    if processed_path.exists() and processed_path.stat().st_size == 0:
        processed_path.unlink()

    skip_large_save = (
        name in ('ogbn-products', 'ogbn-proteins')
        and __import__('os').environ.get('SUPPORT_GRAPH_SKIP_OGB_LARGE_SAVE', '0').lower() not in ('0', 'false', 'no', 'off')
        and not processed_path.exists()
    )
    original_torch_load = torch.load
    original_torch_save = None
    def _torch_load_trusted_ogb_cache(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_torch_load(*args, **kwargs)

    torch.load = _torch_load_trusted_ogb_cache
    if skip_large_save:
        original_torch_save = torch.save
        torch.save = lambda *args, **kwargs: None
    try:
        ogb = NodePropPredDataset(name=name, root=f'{data_dir}/ogb')
    finally:
        torch.load = original_torch_load
        if original_torch_save is not None:
            torch.save = original_torch_save
    dataset.graph = ogb.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    if dataset.graph.get('node_feat') is None:
        edge_feat = dataset.graph.get('edge_feat')
        if edge_feat is None:
            raise ValueError(f'{name} does not provide node_feat or edge_feat.')
        from torch_geometric.utils import scatter
        edge_feat = torch.as_tensor(edge_feat).float()
        col = dataset.graph['edge_index'][1]
        dataset.graph['node_feat'] = scatter(edge_feat, col, dim=0, dim_size=int(dataset.graph['num_nodes']), reduce='sum')
        dataset.graph['edge_feat'] = edge_feat
    else:
        dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])
    if name == 'ogbn-products':
        _mark_symmetric_edge_index(dataset.graph, assume=True)
    if ogb.labels is None:
        dataset.label = None
    elif len(ogb.labels.shape) == 1:
        dataset.label = torch.as_tensor(ogb.labels).reshape(-1, 1)
    else:
        dataset.label = torch.as_tensor(ogb.labels)

    def idx_fn():
        s = ogb.get_idx_split()
        return {k: torch.as_tensor(s[k]) for k in s}
    dataset.load_fixed_splits = idx_fn
    return dataset

POKEC_MAT_DRIVE_IDS = tuple(
    dict.fromkeys(
        [
            dataset_drive_url.get('pokec', ''),
            '1575QYJwJlj7AWuOKMlwVmMz8FcslUncu',
        ]
    )
)
POKEC_SPLITS_DRIVE_ID = '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_'
POKEC_MAT_URLS = (
    'https://raw.githubusercontent.com/CUAI/Non-Homophily-Large-Scale/master/data/pokec.mat',
    'https://raw.githubusercontent.com/CUAI/Non-Homophily-Large-Scale/main/data/pokec.mat',
)
POKEC_SPLIT_URLS = (
    'https://raw.githubusercontent.com/CUAI/Non-Homophily-Large-Scale/master/data/splits/pokec-splits.npy',
    f'https://drive.usercontent.google.com/download?id={POKEC_SPLITS_DRIVE_ID}&export=download&confirm=t',
)
GRAPHLAND_POKEC_NAME = 'pokec-regions'
GRAPHLAND_POKEC_URL = f'https://zenodo.org/records/16895532/files/{GRAPHLAND_POKEC_NAME}.zip'
GRAPHLAND_POKEC_SPLIT = 'RH'

def load_pokec_dataset(data_dir):
    source = os.environ.get('SUPPORT_GRAPH_POKEC_SOURCE', 'auto').strip().lower()
    if source in ('mat', 'legacy', 'legacy-mat'):
        return load_pokec_mat(data_dir)
    if source == 'auto' and _has_existing_pokec_mat(data_dir):
        return load_pokec_mat(data_dir)
    return load_graphland_pokec_dataset(data_dir)

def _valid_local_file(file_path, expected_prefixes=None):
    file_path = Path(file_path)
    if not file_path.exists() or file_path.stat().st_size == 0:
        return False
    with file_path.open('rb') as handle:
        head = handle.read(512)
    lowered = head.lstrip().lower()
    if lowered.startswith(b'<') or b'error 404' in lowered or b'not found' in lowered:
        return False
    if expected_prefixes and not any(head.startswith(prefix) for prefix in expected_prefixes):
        return False
    return True

def _remove_broken_file(file_path, expected_prefixes=None):
    file_path = Path(file_path)
    if file_path.exists() and not _valid_local_file(file_path, expected_prefixes):
        file_path.unlink()

def _local_pokec_candidates(data_dir, filename):
    data_root = Path(data_dir).expanduser()
    repo_root = Path(__file__).resolve().parents[1]
    env_path = os.environ.get('SUPPORT_GRAPH_POKEC_MAT') if filename == 'pokec.mat' else os.environ.get('SUPPORT_GRAPH_POKEC_SPLITS')
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            data_root / 'pokec' / filename,
            data_root / filename,
            data_root / 'Dataset' / 'LINKXdataset' / 'pokec' / filename,
            data_root / 'Dataset' / 'AGSGSAINTII' / 'pokec' / filename,
            data_root / 'Dataset' / 'AGSGSAINTCHEB' / 'pokec' / filename,
            data_root / 'Dataset' / 'AGSGSAINTloop' / 'pokec' / filename,
            data_root / 'Dataset' / 'GSAINT' / 'pokec' / filename,
            data_root / 'Dataset' / 'tmp' / 'pokec' / filename,
            data_root / 'scaffold_gnn' / 'data' / 'pokec' / filename,
            repo_root / 'data' / 'pokec' / filename,
        ]
    )
    seen = set()
    unique = []
    for candidate in candidates:
        resolved = Path(candidate)
        key = str(resolved)
        if key not in seen:
            unique.append(resolved)
            seen.add(key)
    return unique

def _link_or_copy_existing_file(target_path, candidates, expected_prefixes=None):
    target_path = Path(target_path)
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate == target_path:
            continue
        if not _valid_local_file(candidate, expected_prefixes):
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(candidate, target_path)
            action = 'linked'
        except OSError:
            shutil.copy2(candidate, target_path)
            action = 'copied'
        print(f'Using local Pokec file ({action}): {candidate} -> {target_path}')
        return True
    return False

def _download_with_gdown(file_id, target_path, expected_prefixes=None):
    if not file_id:
        return False
    try:
        import gdown
    except ImportError:
        print(f'gdown is not installed; skipping Google Drive download for id={file_id}')
        return False

    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + '.tmp')
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        result = gdown.download(id=file_id, output=str(tmp_path), quiet=False)
        if result and _valid_local_file(tmp_path, expected_prefixes):
            tmp_path.replace(target_path)
            return True
    except Exception as exc:
        print(f'Pokec gdown download failed for id={file_id}: {exc}')
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return False

def _download_url(url, target_path, expected_prefixes=None):
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + '.tmp')
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        print(f'Downloading Pokec file from {url}...')
        urllib.request.urlretrieve(url, tmp_path)
        if _valid_local_file(tmp_path, expected_prefixes):
            tmp_path.replace(target_path)
            return True
        print(f'Ignoring invalid Pokec download from {url}')
    except Exception as exc:
        print(f'Pokec URL download failed for {url}: {exc}')
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return False

def _ensure_pokec_mat(data_dir, pokec_mat_path):
    pokec_mat_path = Path(pokec_mat_path)
    mat_prefixes = (b'MATLAB', b'\x89HDF')
    _remove_broken_file(pokec_mat_path, mat_prefixes)
    if _valid_local_file(pokec_mat_path, mat_prefixes):
        return

    if _link_or_copy_existing_file(pokec_mat_path, _local_pokec_candidates(data_dir, 'pokec.mat'), mat_prefixes):
        return

    for file_id in POKEC_MAT_DRIVE_IDS:
        if _download_with_gdown(file_id, pokec_mat_path, mat_prefixes):
            return

    for url in POKEC_MAT_URLS:
        if _download_url(url, pokec_mat_path, mat_prefixes):
            return

    searched = '\n  '.join(str(p) for p in _local_pokec_candidates(data_dir, 'pokec.mat'))
    raise FileNotFoundError(
        'Could not obtain the Pokec benchmark graph file automatically.\n'
        f'Expected target: {pokec_mat_path}\n'
        'The configured public Google Drive id is not currently accessible, and the '
        'old GitHub fallback no longer contains pokec.mat.\n'
        'Place the standard LINKX/CUAI pokec.mat file at the target path above, or set '
        'SUPPORT_GRAPH_POKEC_MAT=/path/to/pokec.mat.\n'
        'Searched local candidates:\n'
        f'  {searched}'
    )

def _ensure_pokec_splits(data_dir, splits_path):
    splits_path = Path(splits_path)
    npy_prefixes = (b'\x93NUMPY',)
    _remove_broken_file(splits_path, npy_prefixes)
    if _valid_local_file(splits_path, npy_prefixes):
        return

    if _link_or_copy_existing_file(splits_path, _local_pokec_candidates(data_dir, 'pokec-splits.npy'), npy_prefixes):
        return

    if _download_with_gdown(POKEC_SPLITS_DRIVE_ID, splits_path, npy_prefixes):
        return

    for url in POKEC_SPLIT_URLS:
        if _download_url(url, splits_path, npy_prefixes):
            return

    print(f'Could not download Pokec splits; random splits will be used if needed. Expected path: {splits_path}')

def _has_existing_pokec_mat(data_dir):
    mat_prefixes = (b'MATLAB', b'\x89HDF')
    target = Path(data_dir) / 'pokec' / 'pokec.mat'
    if _valid_local_file(target, mat_prefixes):
        return True
    return any(_valid_local_file(p, mat_prefixes) for p in _local_pokec_candidates(data_dir, 'pokec.mat'))

def _download_graphland_pokec(raw_parent):
    raw_parent = Path(raw_parent)
    raw_data_dir = raw_parent / GRAPHLAND_POKEC_NAME
    required = ('info.yaml', 'features.csv', 'targets.csv', 'edgelist.csv', 'split_masks_RH.csv')
    if all((raw_data_dir / name).exists() for name in required):
        return raw_data_dir

    raw_parent.mkdir(parents=True, exist_ok=True)
    zip_path = raw_parent / f'{GRAPHLAND_POKEC_NAME}.zip'
    if not _valid_local_file(zip_path):
        tmp_path = zip_path.with_suffix('.zip.tmp')
        if tmp_path.exists():
            tmp_path.unlink()
        print(f'Downloading GraphLand Pokec from {GRAPHLAND_POKEC_URL}...', flush=True)
        urllib.request.urlretrieve(GRAPHLAND_POKEC_URL, tmp_path)
        tmp_path.replace(zip_path)
    print(f'Extracting GraphLand Pokec cache from {zip_path}...', flush=True)
    with zipfile.ZipFile(zip_path, 'r') as archive:
        archive.extractall(raw_parent)
    return raw_data_dir

def _graphland_transform(name):
    from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler

    transforms = {
        'standard_scaler': lambda: StandardScaler(copy=False),
        'min_max_scaler': lambda: MinMaxScaler(clip=False, copy=False),
        'quantile_transform_normal': lambda: QuantileTransformer(
            output_distribution='normal', subsample=None, random_state=0, copy=False,
        ),
        'quantile_transform_uniform': lambda: QuantileTransformer(
            output_distribution='uniform', subsample=None, random_state=0, copy=False,
        ),
    }
    return transforms[name]()

def _graphland_empty(rows, dtype):
    return np.empty((rows, 0), dtype=dtype)

def _graphland_frame_values(frame, columns, dtype):
    if len(columns) == 0:
        return _graphland_empty(frame.shape[0], dtype)
    return frame[columns].to_numpy(dtype=dtype, copy=True)

def _graphland_encode_features(raw_dir, info):
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    print('Encoding GraphLand Pokec features...', flush=True)
    features_df = pd.read_csv(raw_dir / 'features.csv', index_col=0)
    fraction_names = info.get('fraction_features_names', [])
    num_names = [name for name in info.get('numerical_features_names', []) if name not in fraction_names]
    cat_names = info.get('categorical_features_names', [])

    num_features = _graphland_frame_values(features_df, num_names, np.float32)
    if num_features.size > 0:
        num_features = SimpleImputer(missing_values=np.nan, strategy='most_frequent', copy=False).fit_transform(num_features)
        num_features = _graphland_transform('quantile_transform_normal').fit_transform(num_features).astype(np.float32, copy=False)

    frac_features = _graphland_frame_values(features_df, fraction_names, np.float32)
    if frac_features.size > 0:
        frac_features = SimpleImputer(missing_values=np.nan, strategy='most_frequent', copy=False).fit_transform(frac_features)

    cat_features = _graphland_frame_values(features_df, cat_names, np.int32)
    if cat_features.size > 0:
        try:
            encoder = OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore', dtype=np.float32)
        except TypeError:
            encoder = OneHotEncoder(drop='if_binary', sparse=False, handle_unknown='ignore', dtype=np.float32)
        cat_features = encoder.fit_transform(cat_features).astype(np.float32, copy=False)

    return torch.from_numpy(np.concatenate([num_features, frac_features, cat_features], axis=1)).float()

def _load_graphland_pokec_fallback(data_dir):
    import pandas as pd
    import yaml
    from torch_geometric.data import Data
    from torch_geometric.transforms import ToUndirected

    root = Path(data_dir) / 'GraphLand'
    raw_parent = root / GRAPHLAND_POKEC_NAME / 'raw'
    processed_dir = root / GRAPHLAND_POKEC_NAME / 'processed' / 'support_graph_rh_undirected'
    processed_path = processed_dir / 'data.pt'
    if processed_path.exists():
        return torch.load(processed_path, map_location='cpu', weights_only=False)

    print('Processing GraphLand Pokec raw files...', flush=True)
    raw_dir = _download_graphland_pokec(raw_parent)
    with (raw_dir / 'info.yaml').open() as handle:
        info = yaml.safe_load(handle)

    x = _graphland_encode_features(raw_dir, info)
    print('Loading GraphLand Pokec targets and split masks...', flush=True)
    targets = pd.read_csv(raw_dir / 'targets.csv', index_col=0)[info['target_name']].to_numpy(dtype=np.float32, copy=True)
    labeled_mask = np.isfinite(targets)
    y = np.full(targets.shape[0], -1, dtype=np.int64)
    y[labeled_mask] = targets[labeled_mask].astype(np.int64)

    masks_df = pd.read_csv(raw_dir / f'split_masks_{GRAPHLAND_POKEC_SPLIT}.csv', index_col=0)
    train_mask = (masks_df['train'].to_numpy(dtype=bool) & labeled_mask)
    val_mask = (masks_df['val'].to_numpy(dtype=bool) & labeled_mask)
    test_mask = (masks_df['test'].to_numpy(dtype=bool) & labeled_mask)

    print('Loading GraphLand Pokec edgelist...', flush=True)
    edges = pd.read_csv(raw_dir / 'edgelist.csv').to_numpy(dtype=np.int64, copy=True)
    data = Data(
        edge_index=torch.from_numpy(edges.T).long(),
        x=x,
        y=torch.from_numpy(y).long(),
        train_mask=torch.from_numpy(train_mask).bool(),
        val_mask=torch.from_numpy(val_mask).bool(),
        test_mask=torch.from_numpy(test_mask).bool(),
        num_nodes=x.size(0),
    )
    data = ToUndirected()(data)
    processed_dir.mkdir(parents=True, exist_ok=True)
    print(f'Saving GraphLand Pokec processed cache to {processed_path}...', flush=True)
    torch.save(data, processed_path)
    return data

def load_graphland_pokec_dataset(data_dir):
    split = os.environ.get('SUPPORT_GRAPH_GRAPHLAND_SPLIT', GRAPHLAND_POKEC_SPLIT).strip().upper()
    root = Path(data_dir) / 'GraphLand'
    try:
        from torch_geometric.datasets import GraphLandDataset
        pyg_dataset = GraphLandDataset(
            root=str(root),
            name=GRAPHLAND_POKEC_NAME,
            split=split,
            to_undirected=True,
        )
        data_obj = pyg_dataset[0]
    except ImportError:
        if split != GRAPHLAND_POKEC_SPLIT:
            raise ValueError('Fallback GraphLand Pokec loader currently supports only split RH. Upgrade PyG for other splits.')
        data_obj = _load_graphland_pokec_fallback(data_dir)

    dataset = NCDataset('pokec')
    dataset.graph = _mark_symmetric_edge_index(
        {
            'edge_index': data_obj.edge_index.long(),
            'edge_feat': None,
            'node_feat': data_obj.x.float(),
            'num_nodes': int(data_obj.num_nodes),
        },
        data_obj,
        assume=True,
    )
    dataset.label = data_obj.y.long().view(-1)
    dataset.train_idx = torch.where(data_obj.train_mask)[0]
    dataset.valid_idx = torch.where(data_obj.val_mask)[0]
    dataset.test_idx = torch.where(data_obj.test_mask)[0]
    print(
        f'Loaded GraphLand Pokec ({GRAPHLAND_POKEC_NAME}, split={split}): '
        f'nodes={dataset.graph["num_nodes"]}, edges={dataset.graph["edge_index"].size(1)}, '
        f'features={dataset.graph["node_feat"].size(1)}'
    )
    return dataset

def load_pokec_mat(data_dir):
    pokec_dir = Path(data_dir) / 'pokec'
    pokec_dir.mkdir(parents=True, exist_ok=True)
    pokec_mat_path = pokec_dir / 'pokec.mat'
    _ensure_pokec_mat(data_dir, pokec_mat_path)

    fulldata = scipy.io.loadmat(pokec_mat_path)
    dataset = NCDataset('pokec')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index, 'edge_feat': None, 'node_feat': node_feat, 'num_nodes': num_nodes}
    label = fulldata['label'].flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)
    splits_path = pokec_dir / 'pokec-splits.npy'
    _ensure_pokec_splits(data_dir, splits_path)
    if path.exists(splits_path):
        split = np.load(splits_path, allow_pickle=True)
        if len(split) > 0:
            s0 = split[0].item() if hasattr(split[0], 'item') else split[0]
            dataset.train_idx = torch.as_tensor(np.asarray(s0['train'])).long()
            dataset.valid_idx = torch.as_tensor(np.asarray(s0['valid'])).long()
            dataset.test_idx = torch.as_tensor(np.asarray(s0['test'])).long()
    return dataset

def load_facebook100_dataset(data_dir, name):
    file_path = f'{data_dir}/facebook100/{name}.mat'
    if not path.exists(file_path):
        import os
        if not path.exists(f'{data_dir}/facebook100'):
            os.makedirs(f'{data_dir}/facebook100')
        url = f'https://github.com/sisaman/Facebook100/raw/master/data/{name}.mat'
        print(f'Downloading {name} to {file_path}...')
        try:
            import requests
            r = requests.get(url, allow_redirects=True)
            if r.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(r.content)
            else:
                print(f'Failed to download {name} automatically. Please download manually to {file_path}')
        except Exception as e:
            print(f'Download failed: {e}')
    if not path.exists(file_path):
        raise FileNotFoundError(f'Could not find {file_path}. Please download Facebook100 .mat files.')
    fulldata = scipy.io.loadmat(file_path)
    dist = fulldata['A']
    node_info = fulldata['local_info']
    label = torch.tensor(node_info[:, 5], dtype=torch.long)
    feat_tensors = []
    for col in [0, 1, 2, 3, 4, 6]:
        feat = node_info[:, col]
        vals = np.unique(feat)
        val_map = {v: i for i, v in enumerate(vals)}
        mapped = np.array([val_map[v] for v in feat], dtype=np.int64)
        mapped_f = torch.tensor(mapped, dtype=torch.long)
        feat_tensors.append(F.one_hot(mapped_f, num_classes=len(vals)))
    node_feat = torch.cat(feat_tensors, dim=1).float()
    coo = dist.tocoo()
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    edge_index = i
    dataset = NCDataset(name)
    dataset.graph = _mark_symmetric_edge_index(
        {'edge_index': edge_index, 'edge_feat': None, 'node_feat': node_feat, 'num_nodes': node_info.shape[0]},
        assume=(dist != dist.T).nnz == 0,
    )
    dataset.label = label
    return dataset

def load_webkb_dataset(data_dir, name):
    from torch_geometric.datasets import WebKB
    real_name = 'Cornell'
    if 'texas' in name.lower():
        real_name = 'Texas'
    if 'wisconsin' in name.lower():
        real_name = 'Wisconsin'
    data_obj = WebKB(root=f'{data_dir}/WebKB', name=real_name)[0]
    dataset = NCDataset(name)
    dataset.graph = _graph_from_pyg_data(data_obj)
    dataset.label = data_obj.y
    return dataset

def load_reddit_dataset(data_dir):
    reddit_root = Path(data_dir) / 'Reddit'
    for candidate in (
        reddit_root,
        Path(data_dir) / 'scaffold_gnn' / 'data' / 'Reddit',
        Path(data_dir) / 'Dataset' / 'tmp' / 'Reddit',
        Path(data_dir) / 'Dataset' / 'Reddit',
    ):
        candidate_raw = candidate / 'raw'
        if (candidate_raw / 'reddit_data.npz').exists() and (candidate_raw / 'reddit_graph.npz').exists():
            reddit_root = candidate
            break
    raw_dir = reddit_root / 'raw'
    data_path = raw_dir / 'reddit_data.npz'
    graph_path = raw_dir / 'reddit_graph.npz'
    if not data_path.exists() or not graph_path.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        zip_path = raw_dir / 'reddit.zip'
        if not zip_path.exists():
            urllib.request.urlretrieve('https://data.dgl.ai/dataset/reddit.zip', zip_path)
        with zipfile.ZipFile(zip_path, 'r') as archive:
            archive.extractall(raw_dir)

    data = np.load(data_path)
    node_feat = torch.from_numpy(data['feature']).float()
    label = torch.from_numpy(data['label']).long()
    split = torch.from_numpy(data['node_types']).long()

    adj = sp.load_npz(graph_path).tocoo()
    row = adj.row
    col = adj.col
    mask = row < col
    row = np.asarray(row[mask], dtype=np.int64)
    col = np.asarray(col[mask], dtype=np.int64)
    edge_index = torch.empty((2, row.shape[0]), dtype=torch.long)
    edge_index[0] = torch.from_numpy(row)
    edge_index[1] = torch.from_numpy(col)

    dataset = NCDataset('reddit')
    dataset.graph = {
        'edge_index': edge_index,
        'node_feat': node_feat,
        'edge_feat': None,
        'num_nodes': node_feat.shape[0],
        'edge_index_is_undirected_unique': True,
    }
    dataset.label = label
    dataset.train_idx = torch.where(split == 1)[0]
    dataset.valid_idx = torch.where(split == 2)[0]
    dataset.test_idx = torch.where(split == 3)[0]
    return dataset

def load_reddit2_dataset(data_dir):
    from torch_geometric.datasets import Reddit2
    data_obj = Reddit2(root=f'{data_dir}/Reddit2')[0]
    dataset = NCDataset('reddit2')
    dataset.graph = _graph_from_pyg_data(data_obj, assume_symmetric=True)
    dataset.label = data_obj.y
    dataset.train_idx = torch.where(data_obj.train_mask)[0]
    dataset.valid_idx = torch.where(data_obj.val_mask)[0]
    dataset.test_idx = torch.where(data_obj.test_mask)[0]
    return dataset

def load_cora_full_dataset(data_dir):
    from torch_geometric.datasets import CoraFull
    data_obj = CoraFull(root=f'{data_dir}/CoraFull')[0]
    dataset = NCDataset('cora-full')
    dataset.graph = _graph_from_pyg_data(data_obj)
    dataset.label = data_obj.y
    return dataset


def load_dataset(data_dir, dataname, sub_dataname=''):
    """Use exactly the same graph and split definitions as the baseline runners."""
    from scaffold_gnn.data.datasets import load_dataset as shared_load
    seed = int(os.environ.get('SCAFFOLD_DATASET_SEED', '42'))
    bundle = shared_load(data_dir, dataname, seed=seed, split_protocol='tunedgnn')
    dataset = NCDataset(bundle.name)
    dataset.data_dir = str(data_dir)
    dataset.graph = _graph_from_pyg_data(bundle.data)
    dataset.graph['edge_feat'] = getattr(bundle.data, 'edge_attr', None)
    if getattr(bundle.data, 'edge_weight', None) is not None:
        dataset.graph['edge_weight'] = bundle.data.edge_weight
    dataset.label = bundle.data.y
    dataset.num_classes = bundle.num_classes
    dataset.shared_splits = bundle.splits
    dataset.load_fixed_splits = lambda: bundle.splits
    return dataset
