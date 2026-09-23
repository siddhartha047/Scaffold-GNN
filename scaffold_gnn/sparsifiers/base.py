from abc import ABC, abstractmethod
from contextlib import contextmanager
import os
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx


_NETWORKIT_THREADS = None


@contextmanager
def deterministic_nk_rng(nk, seed):
    """Run a randomised NetworKit kernel so the same seed gives the same result.

    ``nk.setSeed(seed, useThreadId=False)`` points every OpenMP thread at one
    shared RNG, which is a data race: the draw an edge receives depends on
    thread scheduling, so a randomised scorer returns different scores on every
    call at the same seed. Measured on the planted partition (n=200, m=1245):
    ``RandomEdgeSparsifier().scores()`` is bit-identical over four calls at one
    thread and differs on every call at eight, and the supports that came out of
    it ranged over c=11..14 components at a fixed seed and a fixed
    ``PYTHONHASHSEED``. Passing ``useThreadId=True`` instead does not fix it --
    which thread draws for which edge is still scheduler-dependent.

    So the kernel runs single-threaded and the prior thread count is restored
    afterwards. These scorers are O(m) and are not the bottleneck in any sweep
    here, so the seed contract is worth more than their parallelism.

    Not used by ``GreedySpanner``: it draws with ``nk.graphtools.randomEdge`` in
    a serial Python loop, so there is only ever one consumer of the RNG and it
    measures bit-stable already. Pinning it would serialise its Dijkstra calls,
    and it is already the slowest method in the suite.
    """
    prior = nk.getMaxNumberOfThreads()
    nk.setNumberOfThreads(1)
    try:
        if seed is not None:
            nk.setSeed(seed, False)
        yield
    finally:
        nk.setNumberOfThreads(prior)


def _configure_networkit_threads(nk):
    """Apply the batch CPU allocation to parallel NetworKit kernels once."""

    global _NETWORKIT_THREADS
    raw = os.environ.get("SPARSIFIER_CPU_WORKERS", "").strip()
    if not raw:
        return
    try:
        requested = max(1, int(raw))
    except ValueError:
        return
    if _NETWORKIT_THREADS == requested:
        return
    nk.setNumberOfThreads(requested)
    _NETWORKIT_THREADS = requested
    print(f"[SparsifierParallel] networkit_threads={requested}", flush=True)


class BaseSparsifier(ABC):
    @abstractmethod
    def sparsify(self, data_or_graph):
        raise NotImplementedError('Subclasses must implement sparsify method')

    def _ensure_nx(self, data_or_graph):
        if isinstance(data_or_graph, Data):
            from scaffold_gnn.utils import pyg_to_nx
            return pyg_to_nx(data_or_graph)
        elif isinstance(data_or_graph, nx.Graph):
            return data_or_graph
        else:
            raise TypeError(f'Unsupported type: {type(data_or_graph)}')

    def _ensure_pyg(self, data_or_graph, original_data=None):
        if isinstance(data_or_graph, Data):
            return data_or_graph
        elif isinstance(data_or_graph, nx.Graph):
            from scaffold_gnn.utils import nx_to_pyg
            if original_data is not None:
                data = nx_to_pyg(data_or_graph, num_nodes=original_data.num_nodes)
                data.x = original_data.x
                data.y = original_data.y
                if hasattr(original_data, 'train_mask'):
                    data.train_mask = original_data.train_mask
                if hasattr(original_data, 'val_mask'):
                    data.val_mask = original_data.val_mask
                if hasattr(original_data, 'test_mask'):
                    data.test_mask = original_data.test_mask
                return data
            return nx_to_pyg(data_or_graph)
        else:
            raise TypeError(f'Unsupported type: {type(data_or_graph)}')

    def _nx_to_nk(self, G_nx):
        import networkit as nk
        _configure_networkit_threads(nk)
        #node_list = sorted(G_nx.nodes(), key=lambda x: int(x) if isinstance(x, int) else repr(x))
        # node_list = sorted(G_nx.nodes(), key=lambda x: repr(x))
        node_list = list(G_nx.nodes())
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        is_weighted = nx.is_weighted(G_nx)
        is_directed = G_nx.is_directed()
        G_nk = nk.Graph(len(node_list), weighted=is_weighted, directed=is_directed)
        for u, v, d in G_nx.edges(data=True):
            if is_weighted:
                G_nk.addEdge(node_to_idx[u], node_to_idx[v],
                             w=float(d.get('weight', 1.0)))
            else:
                G_nk.addEdge(node_to_idx[u], node_to_idx[v])
        G_nk.indexEdges()
        return G_nk, node_list

    def _pyg_undirected_pairs(self, data):
        edge_index = data.edge_index.detach().cpu().long()
        src = edge_index[0]
        dst = edge_index[1]
        non_loop = src != dst
        if bool(getattr(data, 'edge_index_is_undirected_unique', False)):
            pairs = edge_index[:, non_loop]
        elif bool(getattr(data, 'edge_index_is_symmetric_unique', False)):
            pairs = edge_index[:, non_loop & (src < dst)]
        else:
            lo = torch.minimum(src[non_loop], dst[non_loop])
            hi = torch.maximum(src[non_loop], dst[non_loop])
            pairs = torch.unique(torch.stack((lo, hi), dim=0), dim=1)
        return pairs.contiguous()

    def _ensure_nk(self, data_or_graph):
        """Build/load NetworKit directly from PyG, avoiding a large NetworkX copy."""

        import networkit as nk
        _configure_networkit_threads(nk)

        if not isinstance(data_or_graph, Data):
            return self._nx_to_nk(self._ensure_nx(data_or_graph))

        cache_path_value = getattr(self, 'networkit_cache_path', None)
        cache_path = Path(cache_path_value) if cache_path_value else None
        if cache_path is not None and cache_path.is_file():
            try:
                graph = nk.graphio.readGraph(
                    str(cache_path), nk.graphio.Format.NetworkitBinary
                )
                if graph.numberOfNodes() == int(data_or_graph.num_nodes):
                    # Binary caches can preserve sparse/stale edge IDs after a
                    # prior sparsifier has materialized a derived graph. Force
                    # a compact 0..m-1 index before score arrays use edgeId().
                    graph.indexEdges(force=True)
                    print(f'[NetworkitCache] hit path={cache_path}', flush=True)
                    return graph, list(range(int(data_or_graph.num_nodes)))
            except Exception as exc:
                print(
                    f'[NetworkitCache] ignoring unreadable cache path={cache_path}: {exc}',
                    flush=True,
                )

        pairs = self._pyg_undirected_pairs(data_or_graph)
        src = pairs[0].numpy()
        dst = pairs[1].numpy()
        graph = nk.graph.GraphFromCoo(
            (src, dst),
            n=int(data_or_graph.num_nodes),
            weighted=False,
            directed=False,
            edgesIndexed=True,
        )
        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.parent / f'.{cache_path.name}.{os.getpid()}.tmp'
                nk.graphio.writeGraph(
                    graph, str(temporary), nk.graphio.Format.NetworkitBinary
                )
                os.replace(temporary, cache_path)
                print(f'[NetworkitCache] saved path={cache_path}', flush=True)
            except Exception as exc:
                print(
                    f'[NetworkitCache] could not save path={cache_path}: {exc}',
                    flush=True,
                )
        return graph, list(range(int(data_or_graph.num_nodes)))

    def _nk_to_nx(self, G_nk, node_list):
        G_nx = nx.DiGraph() if G_nk.isDirected() else nx.Graph()
        G_nx.add_nodes_from(node_list)
        for u, v in G_nk.iterEdges():
            if G_nk.isWeighted():
                G_nx.add_edge(node_list[u], node_list[v], weight=G_nk.weight(u, v))
            else:
                G_nx.add_edge(node_list[u], node_list[v])
        return G_nx

    def _nk_to_pyg(self, G_nk, original_data):
        edge_count = int(G_nk.numberOfEdges())
        src = np.fromiter(
            (u for u, _ in G_nk.iterEdges()), dtype=np.int64, count=edge_count
        )
        dst = np.fromiter(
            (v for _, v in G_nk.iterEdges()), dtype=np.int64, count=edge_count
        )
        pairs = torch.from_numpy(np.stack((src, dst), axis=0))
        edge_index = torch.cat((pairs, pairs.flip(0)), dim=1).contiguous()
        data = Data(
            x=original_data.x,
            edge_index=edge_index,
            y=original_data.y,
            num_nodes=original_data.num_nodes,
        )
        data.edge_index_is_symmetric_unique = True
        data.num_undirected_edges = edge_count
        return data

    def _sparsify_via_nk(self, data_or_graph, nk_sparsifier, target_ratio):
        target_ratio = max(1e-6, min(1.0, target_ratio))
        if isinstance(data_or_graph, Data):
            G_nk, node_list = self._ensure_nk(data_or_graph)
        else:
            G = self._ensure_nx(data_or_graph)
            G_nk, node_list = self._nx_to_nk(G)
        if G_nk.numberOfEdges() == 0:
            if isinstance(data_or_graph, Data):
                return self._nk_to_pyg(G_nk, data_or_graph)
            H_nx = nx.DiGraph() if G_nk.isDirected() else nx.Graph()
            H_nx.add_nodes_from(node_list)
            return H_nx
        H_nk = nk_sparsifier.getSparsifiedGraphOfSize(G_nk, target_ratio)
        if isinstance(data_or_graph, Data):
            return self._nk_to_pyg(H_nk, data_or_graph)
        H_nx = self._nk_to_nx(H_nk, node_list)
        return H_nx

    def _sparsify_via_nk_exact(self, data_or_graph, nk_sparsifier, target_ratio):
        """Guaranteed-exact edge count variant.

        Uses nk_sparsifier.scores() to rank all edges, then takes exactly
        round(target_ratio * m) of them by descending score.  This sidesteps
        networkit's internal cap on zero-score edges that causes
        getSparsifiedGraphOfSize to return fewer edges than requested at high
        target ratios (e.g. Jaccard at 0.9+).
        """
        target_ratio = max(1e-6, min(1.0, target_ratio))
        if isinstance(data_or_graph, Data):
            G_nk, node_list = self._ensure_nk(data_or_graph)
        else:
            G = self._ensure_nx(data_or_graph)
            G_nk, node_list = self._nx_to_nk(G)
        m = G_nk.numberOfEdges()
        if m == 0:
            if isinstance(data_or_graph, Data):
                return self._nk_to_pyg(G_nk, data_or_graph)
            H_nx = nx.DiGraph() if G_nk.isDirected() else nx.Graph()
            H_nx.add_nodes_from(node_list)
            return H_nx
        target_num = max(1, round(m * target_ratio))
        scores = np.asarray(nk_sparsifier.scores(G_nk), dtype=np.float64)
        if target_num >= m:
            selected_ids = np.arange(m, dtype=np.int64)
        else:
            selected_ids = np.argpartition(-scores, target_num - 1)[:target_num]
        selected = np.zeros(G_nk.upperEdgeIdBound(), dtype=bool)
        selected[selected_ids] = True
        selected_edges = np.fromiter(
            (
                (u, v)
                for u, v in G_nk.iterEdges()
                if selected[G_nk.edgeId(u, v)]
            ),
            dtype=np.dtype([('src', np.int64), ('dst', np.int64)]),
            count=target_num,
        )
        out_src = selected_edges['src']
        out_dst = selected_edges['dst']
        if isinstance(data_or_graph, Data):
            pairs = torch.from_numpy(np.stack((out_src, out_dst), axis=0))
            edge_index = torch.cat((pairs, pairs.flip(0)), dim=1).contiguous()
            data = Data(
                x=data_or_graph.x,
                edge_index=edge_index,
                y=data_or_graph.y,
                num_nodes=data_or_graph.num_nodes,
            )
            data.edge_index_is_symmetric_unique = True
            data.num_undirected_edges = int(target_num)
            return data
        H_nx = nx.DiGraph() if G_nk.isDirected() else nx.Graph()
        H_nx.add_nodes_from(node_list)
        for u, v in zip(out_src, out_dst):
            if G_nk.isWeighted():
                H_nx.add_edge(node_list[u], node_list[v], weight=G_nk.weight(u, v))
            else:
                H_nx.add_edge(node_list[u], node_list[v])
        return H_nx
