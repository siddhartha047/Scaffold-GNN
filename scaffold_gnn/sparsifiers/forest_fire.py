import networkit as nk
import networkx as nx
from torch_geometric.data import Data
from .base import BaseSparsifier, deterministic_nk_rng


class ForestFireSparsifier(BaseSparsifier):
    def __init__(self, p=0.35, target_ratio=None, seed=None):
        self.p = float(p)
        self.target_ratio = target_ratio
        self.seed = seed
        if not 0.0 <= self.p <= 1.0:
            raise ValueError('p must be in [0, 1]')

    def sparsify(self, data_or_graph):
        if self.target_ratio is not None:
            # Linear interpolation: low target_ratio → low p (selective fire),
            # high target_ratio → high p (aggressive fire, more edges scored).
            # Previously the binary mapping was backwards (0.2 for >0.5).
            p = 0.2 + 0.4 * float(self.target_ratio)
        else:
            p = self.p

        # The burning process is parallel and draws from a shared RNG, so the
        # seed alone does not pin the support -- see deterministic_nk_rng. Four
        # calls at one seed gave three distinct supports (c=26..31) before this.
        with deterministic_nk_rng(nk, self.seed):
            nk_sp = nk.sparsification.ForestFireSparsifier(p, 5.0)

            if self.target_ratio is not None:
                return self._sparsify_via_nk_exact(
                    data_or_graph, nk_sp, float(self.target_ratio)
                )

            if isinstance(data_or_graph, Data):
                G_nk, node_list = self._ensure_nk(data_or_graph)
            else:
                G = self._ensure_nx(data_or_graph)
                G_nk, node_list = self._nx_to_nk(G)

            # Guard: ForestFireSparsifier segfaults in C++ on empty graphs.
            if G_nk.numberOfEdges() == 0:
                if isinstance(data_or_graph, Data):
                    return self._nk_to_pyg(G_nk, data_or_graph)
                H_nx = nx.DiGraph() if G_nk.isDirected() else nx.Graph()
                H_nx.add_nodes_from(node_list)
                return H_nx

            H_nk = nk_sp.getSparsifiedGraph(G_nk)

        if isinstance(data_or_graph, Data):
            return self._nk_to_pyg(H_nk, data_or_graph)

        H_nx = nx.DiGraph() if G_nk.isDirected() else nx.Graph()
        H_nx.add_nodes_from(node_list)

        for u, v in H_nk.iterEdges():
            if H_nk.isWeighted():
                H_nx.add_edge(node_list[u], node_list[v], weight=H_nk.weight(u, v))
            else:
                H_nx.add_edge(node_list[u], node_list[v])

        return H_nx
