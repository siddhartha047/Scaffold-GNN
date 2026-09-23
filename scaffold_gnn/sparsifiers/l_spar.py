import networkit as nk
import networkx as nx
from torch_geometric.data import Data
from .base import BaseSparsifier


class LSPAR(BaseSparsifier):
    def __init__(self, e=0.6, target_ratio=None):
        self.e = float(e)
        self.target_ratio = target_ratio
        if self.target_ratio is None and not 0.0 <= self.e <= 1.0:
            raise ValueError('e must be in [0, 1]')

    def sparsify(self, data_or_graph):
        if self.target_ratio is not None:
            ratio = max(1e-6, min(1.0, float(self.target_ratio)))
            nk_sp = nk.sparsification.JaccardSimilaritySparsifier()
            return self._sparsify_via_nk_exact(data_or_graph, nk_sp, ratio)

        G = self._ensure_nx(data_or_graph)
        G_nk, node_list = self._nx_to_nk(G)
        jac = nk.sparsification.JaccardSimilaritySparsifier()
        scores = jac.scores(G_nk)

        H_nx = nx.DiGraph() if G_nk.isDirected() else nx.Graph()
        H_nx.add_nodes_from(node_list)

        for idx in range(len(node_list)):
            neighbors = list(G_nk.iterNeighbors(idx))
            d = len(neighbors)
            if d == 0:
                continue

            k = max(1, int(d ** self.e))

            neighbors_sorted = sorted(neighbors,key=lambda nb: scores[G_nk.edgeId(idx, nb)],reverse=True,)

            for nb in neighbors_sorted[:k]:
                if not H_nx.has_edge(node_list[idx], node_list[nb]):
                    if G_nk.isWeighted():
                        w = G_nk.weight(idx, nb)
                        H_nx.add_edge(node_list[idx], node_list[nb], weight=w)
                    else:
                        H_nx.add_edge(node_list[idx], node_list[nb])

        if isinstance(data_or_graph, Data):
            return self._ensure_pyg(H_nx, original_data=data_or_graph)

        return H_nx

def l_spar(G, e):
    return LSPAR(e=e).sparsify(G)
