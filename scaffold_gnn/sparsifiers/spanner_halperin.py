import networkx as nx
from torch_geometric.data import Data
from .base import BaseSparsifier


class HalperinSpanner(BaseSparsifier):
    def __init__(self, stretch=3, seed=None):
        self.stretch = int(stretch)
        self.seed = seed

    def sparsify(self, data_or_graph):
        G = self._ensure_nx(data_or_graph)
        if G.is_directed():
            raise ValueError('HalperinSpanner requires an undirected graph')
        weight_key = 'weight' if nx.is_weighted(G) else None
        H = nx.spanner(G, stretch=self.stretch, weight=weight_key, seed=self.seed)
        if isinstance(data_or_graph, Data):
            return self._ensure_pyg(H, original_data=data_or_graph)
        return H
