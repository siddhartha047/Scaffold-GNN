import networkx as nx
from torch_geometric.data import Data

from .base import BaseSparsifier
from .scaffold.spanning_tree import build_spanning_forest_nx, resolve_edge_budget


class MaxSTSparsifier(BaseSparsifier):
    """Maximum spanning tree via the fast native Kruskal union-find.

    Uses the same builder as ``MSTSparsifier`` but selects heavy edges first
    when the graph is weighted. For unweighted graphs every edge has weight 1
    and the result is any spanning tree/forest.

    ``target_ratio`` caps support construction early when a complete spanning
    forest would exceed the requested edge budget.
    """

    def __init__(self, algorithm="kruskal", weight="weight", target_ratio=None):
        # Kept for backward-compat with existing configs; the fast path always
        # uses Kruskal-with-union-find, so 'prim'/'boruvka' resolve to the same
        # implementation.
        self.algorithm = str(algorithm)
        self.weight = weight
        self.target_ratio = target_ratio

    def sparsify(self, data_or_graph):
        G = self._ensure_nx(data_or_graph)
        if G.is_directed():
            raise ValueError("MaxSTSparsifier requires an undirected graph")
        weight_key = self.weight if nx.is_weighted(G) else None
        max_edges = resolve_edge_budget(G.number_of_edges(), self.target_ratio)
        H = build_spanning_forest_nx(
            G,
            weight_key=weight_key,
            maximum=True,
            max_edges=max_edges,
        )
        if isinstance(data_or_graph, Data):
            return self._ensure_pyg(H, original_data=data_or_graph)
        return H
