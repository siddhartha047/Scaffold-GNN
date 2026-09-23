import random

import networkx as nx
from torch_geometric.data import Data

from .base import BaseSparsifier
from .scaffold.spanning_tree import (
    build_random_spanning_forest_nx,
    resolve_edge_budget,
)


class RandomSpanningTreeSparsifier(BaseSparsifier):
    """Uniform random spanning tree/forest.

    For each connected component of ``G`` we generate a uniform random
    spanning tree via ``networkx.random_spanning_tree`` (Wilson's
    algorithm). The union across components is a spanning forest with
    ``|V| - cc(G)`` edges.

    When ``target_ratio`` is below a complete spanning forest, the fast
    random-priority Kruskal path stops at that edge budget instead of running
    Wilson's algorithm to completion and dropping edges afterward.
    """

    def __init__(self, seed=None, weight=None, target_ratio=None):
        self.seed = seed
        self.weight = weight
        self.target_ratio = target_ratio
        self._rng = random.Random(seed)

    def sparsify(self, data_or_graph):
        G = self._ensure_nx(data_or_graph)
        if G.is_directed():
            raise ValueError("RandomSpanningTreeSparsifier requires an undirected graph")

        H = nx.Graph()
        H.add_nodes_from(G.nodes(data=True))

        max_edges = resolve_edge_budget(G.number_of_edges(), self.target_ratio)
        if max_edges is not None:
            H = build_random_spanning_forest_nx(
                G,
                seed=self.seed,
                max_edges=max_edges,
            )
            if isinstance(data_or_graph, Data):
                return self._ensure_pyg(H, original_data=data_or_graph)
            return H

        for nodes in nx.connected_components(G):
            if len(nodes) <= 1:
                continue
            component = G.subgraph(nodes).copy()
            T = nx.random_spanning_tree(
                component,
                weight=self.weight,
                multiplicative=True,
                seed=self._rng.randint(0, 2**31 - 1),
            )
            for u, v, data in T.edges(data=True):
                H.add_edge(u, v, **(G.get_edge_data(u, v) or data))

        if isinstance(data_or_graph, Data):
            return self._ensure_pyg(H, original_data=data_or_graph)
        return H
