import heapq
import random
from collections import deque

import networkx as nx
from torch_geometric.data import Data

from .base import BaseSparsifier
from .scaffold.spanning_tree import resolve_edge_budget


class RandomShortestPathTreeSparsifier(BaseSparsifier):
    """Randomized shortest-path tree/forest.

    For each connected component:
    1. Pick a random root.
    2. Build a shortest-path tree from that root.
    3. Break ties randomly among equal-length shortest paths.

    On unweighted graphs this is a randomized BFS tree. On weighted
    graphs it is a randomized Dijkstra tree.
    """

    def __init__(self, seed=None, target_ratio=None):
        self.seed = seed
        self.target_ratio = target_ratio
        self._eps = 1e-12
        self._rng = random.Random(seed)

    def sparsify(self, data_or_graph):
        G = self._ensure_nx(data_or_graph)
        if G.is_directed():
            raise ValueError("RandomShortestPathTreeSparsifier requires an undirected graph")

        H = nx.Graph()
        H.add_nodes_from(G.nodes(data=True))

        is_weighted = nx.is_weighted(G)
        remaining = resolve_edge_budget(G.number_of_edges(), self.target_ratio)
        for nodes in nx.connected_components(G):
            if remaining is not None and remaining <= 0:
                break
            if len(nodes) <= 1:
                continue
            component = G.subgraph(nodes).copy()
            root = self._rng.choice(list(component.nodes()))
            component_budget = None if remaining is None else min(
                remaining, max(0, component.number_of_nodes() - 1)
            )
            if is_weighted:
                parent = self._random_dijkstra_tree(
                    component, root, max_edges=component_budget
                )
            else:
                parent = self._random_bfs_tree(
                    component, root, max_edges=component_budget
                )
            added = 0
            for node, pred in parent.items():
                if pred is None:
                    continue
                H.add_edge(node, pred, **(G.get_edge_data(node, pred) or {}))
                added += 1
            if remaining is not None:
                remaining -= added

        if isinstance(data_or_graph, Data):
            return self._ensure_pyg(H, original_data=data_or_graph)
        return H

    def _random_bfs_tree(self, G, root, max_edges=None):
        adj = {u: list(G.neighbors(u)) for u in G.nodes()}
        for nbrs in adj.values():
            self._rng.shuffle(nbrs)

        parent = {root: None}
        if max_edges is not None and max_edges <= 0:
            return parent
        queue = deque([root])
        added = 0
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v in parent:
                    continue
                parent[v] = u
                added += 1
                if max_edges is not None and added >= max_edges:
                    return parent
                queue.append(v)
        return parent

    def _random_dijkstra_tree(self, G, root, max_edges=None):
        adj = {u: list(G.neighbors(u)) for u in G.nodes()}
        tie_break = {}
        for u, nbrs in adj.items():
            self._rng.shuffle(nbrs)
            for v in nbrs:
                tie_break[(u, v)] = self._rng.random()

        dist = {root: 0.0}
        parent = {root: None}
        if max_edges is not None and max_edges <= 0:
            return parent
        best_tie = {root: -1.0}
        heap = [(0.0, -1.0, repr(root), root)]

        while heap:
            d, in_tie, _, u = heapq.heappop(heap)
            if d > dist.get(u, float("inf")) + self._eps:
                continue
            if in_tie > best_tie.get(u, float("inf")) + self._eps:
                continue

            for v in adj[u]:
                w = float(G[u][v].get("weight", 1.0))
                nd = d + max(w, self._eps)
                edge_tie = tie_break[(u, v)]
                cur = dist.get(v)
                cur_tie = best_tie.get(v, float("inf"))
                if cur is None or nd < cur - self._eps or (
                    abs(nd - cur) <= self._eps and edge_tie < cur_tie - self._eps
                ):
                    discovered = cur is None
                    dist[v] = nd
                    parent[v] = u
                    best_tie[v] = edge_tie
                    heapq.heappush(heap, (nd, edge_tie, repr(v), v))
                    if (
                        discovered
                        and max_edges is not None
                        and max(0, len(parent) - 1) >= max_edges
                    ):
                        return parent

        return parent
