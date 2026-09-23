import math
from collections import defaultdict

import networkx as nx
from torch_geometric.data import Data

from .base import BaseSparsifier
from .scaffold.spanning_tree import resolve_edge_budget


class GreedyLowStretchTreeSparsifier(BaseSparsifier):
    """GREEDY-LOW-STRETCH-TREE(G, alpha, eta).

    Grows a spanning tree/forest T of G greedily by scoring the boundary
    edges B (edges whose endpoints lie in different components of T):

        score(e) = ((cut[e]+eps)/(C_max+eps))^eta
                 / ((projStretch[e]+eps)/(S_max+eps))^alpha

    where
        cut[e]          = number of graph edges running between the two
                          components that e connects,
        projStretch[e]  = mean over g=(x,y) in Cut(e) of
                          dist_{T u {e}}(x,y) / w(g).

    At every iteration the boundary edge with the highest score is added
    and its endpoints' components are merged.

    Exposes the same "sparsify(data_or_graph)" interface as the other
    sparsifiers, so it can be used wherever a spanning-tree builder is
    expected.  ``target_ratio`` and ``delta`` are accepted for API
    parity. When ``target_ratio`` is set, greedy construction stops at the
    corresponding edge budget, producing a partial low-stretch forest when
    the budget is below ``|V|-1``.
    """

    def __init__(
        self,
        alpha=1.0,
        eta=1.0,
        verbose=False,
        seed=None,
        # kept for API parity with other sparsifiers:
        delta=None,
        target_ratio=None,
    ):
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.verbose = verbose
        self.seed = seed
        self.delta = delta
        self.target_ratio = target_ratio
        self._eps = 1e-8

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------
    def sparsify(self, data_or_graph):
        G = self._ensure_nx(data_or_graph)
        max_edges = resolve_edge_budget(G.number_of_edges(), self.target_ratio)
        T = self._build_tree(G, max_edges=max_edges)
        if isinstance(data_or_graph, Data):
            return self._ensure_pyg(T, original_data=data_or_graph)
        return T

    # ------------------------------------------------------------------
    # core algorithm
    # ------------------------------------------------------------------
    def _edge_weight(self, G, u, v):
        data = G.get_edge_data(u, v)
        return float(data.get("weight", 1.0)) if data else 1.0

    def _build_tree(self, G, max_edges=None):
        is_weighted = nx.is_weighted(G)
        weight_key = "weight" if is_weighted else None
        eps = self._eps

        def _add(graph, u, v):
            if is_weighted:
                graph.add_edge(u, v, weight=self._edge_weight(G, u, v))
            else:
                graph.add_edge(u, v)

        T = nx.Graph()
        T.add_nodes_from(G.nodes())

        comp = {v: v for v in G.nodes()}
        members = {v: {v} for v in G.nodes()}

        n = G.number_of_nodes()

        requested = resolve_edge_budget(G.number_of_edges(), max_edges=max_edges)
        edge_limit = n - 1 if requested is None else min(n - 1, requested)

        while T.number_of_edges() < edge_limit:
            B = [(u, v) for (u, v) in G.edges() if comp[u] != comp[v]]
            if not B:
                break

            by_pair = defaultdict(list)
            for (u, v) in B:
                by_pair[tuple(sorted((comp[u], comp[v])))].append((u, v))

            cut_size = {}
            proj_stretch = {}

            for (u, v) in B:
                key = tuple(sorted((comp[u], comp[v])))
                cut = by_pair[key]
                cut_size[(u, v)] = len(cut)

                _add(T, u, v)
                try:
                    total = 0.0
                    for (x, y) in cut:
                        try:
                            d = nx.shortest_path_length(T, x, y, weight=weight_key)
                        except nx.NetworkXNoPath:
                            d = 0.0
                        total += d / max(self._edge_weight(G, x, y), eps)
                    proj_stretch[(u, v)] = total / (len(cut) + eps)
                finally:
                    T.remove_edge(u, v)

            C_max = max(cut_size.values()) if cut_size else eps
            S_max = max(proj_stretch.values()) if proj_stretch else eps

            score = {}
            for e in B:
                num = ((cut_size[e] + eps) / (C_max + eps)) ** self.eta
                den = ((proj_stretch[e] + eps) / (S_max + eps)) ** self.alpha
                score[e] = num / max(den, eps)

            best_edge = max(B, key=lambda e: score[e])
            u_star, v_star = best_edge
            T.add_edge(u_star, v_star, **(G.get_edge_data(u_star, v_star) or {}))

            Cu, Cv = comp[u_star], comp[v_star]
            if Cu != Cv:
                big, small = (Cu, Cv) if len(members[Cu]) >= len(members[Cv]) else (Cv, Cu)
                for node in members[small]:
                    comp[node] = big
                members[big].update(members[small])
                members[small] = set()

            if self.verbose:
                print(
                    f"[GLST] added edge={best_edge} "
                    f"cut={cut_size[best_edge]} "
                    f"projStretch={proj_stretch[best_edge]:.4f} "
                    f"edges={T.number_of_edges()}/{edge_limit}"
                )

        return T
