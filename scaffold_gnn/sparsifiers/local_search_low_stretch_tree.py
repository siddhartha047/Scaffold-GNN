import heapq
import math
import random

import networkx as nx
from torch_geometric.data import Data

from .base import BaseSparsifier
from .scaffold.spanning_tree import (
    FAST_TREE_DEFAULT_BUCKETS,
    build_fast_weighted_spanning_forest_nx,
    build_random_spanning_forest_nx,
    build_spanning_forest_nx,
    canonical_support_name,
    resolve_edge_budget,
)


class LocalSearchLowStretchTreeSparsifier(BaseSparsifier):
    """Low-stretch tree via local edge swaps with optional fast approximations.

    Algorithm:
    1. Build an initial tree with GLST.
    2. Repeatedly score non-tree candidate edges and choose either a
       random subset or the top current tree-distance edges.
    3. Adding a candidate creates a unique cycle. Evaluate sampled
       removable edges on that cycle.
    4. Keep the best improving swap by sampled or exact total stretch
       and repeat until no improving move remains or max_passes is
       reached.

    Stretch objective:
        total_stretch(T) = sum_{(u,v) in E(G)} dist_T(u,v) / w(u,v)
    """

    def __init__(
        self,
        max_passes=10,
        init_support="glst",
        fast_tree_buckets=FAST_TREE_DEFAULT_BUCKETS,
        glst_alpha=1.0,
        glst_eta=1.0,
        candidate_strategy="random",
        candidate_sample_size=0,
        eval_sample_size=0,
        cycle_sample_size=0,
        resample_eval_each_pass=False,
        verbose=False,
        seed=None,
        target_ratio=None,
    ):
        self.max_passes = max(1, int(max_passes))
        self.init_support = canonical_support_name(init_support)
        self.fast_tree_buckets = int(fast_tree_buckets)
        self.glst_alpha = float(glst_alpha)
        self.glst_eta = float(glst_eta)
        self.candidate_strategy = str(candidate_strategy)
        self.candidate_sample_size = max(0, int(candidate_sample_size))
        self.eval_sample_size = max(0, int(eval_sample_size))
        self.cycle_sample_size = max(0, int(cycle_sample_size))
        self.resample_eval_each_pass = bool(resample_eval_each_pass)
        self.verbose = verbose
        self.seed = seed
        self.target_ratio = target_ratio
        self._eps = 1e-12
        self._rng = random.Random(seed)

    def sparsify(self, data_or_graph):
        G = self._ensure_nx(data_or_graph)
        if G.is_directed():
            raise ValueError("LocalSearchLowStretchTreeSparsifier requires an undirected graph")

        H = nx.Graph()
        H.add_nodes_from(G.nodes(data=True))

        remaining = resolve_edge_budget(G.number_of_edges(), self.target_ratio)
        for nodes in nx.connected_components(G):
            if remaining is not None and remaining <= 0:
                break
            component = G.subgraph(nodes).copy()
            if component.number_of_nodes() <= 1:
                continue
            component_budget = None if remaining is None else min(
                remaining, max(0, component.number_of_nodes() - 1)
            )
            tree = self._build_component_tree(
                component,
                max_edges=component_budget,
            )
            H.add_edges_from(tree.edges(data=True))
            if remaining is not None:
                remaining -= tree.number_of_edges()

        if isinstance(data_or_graph, Data):
            return self._ensure_pyg(H, original_data=data_or_graph)
        return H

    def _build_component_tree(self, G, max_edges=None):
        full_tree_edges = max(0, G.number_of_nodes() - 1)
        edge_limit = full_tree_edges if max_edges is None else min(
            full_tree_edges, max(0, int(max_edges))
        )
        if edge_limit <= 0:
            T = nx.Graph()
            T.add_nodes_from(G.nodes(data=True))
            return T
        if G.number_of_edges() <= max(0, G.number_of_nodes() - 1) and nx.is_tree(G):
            if edge_limit >= G.number_of_edges():
                return G.copy()

        initial = self._build_initial_tree(G, max_edges=max_edges)

        # The LLST swap objective requires a connected tree. A lower target is
        # intentionally a partial forest, so return the budgeted initializer
        # directly instead of constructing a full tree and trimming it.
        if edge_limit < full_tree_edges:
            return initial

        best_tree = initial.copy()
        eval_edges = self._sample_eval_edges(G)
        best_stretch = self._total_stretch(G, best_tree, eval_edges=eval_edges)

        if self.verbose:
            sampled_eval_edges = G.number_of_edges() if eval_edges is None else len(eval_edges)
            print(
                f"[LLST] init total_stretch={best_stretch:.4f} "
                f"mean_stretch={best_stretch / max(G.number_of_edges(), 1):.4f} "
                f"sampled_eval_edges={sampled_eval_edges}"
            )

        passes = 0
        while passes < self.max_passes:
            if passes > 0 and self.resample_eval_each_pass:
                eval_edges = self._sample_eval_edges(G)
                best_stretch = self._total_stretch(G, best_tree, eval_edges=eval_edges)
            move = self._best_improving_swap(
                G,
                best_tree,
                best_stretch,
                eval_edges=eval_edges,
            )
            if move is None:
                break
            best_tree, best_stretch, added_edge, removed_edge = move
            passes += 1
            if self.verbose:
                print(
                    f"[LLST] pass={passes} add={added_edge} remove={removed_edge} "
                    f"total_stretch={best_stretch:.4f} "
                    f"mean_stretch={best_stretch / max(G.number_of_edges(), 1):.4f}"
                )

        return best_tree

    def _build_initial_tree(self, G, max_edges=None):
        if self.init_support == "glst":
            from .low_stretch_tree import GreedyLowStretchTreeSparsifier

            builder = GreedyLowStretchTreeSparsifier(
                alpha=self.glst_alpha,
                eta=self.glst_eta,
                verbose=False,
                seed=self.seed,
            )
            return builder._build_tree(G, max_edges=max_edges)
        if self.init_support == "maxst":
            weight_key = "weight" if nx.is_weighted(G) else None
            return build_spanning_forest_nx(
                G,
                weight_key=weight_key,
                maximum=True,
                max_edges=max_edges,
            )
        if self.init_support == "mst":
            weight_key = "weight" if nx.is_weighted(G) else None
            return build_spanning_forest_nx(
                G,
                weight_key=weight_key,
                maximum=False,
                max_edges=max_edges,
            )
        if self.init_support in ("fast_mst", "fast_maxst"):
            weight_key = "weight" if nx.is_weighted(G) else None
            return build_fast_weighted_spanning_forest_nx(
                G,
                weight_key=weight_key,
                maximum=self.init_support == "fast_maxst",
                bucket_count=self.fast_tree_buckets,
                max_edges=max_edges,
            )
        if self.init_support == "randst":
            return build_random_spanning_forest_nx(
                G,
                seed=self.seed,
                max_edges=max_edges,
            )
        if self.init_support == "randspt":
            from .random_shortest_path_tree import RandomShortestPathTreeSparsifier

            ratio = None
            if max_edges is not None:
                ratio = float(max_edges) / max(1, G.number_of_edges())
            return RandomShortestPathTreeSparsifier(
                seed=self.seed,
                target_ratio=ratio,
            ).sparsify(G)
        raise ValueError(f"Unknown LLST init_support: {self.init_support!r}")

    def _sample_edges(self, edges, sample_size):
        edge_list = list(edges)
        if sample_size <= 0 or len(edge_list) <= sample_size:
            return edge_list
        return self._rng.sample(edge_list, sample_size)

    def _sample_eval_edges(self, G):
        if self.eval_sample_size <= 0:
            return None
        eval_edges = [self._canon_edge(u, v) for u, v in G.edges()]
        return self._sample_edges(eval_edges, self.eval_sample_size)

    def _best_improving_swap(self, G, T, current_stretch, eval_edges=None):
        best_trial = None
        best_stretch = current_stretch
        tree_index = self._build_tree_index(T)
        tree_edge_set = {self._canon_edge(u, v) for u, v in T.edges()}
        candidate_edges = [
            self._canon_edge(u, v)
            for u, v in G.edges()
            if self._canon_edge(u, v) not in tree_edge_set
        ]

        for candidate_edge in self._choose_candidate_edges(candidate_edges, tree_index):
            u, v = candidate_edge
            path_edges = self._path_edges(u, v, tree_index)
            path_edges = self._sample_edges(path_edges, self.cycle_sample_size)

            edge_data = G.get_edge_data(u, v) or {}
            for remove_u, remove_v in path_edges:
                trial = T.copy()
                trial.add_edge(u, v, **edge_data)
                trial.remove_edge(remove_u, remove_v)
                if not nx.is_tree(trial):
                    continue

                trial_index = self._build_tree_index(trial)
                trial_stretch = self._total_stretch(
                    G,
                    trial,
                    eval_edges=eval_edges,
                    tree_index=trial_index,
                )
                if trial_stretch + self._eps < best_stretch:
                    best_trial = (
                        trial,
                        trial_stretch,
                        candidate_edge,
                        self._canon_edge(remove_u, remove_v),
                    )
                    best_stretch = trial_stretch

        return best_trial

    def _choose_candidate_edges(self, candidate_edges, tree_index):
        if self.candidate_strategy == "random":
            return self._sample_edges(candidate_edges, self.candidate_sample_size)

        if self.candidate_strategy != "tree_distance":
            raise ValueError(
                f"Unknown LLST candidate_strategy: {self.candidate_strategy!r}"
            )

        if self.candidate_sample_size <= 0 or self.candidate_sample_size >= len(candidate_edges):
            k = len(candidate_edges)
        else:
            k = self.candidate_sample_size

        scored = (
            (self._tree_distance(u, v, tree_index), (u, v))
            for u, v in candidate_edges
        )
        return [edge for _, edge in heapq.nlargest(k, scored, key=lambda item: item[0])]

    def _total_stretch(self, G, T, eval_edges=None, tree_index=None):
        tree_index = self._build_tree_index(T) if tree_index is None else tree_index
        total = 0.0
        sampled_edges = list(eval_edges) if eval_edges is not None else [
            self._canon_edge(u, v) for u, v in G.edges()
        ]
        if not sampled_edges:
            return 0.0
        for u, v in sampled_edges:
            data = G.get_edge_data(u, v) or {}
            edge_weight = float(data.get("weight", 1.0))
            edge_weight = max(edge_weight, self._eps)
            dist = self._tree_distance(u, v, tree_index)
            total += dist / edge_weight
        scale = 1.0 if eval_edges is None else float(G.number_of_edges()) / float(len(sampled_edges))
        return total * scale

    def _build_tree_index(self, T):
        nodes = list(T.nodes())
        if not nodes:
            return None

        index_of = {node: idx for idx, node in enumerate(nodes)}
        n = len(nodes)
        root = nodes[0]
        root_idx = index_of[root]

        parent = [-1] * n
        depth = [0] * n
        dist_to_root = [0.0] * n
        edge_to_parent = [None] * n
        seen = {root}
        stack = [root]

        while stack:
            u = stack.pop()
            iu = index_of[u]
            for v, data in T[u].items():
                if v in seen:
                    continue
                seen.add(v)
                iv = index_of[v]
                parent[iv] = iu
                depth[iv] = depth[iu] + 1
                dist_to_root[iv] = dist_to_root[iu] + float((data or {}).get("weight", 1.0))
                edge_to_parent[iv] = self._canon_edge(u, v)
                stack.append(v)

        log_n = max(1, n.bit_length())
        up = [[root_idx] * n for _ in range(log_n)]
        for idx in range(n):
            up[0][idx] = root_idx if parent[idx] < 0 else parent[idx]
        for level in range(1, log_n):
            prev = up[level - 1]
            curr = up[level]
            for idx in range(n):
                curr[idx] = prev[prev[idx]]

        return {
            "index_of": index_of,
            "parent": parent,
            "depth": depth,
            "dist_to_root": dist_to_root,
            "edge_to_parent": edge_to_parent,
            "up": up,
        }

    def _lca_index(self, idx_u, idx_v, tree_index):
        depth = tree_index["depth"]
        up = tree_index["up"]

        if depth[idx_u] < depth[idx_v]:
            idx_u, idx_v = idx_v, idx_u

        diff = depth[idx_u] - depth[idx_v]
        bit = 0
        while diff:
            if diff & 1:
                idx_u = up[bit][idx_u]
            diff >>= 1
            bit += 1

        if idx_u == idx_v:
            return idx_u

        for level in range(len(up) - 1, -1, -1):
            if up[level][idx_u] != up[level][idx_v]:
                idx_u = up[level][idx_u]
                idx_v = up[level][idx_v]
        return up[0][idx_u]

    def _tree_distance(self, u, v, tree_index):
        if u == v:
            return 0.0
        index_of = tree_index["index_of"]
        dist_to_root = tree_index["dist_to_root"]
        idx_u = index_of[u]
        idx_v = index_of[v]
        idx_lca = self._lca_index(idx_u, idx_v, tree_index)
        return (
            dist_to_root[idx_u]
            + dist_to_root[idx_v]
            - 2.0 * dist_to_root[idx_lca]
        )

    def _path_edges(self, u, v, tree_index):
        index_of = tree_index["index_of"]
        parent = tree_index["parent"]
        depth = tree_index["depth"]
        edge_to_parent = tree_index["edge_to_parent"]

        idx_u = index_of[u]
        idx_v = index_of[v]
        path_from_u = []
        path_from_v = []

        while depth[idx_u] > depth[idx_v]:
            path_from_u.append(edge_to_parent[idx_u])
            idx_u = parent[idx_u]
        while depth[idx_v] > depth[idx_u]:
            path_from_v.append(edge_to_parent[idx_v])
            idx_v = parent[idx_v]
        while idx_u != idx_v:
            path_from_u.append(edge_to_parent[idx_u])
            path_from_v.append(edge_to_parent[idx_v])
            idx_u = parent[idx_u]
            idx_v = parent[idx_v]

        path_from_v.reverse()
        return path_from_u + path_from_v

    def _canon_edge(self, u, v):
        return (u, v) if repr(u) <= repr(v) else (v, u)
