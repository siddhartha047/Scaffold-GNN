import math
import random
import subprocess
import torch
import networkx as nx
from torch_geometric.data import Data

from .base import BaseSparsifier


class SupportGraphDilationSparsifier(BaseSparsifier):
    """Support-graph sparsifier that adds only the maximum-dilation edge.

    This follows the algorithm:
    1. Start from a spanning tree/forest.
    2. Repeatedly compute dilation for each remaining edge.
    3. Add the edge with maximum dilation.
    """

    def __init__(
        self,
        delta=0.4,
        init_support="mst",
        sampling_mode="full",
        sample_size=1000,
        verbose=False,
        seed=None,
        target_ratio=None,
        slst_num_roots=8,
        slst_eval_sample_size=8192,
        slst_exact_eval_threshold=20000,
    ):
        self.delta = delta
        self.init_support = init_support
        self.sampling_mode = sampling_mode
        self.sample_size = sample_size
        self.verbose = verbose
        self.seed = seed
        self.target_ratio = target_ratio
        self.slst_num_roots = int(slst_num_roots)
        self.slst_eval_sample_size = int(slst_eval_sample_size)
        self.slst_exact_eval_threshold = int(slst_exact_eval_threshold)
        self._rng = random.Random(seed)
        self._torch_gen = torch.Generator()
        if seed is not None:
            self._torch_gen.manual_seed(seed)

    def sparsify(self, data_or_graph):
        if isinstance(data_or_graph, Data) and self.init_support == "mst":
            import os

            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cpp_binary = os.path.join(repo_root, "cpp", "support_graph_sparsifier")
            if os.path.exists(cpp_binary):
                from .support_graph_cpp import SupportGraphCppSparsifier

                cpp = SupportGraphCppSparsifier(
                    delta=self.delta,
                    sampling_mode=self.sampling_mode,
                    sample_size=self.sample_size,
                    node_congestion=False,
                    seed=self.seed if self.seed is not None else 1,
                    verbose=self.verbose,
                    target_ratio=self.target_ratio,
                    binary_path=cpp_binary,
                    batch_mode="single",
                    batch_size=1,
                    mode="dilation",
                )
                try:
                    return cpp.sparsify(data_or_graph)
                except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                    print(
                        f"[SupportGraphDilation] C++ backend unavailable ({type(exc).__name__}: {exc}). "
                        "Falling back to Python implementation.",
                        flush=True,
                    )

        G = self._ensure_nx(data_or_graph)

        delta = self.delta
        if self.target_ratio is not None:
            delta = max(0.001, min(1.0, float(self.target_ratio)))
            if self.verbose:
                print(
                    f"[SupportGraphDilation] Auto-tuned delta={delta:.4f} "
                    f"for target_ratio={self.target_ratio}"
                )

        if self.init_support == "mst":
            weight_key = "weight" if nx.is_weighted(G) else None
            base_support = nx.minimum_spanning_tree(
                G, weight=weight_key, algorithm="kruskal"
            )
        elif self.init_support == "maxst":
            from .max_spanning_tree import MaxSTSparsifier

            base_support = MaxSTSparsifier().sparsify(G)
        elif self.init_support == "spanner":
            from .spanner_halperin import HalperinSpanner

            n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
            if n_nodes <= 1 or n_edges == 0 or delta * n_edges < 1:
                max_spanner_stretch = 2
            else:
                log_val = math.log(delta * n_edges) / math.log(n_nodes) - 1
                if log_val > 0:
                    max_spanner_stretch = max(2, math.ceil(math.pow(log_val, -1)))
                else:
                    max_spanner_stretch = 2
            halperin = HalperinSpanner(stretch=max_spanner_stretch, seed=self.seed)
            base_support = halperin.sparsify(G)
        elif self.init_support == "slst":
            from .scalable_low_stretch_tree import ScalableLowStretchTreeSparsifier

            slst = ScalableLowStretchTreeSparsifier(
                num_roots=self.slst_num_roots,
                eval_sample_size=self.slst_eval_sample_size,
                exact_eval_threshold=self.slst_exact_eval_threshold,
                seed=self.seed,
                verbose=self.verbose,
            )
            base_support = slst.sparsify(G)
        elif self.init_support == "randspt":
            from .random_shortest_path_tree import RandomShortestPathTreeSparsifier

            randspt = RandomShortestPathTreeSparsifier(seed=self.seed)
            base_support = randspt.sparsify(G)
        else:
            raise ValueError(f"Unknown init_support: {self.init_support}")

        H = self._support_graph_dilation(G, delta, base_support)
        if isinstance(data_or_graph, Data):
            return self._ensure_pyg(H, original_data=data_or_graph)
        return H

    def _sample_edges(self, H, candidate_edges):
        edge_list = sorted(candidate_edges)
        if self.sampling_mode == "full":
            return edge_list
        if not edge_list:
            return []

        k = min(self.sample_size, len(edge_list))
        if self.sampling_mode == "random":
            return self._rng.sample(edge_list, k)

        if self.sampling_mode == "weighted":
            deg = dict(H.degree())
            weights = []
            for u, v in edge_list:
                du = max(deg.get(u, 0), 1)
                dv = max(deg.get(v, 0), 1)
                weights.append(1.0 / du + 1.0 / dv)
            weights = torch.tensor(weights, dtype=torch.float)
            weight_sum = weights.sum()
            if weight_sum == 0:
                return self._rng.sample(edge_list, k) if len(edge_list) >= k else edge_list
            probs = weights / weight_sum
            idx = torch.multinomial(
                probs, k, replacement=False, generator=self._torch_gen
            )
            return [edge_list[i] for i in idx.tolist()]

        raise ValueError(f"Unknown sampling_mode: {self.sampling_mode}")

    @staticmethod
    def _edge_weight(G, u, v):
        return float(G[u][v].get("weight", 1.0))

    def _support_graph_dilation(self, G, delta, init):
        m = G.number_of_edges()
        H = init.copy()
        if H.number_of_edges() == 0:
            H = nx.Graph()
            H.add_nodes_from(G.nodes())

        weight_key = "weight" if nx.is_weighted(G) else None
        target_edges = math.ceil(delta * m - 1e-12)

        while H.number_of_edges() < target_edges:
            remaining_edges = set(G.edges()) - set(H.edges())
            if not remaining_edges:
                break

            sampled_edges = self._sample_edges(H, remaining_edges)
            best_edge = None
            best_dilation = float("-inf")

            for u, v in sampled_edges:
                edge_weight = self._edge_weight(G, u, v)
                if edge_weight <= 0:
                    continue
                try:
                    dist_h = nx.shortest_path_length(H, source=u, target=v, weight=weight_key)
                except nx.NetworkXNoPath:
                    best_edge = (u, v)
                    best_dilation = float("inf")
                    break

                dilation = dist_h / edge_weight
                if dilation > best_dilation:
                    best_dilation = dilation
                    best_edge = (u, v)

            if best_edge is None:
                break

            H.add_edge(*best_edge, **G.get_edge_data(*best_edge))

            if self.verbose:
                dilation_str = "inf" if math.isinf(best_dilation) else f"{best_dilation:.4f}"
                print(
                    f"[SupportGraphDilation] added edge={best_edge} "
                    f"dilation={dilation_str} edges={H.number_of_edges()}/{target_edges}"
                )

        return H
