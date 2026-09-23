import math
import random
import subprocess
import torch
import networkx as nx
from torch_geometric.data import Data
from .base import BaseSparsifier

class SupportGraphSparsifier(BaseSparsifier):

    def __init__(
        self,
        delta=0.4,
        init_support='mst',
        node_congestion=True,
        sampling_mode='full',
        sample_size=1000,
        verbose=False,
        seed=None,
        target_ratio=None,
        batch_mode='single',
        batch_size=1,
        slst_num_roots=8,
        slst_eval_sample_size=8192,
        slst_exact_eval_threshold=20000,
    ):
        self.delta = delta
        self.init_support = init_support
        self.node_congestion = node_congestion
        self.sampling_mode = sampling_mode
        self.sample_size = sample_size
        self.verbose = verbose
        self.seed = seed
        self.target_ratio = target_ratio
        self.batch_mode = batch_mode
        self.batch_size = int(batch_size)
        self.slst_num_roots = int(slst_num_roots)
        self.slst_eval_sample_size = int(slst_eval_sample_size)
        self.slst_exact_eval_threshold = int(slst_exact_eval_threshold)
        self._rng = random.Random(seed)
        self._torch_gen = torch.Generator()
        if seed is not None:
            self._torch_gen.manual_seed(seed)

    def sparsify(self, data_or_graph):
        # Prefer the C++ backend only for MST init_support, which is the only
        # initializer implemented there today.
        if isinstance(data_or_graph, Data) and self.init_support == 'mst':
            import os
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cpp_binary = os.path.join(repo_root, 'cpp', 'support_graph_sparsifier')
            if os.path.exists(cpp_binary):
                from .support_graph_cpp import SupportGraphCppSparsifier
                cpp = SupportGraphCppSparsifier(
                    delta=self.delta,
                    sampling_mode=self.sampling_mode,
                    sample_size=self.sample_size,
                    node_congestion=self.node_congestion,
                    seed=self.seed if self.seed is not None else 1,
                    verbose=self.verbose,
                    target_ratio=self.target_ratio,
                    binary_path=cpp_binary,
                    batch_mode=self.batch_mode,
                    batch_size=self.batch_size,
                )
                try:
                    return cpp.sparsify(data_or_graph)
                except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                    print(
                        f"[SupportGraph] C++ backend unavailable ({type(exc).__name__}: {exc}). "
                        "Falling back to Python implementation.",
                        flush=True,
                    )

        G = self._ensure_nx(data_or_graph)

        delta = self.delta
        if self.target_ratio is not None:
            delta = max(0.001, min(1.0, float(self.target_ratio)))
            print(f"[SupportGraph] Auto-tuned delta={delta:.4f} for target_ratio={self.target_ratio}")

        if self.init_support == 'mst':
            base_support = nx.minimum_spanning_tree(G, weight=None, algorithm='kruskal')
        elif self.init_support == 'maxst':
            from .max_spanning_tree import MaxSTSparsifier

            base_support = MaxSTSparsifier().sparsify(G)
        elif self.init_support == 'spanner':
            from .spanner_halperin import HalperinSpanner
            _n, _m = G.number_of_nodes(), G.number_of_edges()
            if _n <= 1 or _m == 0 or delta * _m < 1:
                max_spanner_stretch = 2
            else:
                log_val = math.log(delta * _m) / math.log(_n) - 1
                if log_val > 0:
                    max_spanner_stretch = max(2, math.ceil(math.pow(log_val, -1)))
                else:
                    max_spanner_stretch = 2
            halperin = HalperinSpanner(stretch=max_spanner_stretch, seed=self.seed)
            base_support = halperin.sparsify(G)
        elif self.init_support == 'slst':
            from .scalable_low_stretch_tree import ScalableLowStretchTreeSparsifier

            slst = ScalableLowStretchTreeSparsifier(
                num_roots=self.slst_num_roots,
                eval_sample_size=self.slst_eval_sample_size,
                exact_eval_threshold=self.slst_exact_eval_threshold,
                seed=self.seed,
                verbose=self.verbose,
            )
            base_support = slst.sparsify(G)
        elif self.init_support == 'randspt':
            from .random_shortest_path_tree import RandomShortestPathTreeSparsifier

            randspt = RandomShortestPathTreeSparsifier(seed=self.seed)
            base_support = randspt.sparsify(G)
        else:
            raise ValueError(f'Unknown init_support: {self.init_support}')
        H, _, _ = self._support_graph(G, delta, base_support)
        if isinstance(data_or_graph, Data):
            return self._ensure_pyg(H, original_data=data_or_graph)
        return H

    def _sample_edges(self, G, H, I):
        # Sort for deterministic canonical ordering regardless of Python set hash randomness
        I_list = sorted(I)
        if self.sampling_mode == 'full':
            return I_list
        if not I_list:
            return []
        K = min(self.sample_size, len(I_list))
        if self.sampling_mode == 'random':
            return self._rng.sample(I_list, K)
        if self.sampling_mode == 'weighted':
            deg = dict(H.degree())
            weights = []
            for u, v in I_list:
                du = max(deg.get(u, 0), 1)
                dv = max(deg.get(v, 0), 1)
                weights.append(1.0 / du + 1.0 / dv)
            weights = torch.tensor(weights, dtype=torch.float)
            weight_sum = weights.sum()
            if weight_sum == 0:
                return self._rng.sample(I_list, K) if len(I_list) >= K else I_list
            probs = weights / weight_sum
            idx = torch.multinomial(probs, K, replacement=False, generator=self._torch_gen)
            return [I_list[i] for i in idx.tolist()]
        raise ValueError(f'Unknown sampling_mode: {self.sampling_mode}')

    def _support_graph(self, G, delta, init):
        m = G.number_of_edges()
        H = init.copy()
        if init.number_of_edges() == 0:
            H = nx.Graph()
            H.add_nodes_from(G)
        while H.number_of_edges() < delta * m:
            I = set(G.edges) - set(H.edges)
            if not I:
                break
            I_sampled = self._sample_edges(G, H, I)
            congestions = {}
            max_congestion = 0
            max_congestion_obj = None
            max_dilation = 0
            disconnected_edges = []
            for e in I_sampled:
                try:
                    p = nx.shortest_path(H, source=e[0], target=e[1], weight=None)
                except nx.NetworkXNoPath:
                    disconnected_edges.append(e)
                    continue
                dilation_e = len(p) - 1
                if dilation_e > max_dilation:
                    max_dilation = dilation_e
                if self.node_congestion:
                    for v in p:
                        if v not in congestions:
                            congestions[v] = [0, []]
                        congestions[v][0] += 1
                        congestions[v][1].append((e, dilation_e))
                        if congestions[v][0] > max_congestion:
                            max_congestion = congestions[v][0]
                            max_congestion_obj = v
                else:
                    for i in range(len(p) - 1):
                        pe = (min(p[i], p[i + 1]), max(p[i], p[i + 1]))
                        if pe not in congestions:
                            congestions[pe] = [0, []]
                        congestions[pe][0] += 1
                        congestions[pe][1].append((e, dilation_e))
                        if congestions[pe][0] > max_congestion:
                            max_congestion = congestions[pe][0]
                            max_congestion_obj = pe
            if not congestions:
                if disconnected_edges:
                    H.add_edge(*disconnected_edges[0])
                    continue
                else:
                    break
            best_edge, _ = max(congestions[max_congestion_obj][1], key=lambda x: x[1])
            H.add_edge(*best_edge)
        return (H, None, None)
