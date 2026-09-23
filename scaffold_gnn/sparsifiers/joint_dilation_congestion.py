import heapq
import math
import random

import networkx as nx
import torch
from torch_geometric.data import Data

from .base import BaseSparsifier
from .scaffold.spanning_tree import (
    FAST_TREE_DEFAULT_BUCKETS,
    build_fast_weighted_spanning_forest_nx,
    build_random_spanning_forest_nx,
    build_spanning_forest_nx,
    canonical_support_name,
)


class JointDilationCongestionSparsifier(BaseSparsifier):
    """Unified JOINT-DILATION-CONGESTION-SPARSIFY(G, delta, alpha, beta_E, beta_V).

    Starts from a low-dilation spanning tree/forest H and iteratively
    adds the missing edge with the highest joint score:

        score(e) = ((dil[e]+eps)/(D_max+eps))^alpha
                 * ((eConPath[e]+eps)/(E_max+eps))^beta_E
                 * ((vConPath[e]+eps)/(V_max+eps))^beta_V

    where at every iteration
        dil[e]   = dist_H(u, v) / w(e)
        eCon[e'] = number of sampled missing edges routed through edge e' of H
        vCon[x]  = number of sampled missing edges routed through internal node x
        eConPath[e]  = normalized p-norm congestion on the replacement path edges
        vConPath[e]  = normalized q-norm congestion on the replacement path nodes

    ``init_support`` controls the base tree:
        "mst"     --- plain minimum spanning tree (default, hop-count).
        "maxst"   --- maximum spanning tree.
        "fast-mst" / "fast-maxst" --- Benchmark-compatible bucketed,
                         approximate weighted spanning forests.
        "randst"  --- uniform random spanning tree (Wilson).
        "glst"    --- the GreedyLowStretchTreeSparsifier from this module.
        "slst"    --- scalable multi-root shortest-path-tree heuristic.
        "randspt" --- randomized shortest-path tree.
        "llst"    --- local-search low-stretch tree for small graphs.
    Alternatively, pass a pre-built ``nx.Graph`` (or any object with
    ``edges()`` / ``nodes()``) to use it directly as the base support.

    The target edge count is ``ceil(delta * |E|)`` (or
    ``ceil(target_ratio * |E|)`` when ``target_ratio`` is set, for API
    parity with the other sparsifiers).

    Backward compatibility:
        ``beta`` is treated as a shared fallback for ``edge_beta`` and
        ``node_beta`` when the more specific values are not provided.

    ``growth_mode`` controls how the candidate-addition loop is run:
        "exact"      --- recompute the sampled candidate pool every step.
        "lazy_topk"  --- keep stale heap scores and refresh only the
                         active top-k candidates, with optional periodic
                         sampled rebuilds.
    """

    def __init__(
        self,
        delta=0.40,
        alpha=1.0,
        beta=1.0,
        edge_beta=None,
        node_beta=None,
        edge_norm_p=2.0,
        node_norm_q=2.0,
        init_support="glst",
        fast_tree_buckets=FAST_TREE_DEFAULT_BUCKETS,
        glst_alpha=1.0,
        glst_eta=1.0,
        sampling_mode="weighted",
        sample_size=256,
        batch_mode="topk",
        batch_size=32,
        growth_mode="exact",
        lazy_top_k=16,
        lazy_rebuild_interval=0,
        swap_refine=False,
        swap_max_passes=5,
        swap_sampling_mode="full",
        swap_sample_size=256,
        swap_cycle_sample_size=0,
        swap_no_improve_patience=1,
        swap_search_mode="restart",
        swap_lazy_top_k=16,
        swap_lazy_rebuild_interval=0,
        slst_num_roots=8,
        slst_eval_sample_size=8192,
        slst_exact_eval_threshold=20000,
        slst_fast=False,
        llst_max_passes=10,
        llst_init_support="glst",
        llst_glst_alpha=1.0,
        llst_glst_eta=1.0,
        llst_candidate_strategy="random",
        llst_candidate_sample_size=0,
        llst_eval_sample_size=0,
        llst_cycle_sample_size=0,
        llst_resample_eval_each_pass=False,
        llst_verbose=False,
        support_budget_mode="early_stop",
        support_weight_method="uniform",
        weighted_paths=False,
        verbose=False,
        seed=None,
        target_ratio=None,
    ):
        self.delta = float(delta)
        self.alpha = float(alpha)
        beta_fallback = float(beta)
        self.edge_beta = beta_fallback if edge_beta is None else float(edge_beta)
        self.node_beta = beta_fallback if node_beta is None else float(node_beta)
        self.edge_norm_p = float(edge_norm_p)
        self.node_norm_q = float(node_norm_q)
        self.init_support = (
            init_support
            if isinstance(init_support, nx.Graph)
            else canonical_support_name(init_support)
        )
        self.fast_tree_buckets = int(fast_tree_buckets)
        self.glst_alpha = float(glst_alpha)
        self.glst_eta = float(glst_eta)
        self.sampling_mode = str(sampling_mode)
        self.sample_size = int(sample_size)
        self.batch_mode = str(batch_mode)
        self.batch_size = int(batch_size)
        self.growth_mode = str(growth_mode)
        self.lazy_top_k = max(1, int(lazy_top_k))
        self.lazy_rebuild_interval = max(0, int(lazy_rebuild_interval))
        self.swap_refine = bool(swap_refine)
        self.swap_max_passes = max(0, int(swap_max_passes))
        self.swap_sampling_mode = str(swap_sampling_mode)
        self.swap_sample_size = max(1, int(swap_sample_size))
        self.swap_cycle_sample_size = int(swap_cycle_sample_size)
        self.swap_no_improve_patience = max(1, int(swap_no_improve_patience))
        self.swap_search_mode = str(swap_search_mode)
        self.swap_lazy_top_k = max(1, int(swap_lazy_top_k))
        self.swap_lazy_rebuild_interval = max(0, int(swap_lazy_rebuild_interval))
        self.slst_num_roots = int(slst_num_roots)
        self.slst_eval_sample_size = int(slst_eval_sample_size)
        self.slst_exact_eval_threshold = int(slst_exact_eval_threshold)
        self.slst_fast = bool(slst_fast)
        self.llst_max_passes = int(llst_max_passes)
        self.llst_init_support = canonical_support_name(llst_init_support)
        self.llst_glst_alpha = float(llst_glst_alpha)
        self.llst_glst_eta = float(llst_glst_eta)
        self.llst_candidate_strategy = str(llst_candidate_strategy)
        self.llst_candidate_sample_size = int(llst_candidate_sample_size)
        self.llst_eval_sample_size = int(llst_eval_sample_size)
        self.llst_cycle_sample_size = int(llst_cycle_sample_size)
        self.llst_resample_eval_each_pass = bool(llst_resample_eval_each_pass)
        self.llst_verbose = bool(llst_verbose)
        normalized_budget_mode = str(support_budget_mode).strip().lower().replace("-", "_")
        budget_mode_aliases = {
            "early": "early_stop",
            "stop_early": "early_stop",
            "full_then_trim": "full_then_random_trim",
            "benchmark": "full_then_random_trim",
        }
        self.support_budget_mode = budget_mode_aliases.get(
            normalized_budget_mode,
            normalized_budget_mode,
        )
        if self.support_budget_mode not in ("early_stop", "full_then_random_trim"):
            raise ValueError(
                "support_budget_mode must be 'early_stop' or "
                "'full_then_random_trim'."
            )
        # Dilation numerator: hop count (False) or the sum of tree edge weights
        # along the support path (True). tree_score.scaffold_tree_scores has
        # implemented and tested both since the start; nothing reached it from
        # the CLI until now, so every measured cell to date used hop counts.
        self.weighted_paths = bool(weighted_paths)
        self.support_weight_method = str(support_weight_method).strip().lower()
        if self.support_weight_method not in ("uniform", "cosine", "euclidean", "dot"):
            raise ValueError(
                "support_weight_method must be one of "
                "{'uniform', 'cosine', 'euclidean', 'dot'}."
            )
        self.verbose = verbose
        self.seed = seed
        self.target_ratio = target_ratio
        self._eps = 1e-8
        self._rng = random.Random(seed)
        self._torch_gen = torch.Generator()
        if seed is not None:
            self._torch_gen.manual_seed(seed)

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------
    def sparsify(self, data_or_graph):
        G = self._ensure_nx(data_or_graph)

        delta = self.delta
        if self.target_ratio is not None:
            delta = max(0.001, min(1.0, float(self.target_ratio)))
            if self.verbose:
                print(
                    f"[JointDilCon] Auto-tuned delta={delta:.4f} "
                    f"for target_ratio={self.target_ratio}"
                )

        target_edges = math.ceil(delta * G.number_of_edges() - 1e-12)
        base_support = self._prepare_init_support(G, target_edges)
        H = self._grow(G, base_support, delta)
        self._record_support_budget_stats()
        if self.swap_refine and self.swap_max_passes > 0:
            H = self._refine_with_swaps(G, H)

        if isinstance(data_or_graph, Data):
            return self._ensure_pyg(H, original_data=data_or_graph)
        return H

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _edge_weight(self, G, u, v):
        data = G.get_edge_data(u, v)
        return float(data.get("weight", 1.0)) if data else 1.0

    @torch.no_grad()
    def _compute_feature_edge_scores(self, features, src, dst, batch_size=1_000_000):
        method = self.support_weight_method
        src = torch.as_tensor(src).detach().cpu().long().reshape(-1)
        dst = torch.as_tensor(dst).detach().cpu().long().reshape(-1)
        if method == "uniform":
            return torch.ones(src.numel(), dtype=torch.float)
        if features is None:
            raise ValueError(
                f"support_weight_method='{method}' requires node features."
            )
        x = torch.as_tensor(features).detach().cpu().float()
        if x.ndim == 1:
            x = x.unsqueeze(1)
        parts = []
        for start in range(0, src.numel(), max(1, int(batch_size))):
            stop = min(src.numel(), start + max(1, int(batch_size)))
            left = x[src[start:stop]]
            right = x[dst[start:stop]]
            if method == "cosine":
                scores = torch.nn.functional.cosine_similarity(
                    left,
                    right,
                    dim=-1,
                    eps=1e-12,
                )
                scores = (scores + 1.0) * 0.5
            elif method == "euclidean":
                scores = 1.0 / (
                    1.0 + torch.linalg.vector_norm(left - right, ord=2, dim=-1)
                )
            else:
                scores = (left * right).sum(dim=-1)
            parts.append(scores.float())
        output = torch.cat(parts) if parts else torch.empty(0, dtype=torch.float)
        output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        if method == "dot" and output.numel():
            low = output.min()
            high = output.max()
            output = (
                torch.ones_like(output)
                if float(high - low) <= 1e-12
                else (output - low) / (high - low)
            )
        return output.clamp(0.0, 1.0)

    def _ensure_nx(self, data_or_graph):
        G = super()._ensure_nx(data_or_graph)
        if not isinstance(data_or_graph, Data) or self.support_weight_method == "uniform":
            return G
        edge_rows = list(G.edges())
        if not edge_rows:
            return G
        src = [u for u, _ in edge_rows]
        dst = [v for _, v in edge_rows]
        scores = self._compute_feature_edge_scores(data_or_graph.x, src, dst)
        nx.set_edge_attributes(
            G,
            {
                edge: float(score)
                for edge, score in zip(edge_rows, scores.tolist())
            },
            "weight",
        )
        return G

    def _canon_edge(self, u, v):
        return (u, v) if repr(u) <= repr(v) else (v, u)

    def _support_trim_seed(self):
        active_seed = getattr(self, "_active_seed", None)
        if callable(active_seed):
            return active_seed()
        return self.seed

    def _prepare_init_support(self, G, target_edges):
        """Build a budget prefix or build the full support and trim it.

        ``early_stop`` preserves the original Scaffold behavior.  The second
        mode mirrors Benchmark's layer handling: construct the complete support
        first, then choose a uniform target-sized subset using the active run
        or resparsification seed.
        """
        build_limit = target_edges if self.support_budget_mode == "early_stop" else None
        support = self._build_init_support(G, max_edges=build_limit)
        before_trim = support.number_of_edges()
        trimmed = 0

        if (
            self.support_budget_mode == "full_then_random_trim"
            and before_trim > target_edges
        ):
            edge_rows = sorted(
                support.edges(data=True),
                key=lambda row: repr(self._canon_edge(row[0], row[1])),
            )
            generator = torch.Generator(device="cpu")
            trim_seed = self._support_trim_seed()
            if trim_seed is None:
                generator.seed()
            else:
                generator.manual_seed(int(trim_seed))
            # Benchmark first samples its one selected support layer, then
            # drops the excess positions from that layer. Reproduce those RNG
            # transitions so equal seeds select equal subsets.
            torch.multinomial(
                torch.ones(1, dtype=torch.double),
                num_samples=1,
                replacement=False,
                generator=generator,
            )
            drop_count = before_trim - int(target_edges)
            retained = torch.randperm(
                before_trim,
                generator=generator,
            )[drop_count:].sort().values.tolist()
            trimmed_support = support.__class__()
            trimmed_support.graph.update(support.graph)
            trimmed_support.add_nodes_from(support.nodes(data=True))
            trimmed_support.add_edges_from(edge_rows[index] for index in retained)
            support = trimmed_support
            trimmed = before_trim - support.number_of_edges()

        self._support_budget_stats = {
            "support_budget_mode": self.support_budget_mode,
            "support_edges_before_trim": int(before_trim),
            "support_edges_trimmed": int(trimmed),
            "support_edges_after_trim": int(support.number_of_edges()),
        }
        if self.support_budget_mode == "full_then_random_trim":
            print(
                "[SCAFFOLD support budget] "
                f"mode={self.support_budget_mode} before_trim={before_trim} "
                f"trimmed={trimmed} after_trim={support.number_of_edges()} "
                f"target={target_edges}"
            )
        return support

    def _record_support_budget_stats(self):
        stats = getattr(self, "_support_budget_stats", None)
        if not isinstance(stats, dict):
            return
        last_stats = getattr(self, "last_cluster_stats", None)
        if not isinstance(last_stats, dict):
            last_stats = {}
            self.last_cluster_stats = last_stats
        last_stats.update(stats)

    def _build_init_support(self, G, max_edges=None):
        if isinstance(self.init_support, nx.Graph):
            return self.init_support.copy()
        if self.init_support == "mst":
            weight_key = "weight" if nx.is_weighted(G) else None
            return build_spanning_forest_nx(
                G,
                weight_key=weight_key,
                maximum=False,
                max_edges=max_edges,
            )
        if self.init_support == "maxst":
            weight_key = "weight" if nx.is_weighted(G) else None
            return build_spanning_forest_nx(
                G,
                weight_key=weight_key,
                maximum=True,
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
        if self.init_support == "glst":
            from .low_stretch_tree import GreedyLowStretchTreeSparsifier

            glst = GreedyLowStretchTreeSparsifier(
                alpha=self.glst_alpha,
                eta=self.glst_eta,
                verbose=self.verbose,
                seed=self.seed,
            )
            return glst._build_tree(G, max_edges=max_edges)
        if self.init_support == "slst":
            from .scalable_low_stretch_tree import ScalableLowStretchTreeSparsifier

            slst = ScalableLowStretchTreeSparsifier(
                num_roots=self.slst_num_roots,
                eval_sample_size=self.slst_eval_sample_size,
                exact_eval_threshold=self.slst_exact_eval_threshold,
                seed=self.seed,
                verbose=self.verbose,
                fast_mode=self.slst_fast,
                target_ratio=(
                    None
                    if max_edges is None
                    else float(max_edges) / max(1, G.number_of_edges())
                ),
            )
            return slst.sparsify(G)
        if self.init_support == "randspt":
            from .random_shortest_path_tree import RandomShortestPathTreeSparsifier

            return RandomShortestPathTreeSparsifier(
                seed=self.seed,
                target_ratio=(
                    None
                    if max_edges is None
                    else float(max_edges) / max(1, G.number_of_edges())
                ),
            ).sparsify(G)
        if self.init_support == "llst":
            from .local_search_low_stretch_tree import LocalSearchLowStretchTreeSparsifier

            llst = LocalSearchLowStretchTreeSparsifier(
                max_passes=self.llst_max_passes,
                init_support=self.llst_init_support,
                fast_tree_buckets=self.fast_tree_buckets,
                glst_alpha=self.llst_glst_alpha,
                glst_eta=self.llst_glst_eta,
                candidate_strategy=self.llst_candidate_strategy,
                candidate_sample_size=self.llst_candidate_sample_size,
                eval_sample_size=self.llst_eval_sample_size,
                cycle_sample_size=self.llst_cycle_sample_size,
                resample_eval_each_pass=self.llst_resample_eval_each_pass,
                verbose=self.llst_verbose or self.verbose,
                seed=self.seed,
                target_ratio=(
                    None
                    if max_edges is None
                    else float(max_edges) / max(1, G.number_of_edges())
                ),
            )
            return llst.sparsify(G)
        raise ValueError(f"Unknown init_support: {self.init_support!r}")

    def _sample_edges(self, H, candidate_edges, mode=None, sample_size=None):
        mode = self.sampling_mode if mode is None else str(mode)
        edge_list = sorted(candidate_edges)
        if mode == "full":
            return edge_list
        if not edge_list:
            return []

        chosen_sample_size = self.sample_size if sample_size is None else int(sample_size)
        k = min(max(1, chosen_sample_size), len(edge_list))
        if mode == "random":
            return self._rng.sample(edge_list, k)

        if mode == "weighted":
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

        raise ValueError(f"Unknown sampling_mode: {mode}")

    def _choose_edges_to_add(self, candidates, scores, remaining_budget):
        if not candidates or remaining_budget <= 0:
            return []
        if self.batch_mode == "single":
            return [max(candidates, key=lambda e: scores[e])]
        ordered = sorted(candidates, key=lambda e: scores[e], reverse=True)
        if self.batch_mode == "topk":
            limit = max(1, min(self.batch_size, remaining_budget, len(ordered)))
            return ordered[:limit]
        if self.batch_mode == "all":
            return ordered[:remaining_budget]
        raise ValueError(f"Unknown batch_mode: {self.batch_mode}")

    def _candidate_path_stats(self, G, H, edge, weight_key, is_weighted):
        u, v = edge
        try:
            path = nx.shortest_path(H, u, v, weight=weight_key)
        except nx.NetworkXNoPath:
            return {
                "edge": edge,
                "disconnected": True,
                "dil": float("inf"),
                "path_edges": [],
                "path_nodes": [],
            }

        if is_weighted:
            dist = 0.0
            for i in range(len(path) - 1):
                dist += self._edge_weight(H, path[i], path[i + 1])
        else:
            dist = len(path) - 1

        return {
            "edge": edge,
            "disconnected": False,
            "dil": dist / max(self._edge_weight(G, u, v), self._eps),
            "path_edges": [
                self._canon_edge(path[i], path[i + 1]) for i in range(len(path) - 1)
            ],
            "path_nodes": list(path[1:-1]),
        }

    def _score_candidate(self, dil, eConPath, vConPath, d_max, e_max, v_max):
        if math.isinf(dil):
            return float("inf")
        eps = self._eps
        a_term = ((dil + eps) / (d_max + eps)) ** self.alpha
        e_term = ((eConPath + eps) / (e_max + eps)) ** self.edge_beta
        v_term = ((vConPath + eps) / (v_max + eps)) ** self.node_beta
        return a_term * e_term * v_term

    def _evaluate_candidate_set(self, G, H, candidate_edges, weight_key, is_weighted):
        candidate_edges = list(dict.fromkeys(candidate_edges))
        if not candidate_edges:
            return {}

        edge_con = {}
        node_con = {}
        metrics = {}

        for edge in candidate_edges:
            stats = self._candidate_path_stats(G, H, edge, weight_key, is_weighted)
            metrics[edge] = stats
            if stats["disconnected"]:
                continue
            for edge_key in stats["path_edges"]:
                edge_con[edge_key] = edge_con.get(edge_key, 0) + 1
            for node in stats["path_nodes"]:
                node_con[node] = node_con.get(node, 0) + 1

        for edge, stats in metrics.items():
            stats["eConPath"] = self._norm_congestion(
                [edge_con.get(edge_key, 0) for edge_key in stats["path_edges"]],
                self.edge_norm_p,
            )
            stats["vConPath"] = self._norm_congestion(
                [node_con.get(node, 0) for node in stats["path_nodes"]],
                self.node_norm_q,
            )

        finite_dils = [
            stats["dil"]
            for stats in metrics.values()
            if not stats["disconnected"] and not math.isinf(stats["dil"])
        ]
        d_max = max(finite_dils) if finite_dils else self._eps
        e_max = max((stats["eConPath"] for stats in metrics.values()), default=self._eps)
        v_max = max((stats["vConPath"] for stats in metrics.values()), default=self._eps)

        for stats in metrics.values():
            stats["score"] = self._score_candidate(
                stats["dil"],
                stats["eConPath"],
                stats["vConPath"],
                d_max,
                e_max,
                v_max,
            )

        return metrics

    def _norm_congestion(self, values, order):
        if not values:
            return 0.0
        if math.isinf(order):
            return max(float(v) for v in values)
        if order <= 0:
            raise ValueError(f"Norm order must be positive, got {order}")
        eps = self._eps
        total = sum((float(v) + eps) ** order for v in values)
        return (total / (len(values) + eps)) ** (1.0 / order)

    def _evaluate_support_objective(self, G, H, missing_edges=None):
        weight_key = "weight" if nx.is_weighted(G) else None
        is_weighted = nx.is_weighted(G)
        eps = self._eps

        if missing_edges is None:
            H_edge_set = {self._canon_edge(u, v) for u, v in H.edges()}
            missing_edges = [
                self._canon_edge(u, v)
                for u, v in G.edges()
                if self._canon_edge(u, v) not in H_edge_set
            ]
        else:
            missing_edges = [self._canon_edge(u, v) for (u, v) in missing_edges]
        if not missing_edges:
            return {
                "objective": 0.0,
                "avg_dil": 0.0,
                "avg_eConPath": 0.0,
                "avg_vConPath": 0.0,
                "num_missing_edges": 0,
                "edge_congestion": {},
                "node_congestion": {},
            }

        edge_con = {self._canon_edge(u, v): 0 for u, v in H.edges()}
        node_con = {v: 0 for v in H.nodes()}
        dil = {}
        path_edges_map = {}
        path_nodes_map = {}

        for (u, v) in missing_edges:
            try:
                path = nx.shortest_path(H, u, v, weight=weight_key)
            except nx.NetworkXNoPath:
                return {
                    "objective": float("inf"),
                    "avg_dil": float("inf"),
                    "avg_eConPath": float("inf"),
                    "avg_vConPath": float("inf"),
                    "num_missing_edges": len(missing_edges),
                    "edge_congestion": edge_con,
                    "node_congestion": node_con,
                }

            if is_weighted:
                dist = 0.0
                for i in range(len(path) - 1):
                    dist += self._edge_weight(H, path[i], path[i + 1])
            else:
                dist = len(path) - 1

            dil[(u, v)] = dist / max(self._edge_weight(G, u, v), eps)
            path_edges = [
                self._canon_edge(path[i], path[i + 1]) for i in range(len(path) - 1)
            ]
            path_nodes = list(path[1:-1])
            path_edges_map[(u, v)] = path_edges
            path_nodes_map[(u, v)] = path_nodes
            for edge_key in path_edges:
                edge_con[edge_key] = edge_con.get(edge_key, 0) + 1
            for node in path_nodes:
                node_con[node] = node_con.get(node, 0) + 1

        total_objective = 0.0
        total_dil = 0.0
        total_eConPath = 0.0
        total_vConPath = 0.0

        for edge in missing_edges:
            eConPath = self._norm_congestion(
                [edge_con.get(edge_key, 0) for edge_key in path_edges_map[edge]],
                self.edge_norm_p,
            )
            vConPath = self._norm_congestion(
                [node_con.get(node, 0) for node in path_nodes_map[edge]],
                self.node_norm_q,
            )

            total_dil += dil[edge]
            total_eConPath += eConPath
            total_vConPath += vConPath

            d_term = (dil[edge] + eps) ** self.alpha if self.alpha != 0 else 1.0
            e_term = (eConPath + eps) ** self.edge_beta if self.edge_beta != 0 else 1.0
            v_term = (vConPath + eps) ** self.node_beta if self.node_beta != 0 else 1.0
            total_objective += d_term * e_term * v_term

        denom = float(len(missing_edges))
        return {
            "objective": total_objective / denom,
            "avg_dil": total_dil / denom,
            "avg_eConPath": total_eConPath / denom,
            "avg_vConPath": total_vConPath / denom,
            "num_missing_edges": len(missing_edges),
            "edge_congestion": edge_con,
            "node_congestion": node_con,
        }

    def _sample_cycle_edges(self, cycle_edges, edge_loads=None):
        if self.swap_cycle_sample_size <= 0:
            return cycle_edges
        if len(cycle_edges) <= self.swap_cycle_sample_size:
            return cycle_edges
        if edge_loads:
            return sorted(
                cycle_edges,
                key=lambda edge: (-edge_loads.get(edge, 0), repr(edge)),
            )[: self.swap_cycle_sample_size]
        return self._rng.sample(cycle_edges, self.swap_cycle_sample_size)

    def _prepare_trial_eval_edges(self, eval_edges, add_edge, remove_edge):
        trial_eval_edges = [
            self._canon_edge(u, v)
            for (u, v) in eval_edges
            if self._canon_edge(u, v) != add_edge
        ]
        remove_edge = self._canon_edge(*remove_edge)
        if remove_edge not in trial_eval_edges:
            trial_eval_edges.append(remove_edge)
        return trial_eval_edges

    def _best_swap_for_add_edge(self, G, H, add_edge, eval_edges, weight_key, edge_loads=None):
        u, v = add_edge
        add_edge_data = G.get_edge_data(u, v) or {}
        try:
            path = nx.shortest_path(H, u, v, weight=weight_key)
        except nx.NetworkXNoPath:
            return None

        cycle_edges = self._sample_cycle_edges([
            self._canon_edge(path[i], path[i + 1]) for i in range(len(path) - 1)
        ], edge_loads=edge_loads)
        best = None
        best_objective = float("inf")

        for remove_edge in cycle_edges:
            trial = H.copy()
            trial.add_edge(u, v, **add_edge_data)
            trial.remove_edge(*remove_edge)
            trial_eval_edges = self._prepare_trial_eval_edges(
                eval_edges, add_edge, remove_edge
            )
            trial_metrics = self._evaluate_support_objective(
                G, trial, missing_edges=trial_eval_edges
            )
            if best is None or trial_metrics["objective"] + self._eps < best_objective:
                best = (trial, trial_metrics, add_edge, remove_edge, trial_eval_edges)
                best_objective = trial_metrics["objective"]

        return best

    def _refresh_swap_candidate(self, G, H, add_edge, eval_edges, current_metrics, weight_key):
        current_objective = current_metrics["objective"]
        best = self._best_swap_for_add_edge(
            G,
            H,
            add_edge,
            eval_edges,
            weight_key,
            edge_loads=current_metrics.get("edge_congestion"),
        )
        if best is None:
            return None
        trial, trial_metrics, add_edge, remove_edge, trial_eval_edges = best
        if math.isinf(trial_metrics["objective"]):
            score = float("-inf")
        else:
            score = current_objective - trial_metrics["objective"]
        return {
            "score": score,
            "trial": trial,
            "metrics": trial_metrics,
            "add_edge": add_edge,
            "remove_edge": remove_edge,
            "trial_eval_edges": trial_eval_edges,
        }

    def _best_improving_swap(self, G, H, current_metrics, eval_edges):
        current_objective = current_metrics["objective"]
        if math.isinf(current_objective):
            return None

        weight_key = "weight" if nx.is_weighted(G) else None
        H_edge_set = {self._canon_edge(u, v) for u, v in H.edges()}
        all_missing_edges = [
            self._canon_edge(u, v) for u, v in G.edges() if self._canon_edge(u, v) not in H_edge_set
        ]
        add_candidates = self._sample_edges(
            H,
            all_missing_edges,
            mode=self.swap_sampling_mode,
            sample_size=self.swap_sample_size,
        )
        best = None
        best_objective = current_objective

        for add_edge in add_candidates:
            refreshed = self._refresh_swap_candidate(
                G,
                H,
                add_edge,
                eval_edges,
                current_metrics,
                weight_key,
            )
            if refreshed is None:
                continue
            if refreshed["metrics"]["objective"] + self._eps < best_objective:
                best = (
                    refreshed["trial"],
                    refreshed["metrics"],
                    refreshed["add_edge"],
                    refreshed["remove_edge"],
                )
                best_objective = refreshed["metrics"]["objective"]

        return best

    def _refine_with_swaps(self, G, H):
        if self.swap_search_mode == "restart":
            return self._refine_with_swaps_restart(G, H)
        if self.swap_search_mode == "lazy_topk":
            return self._refine_with_swaps_lazy_topk(G, H)
        raise ValueError(f"Unknown swap_search_mode: {self.swap_search_mode!r}")

    def _refine_with_swaps_restart(self, G, H):
        H_edge_set = {self._canon_edge(u, v) for u, v in H.edges()}
        missing_edges = [
            self._canon_edge(u, v) for u, v in G.edges() if self._canon_edge(u, v) not in H_edge_set
        ]
        eval_edges = self._sample_edges(
            H,
            missing_edges,
            mode=self.swap_sampling_mode,
            sample_size=self.swap_sample_size,
        )
        current_metrics = self._evaluate_support_objective(G, H, missing_edges=eval_edges)
        if self.verbose:
            print(
                f"[JointDilConSwap] start objective={current_metrics['objective']:.6f} "
                f"avg_dil={current_metrics['avg_dil']:.4f} "
                f"avg_eConPath={current_metrics['avg_eConPath']:.4f} "
                f"avg_vConPath={current_metrics['avg_vConPath']:.4f} "
                f"sampled_missing={len(eval_edges)}"
            )

        accepted_swaps = 0
        stalled_passes = 0
        while accepted_swaps < self.swap_max_passes and stalled_passes < self.swap_no_improve_patience:
            H_edge_set = {self._canon_edge(u, v) for u, v in H.edges()}
            missing_edges = [
                self._canon_edge(u, v)
                for u, v in G.edges()
                if self._canon_edge(u, v) not in H_edge_set
            ]
            eval_edges = self._sample_edges(
                H,
                missing_edges,
                mode=self.swap_sampling_mode,
                sample_size=self.swap_sample_size,
            )
            current_metrics = self._evaluate_support_objective(
                G, H, missing_edges=eval_edges
            )
            best = self._best_improving_swap(G, H, current_metrics, eval_edges)
            if best is None:
                stalled_passes += 1
                if self.verbose:
                    print(
                        f"[JointDilConSwap] no improvement on sampled pass "
                        f"{stalled_passes}/{self.swap_no_improve_patience} "
                        f"sampled_missing={len(eval_edges)}"
                    )
                continue
            stalled_passes = 0
            accepted_swaps += 1
            H, current_metrics, add_edge, remove_edge = best
            if self.verbose:
                print(
                    f"[JointDilConSwap] pass={accepted_swaps} add={add_edge} remove={remove_edge} "
                    f"objective={current_metrics['objective']:.6f} "
                    f"avg_dil={current_metrics['avg_dil']:.4f} "
                    f"avg_eConPath={current_metrics['avg_eConPath']:.4f} "
                    f"avg_vConPath={current_metrics['avg_vConPath']:.4f} "
                    f"sampled_missing={len(eval_edges)}"
                )

        return H

    def _refine_with_swaps_lazy_topk(self, G, H):
        weight_key = "weight" if nx.is_weighted(G) else None
        eps = self._eps

        heap = []
        entry_versions = {}
        entry_counter = 0
        additions_since_rebuild = 0
        rebuild_count = 0

        def current_missing_edges():
            H_edge_set = {self._canon_edge(u, v) for u, v in H.edges()}
            return [
                self._canon_edge(u, v)
                for u, v in G.edges()
                if self._canon_edge(u, v) not in H_edge_set
            ]

        def push_entry(edge, score):
            nonlocal entry_counter
            entry_counter += 1
            entry_versions[edge] = entry_counter
            heapq.heappush(heap, (-score, entry_counter, edge))

        def rebuild_heap():
            nonlocal heap, entry_versions, additions_since_rebuild, rebuild_count
            missing_edges = current_missing_edges()
            eval_edges = self._sample_edges(
                H,
                missing_edges,
                mode=self.swap_sampling_mode,
                sample_size=self.swap_sample_size,
            )
            current_metrics = self._evaluate_support_objective(
                G, H, missing_edges=eval_edges
            )
            heap = []
            entry_versions = {}
            if not math.isinf(current_metrics["objective"]):
                add_candidates = self._sample_edges(
                    H,
                    missing_edges,
                    mode=self.swap_sampling_mode,
                    sample_size=self.swap_sample_size,
                )
                for add_edge in add_candidates:
                    refreshed = self._refresh_swap_candidate(
                        G,
                        H,
                        add_edge,
                        eval_edges,
                        current_metrics,
                        weight_key,
                    )
                    if refreshed is None:
                        continue
                    push_entry(add_edge, refreshed["score"])
            additions_since_rebuild = 0
            rebuild_count += 1
            if self.verbose:
                print(
                    f"[JointDilConSwapLazy] rebuild={rebuild_count} "
                    f"objective={current_metrics['objective']:.6f} "
                    f"sampled_missing={len(eval_edges)} "
                    f"edges={H.number_of_edges()}"
                )
            return current_metrics, eval_edges

        current_metrics, eval_edges = rebuild_heap()
        if self.verbose:
            print(
                f"[JointDilConSwap] start objective={current_metrics['objective']:.6f} "
                f"avg_dil={current_metrics['avg_dil']:.4f} "
                f"avg_eConPath={current_metrics['avg_eConPath']:.4f} "
                f"avg_vConPath={current_metrics['avg_vConPath']:.4f} "
                f"sampled_missing={len(eval_edges)}"
            )

        accepted_swaps = 0
        stalled_passes = 0
        while accepted_swaps < self.swap_max_passes and stalled_passes < self.swap_no_improve_patience:
            missing_edge_set = set(current_missing_edges())
            if not missing_edge_set:
                break
            if not heap or (
                self.swap_lazy_rebuild_interval > 0
                and additions_since_rebuild >= self.swap_lazy_rebuild_interval
            ):
                current_metrics, eval_edges = rebuild_heap()
                missing_edge_set = set(current_missing_edges())
                if not heap:
                    break

            active_limit = min(self.swap_lazy_top_k, len(missing_edge_set))
            active_edges = []
            while heap and len(active_edges) < active_limit:
                _, version, edge = heapq.heappop(heap)
                if edge not in missing_edge_set:
                    continue
                if entry_versions.get(edge) != version:
                    continue
                active_edges.append(edge)

            if not active_edges:
                current_metrics, eval_edges = rebuild_heap()
                if not heap:
                    break
                continue

            refreshed_candidates = []
            for add_edge in active_edges:
                refreshed = self._refresh_swap_candidate(
                    G,
                    H,
                    add_edge,
                    eval_edges,
                    current_metrics,
                    weight_key,
                )
                if refreshed is not None:
                    refreshed_candidates.append(refreshed)

            best = None
            if refreshed_candidates:
                best = max(refreshed_candidates, key=lambda item: item["score"])

            if best is None or best["score"] <= eps:
                stalled_passes += 1
                if self.verbose:
                    best_score = "n/a" if best is None else f"{best['score']:.6f}"
                    print(
                        f"[JointDilConSwapLazy] no improvement pass "
                        f"{stalled_passes}/{self.swap_no_improve_patience} "
                        f"active={len(active_edges)} best_score={best_score}"
                    )
                continue

            H = best["trial"]
            current_metrics = best["metrics"]
            eval_edges = best["trial_eval_edges"]
            stalled_passes = 0
            accepted_swaps += 1
            additions_since_rebuild += 1
            missing_edge_set = set(current_missing_edges())

            for candidate in refreshed_candidates:
                edge = candidate["add_edge"]
                if edge == best["add_edge"]:
                    continue
                if edge not in missing_edge_set:
                    continue
                push_entry(edge, candidate["score"])

            returned_edge = self._canon_edge(*best["remove_edge"])
            if returned_edge in missing_edge_set:
                returned_candidate = self._refresh_swap_candidate(
                    G,
                    H,
                    returned_edge,
                    eval_edges,
                    current_metrics,
                    weight_key,
                )
                if returned_candidate is not None:
                    push_entry(returned_edge, returned_candidate["score"])

            if self.verbose:
                print(
                    f"[JointDilConSwapLazy] pass={accepted_swaps} "
                    f"add={best['add_edge']} remove={best['remove_edge']} "
                    f"improvement={best['score']:.6f} "
                    f"objective={current_metrics['objective']:.6f} "
                    f"sampled_missing={len(eval_edges)}"
                )

        return H

    # ------------------------------------------------------------------
    # core loop
    # ------------------------------------------------------------------
    def _grow(self, G, base_support, delta):
        if self.growth_mode == "exact":
            return self._grow_exact(G, base_support, delta)
        if self.growth_mode == "lazy_topk":
            return self._grow_lazy_topk(G, base_support, delta)
        raise ValueError(f"Unknown growth_mode: {self.growth_mode!r}")

    def _grow_exact(self, G, base_support, delta):
        weight_key = "weight" if nx.is_weighted(G) else None
        is_weighted = nx.is_weighted(G)
        eps = self._eps

        H = base_support.copy()
        H_edge_set = {tuple(sorted(e)) for e in H.edges()}
        I = [tuple(sorted(e)) for e in G.edges() if tuple(sorted(e)) not in H_edge_set]

        target_edges = math.ceil(delta * G.number_of_edges() - 1e-12)

        while H.number_of_edges() < target_edges and I:
            I_sampled = self._sample_edges(H, I)
            if not I_sampled:
                break
            edge_con = {tuple(sorted(e)): 0 for e in H.edges()}
            node_con = {v: 0 for v in H.nodes()}
            dil = {}
            eConPath = {}
            vConPath = {}
            path_edges_map = {}
            path_nodes_map = {}
            disconnected = []

            for (u, v) in I_sampled:
                try:
                    path = nx.shortest_path(H, u, v, weight=weight_key)
                    if is_weighted:
                        d = 0.0
                        for i in range(len(path) - 1):
                            d += self._edge_weight(H, path[i], path[i + 1])
                    else:
                        d = len(path) - 1
                except nx.NetworkXNoPath:
                    disconnected.append((u, v))
                    dil[(u, v)] = float("inf")
                    path_edges_map[(u, v)] = []
                    continue

                w = max(self._edge_weight(G, u, v), eps)
                dil[(u, v)] = d / w
                pe = [tuple(sorted((path[i], path[i + 1])))
                      for i in range(len(path) - 1)]
                pn = list(path[1:-1])
                path_edges_map[(u, v)] = pe
                path_nodes_map[(u, v)] = pn
                for ek in pe:
                    edge_con[ek] = edge_con.get(ek, 0) + 1
                for node in pn:
                    node_con[node] = node_con.get(node, 0) + 1

            for e in I_sampled:
                eConPath[e] = self._norm_congestion(
                    [edge_con.get(ek, 0) for ek in path_edges_map[e]],
                    self.edge_norm_p,
                )
                vConPath[e] = self._norm_congestion(
                    [node_con.get(node, 0) for node in path_nodes_map.get(e, [])],
                    self.node_norm_q,
                )

            remaining_budget = target_edges - H.number_of_edges()

            if disconnected:
                chosen_edges = disconnected[: max(1, min(self.batch_size, remaining_budget, len(disconnected)))]
                for chosen in chosen_edges:
                    H.add_edge(*chosen, **(G.get_edge_data(*chosen) or {}))
                    I.remove(chosen)
                if self.verbose:
                    print(
                        f"[JointDilCon] added {len(chosen_edges)} disconnected edges "
                        f"edges={H.number_of_edges()}/{target_edges}"
                    )
                continue

            finite_dils = [v for v in dil.values() if not math.isinf(v)]
            D_max = max(finite_dils) if finite_dils else eps
            E_max = max(eConPath.values()) if eConPath else eps
            V_max = max(vConPath.values()) if vConPath else eps

            scores = {}
            for e in I_sampled:
                if math.isinf(dil[e]):
                    scores[e] = float("inf")
                    continue
                a_term = ((dil[e] + eps) / (D_max + eps)) ** self.alpha
                e_term = ((eConPath[e] + eps) / (E_max + eps)) ** self.edge_beta
                v_term = ((vConPath[e] + eps) / (V_max + eps)) ** self.node_beta
                scores[e] = a_term * e_term * v_term

            chosen_edges = self._choose_edges_to_add(I_sampled, scores, remaining_budget)
            for chosen in chosen_edges:
                H.add_edge(*chosen, **(G.get_edge_data(*chosen) or {}))
                I.remove(chosen)

            if self.verbose:
                lead = chosen_edges[0]
                print(
                    f"[JointDilCon] added {len(chosen_edges)} edges "
                    f"lead_edge={lead} "
                    f"dil={dil[lead]:.4f} "
                    f"eConPath={eConPath[lead]:.4f} "
                    f"vConPath={vConPath[lead]:.4f} "
                    f"D_max={D_max:.4f} E_max={E_max:.4f} V_max={V_max:.4f} "
                    f"edges={H.number_of_edges()}/{target_edges} "
                    f"sampled={len(I_sampled)}/{len(I) + 1}"
                )

        return H

    def _grow_lazy_topk(self, G, base_support, delta):
        weight_key = "weight" if nx.is_weighted(G) else None
        is_weighted = nx.is_weighted(G)

        H = base_support.copy()
        H_edge_set = {self._canon_edge(u, v) for u, v in H.edges()}
        remaining_edges = {
            self._canon_edge(u, v)
            for u, v in G.edges()
            if self._canon_edge(u, v) not in H_edge_set
        }
        target_edges = math.ceil(delta * G.number_of_edges() - 1e-12)

        heap = []
        entry_versions = {}
        entry_counter = 0
        additions_since_rebuild = 0
        rebuild_count = 0

        def push_entry(edge, score):
            nonlocal entry_counter
            entry_counter += 1
            entry_versions[edge] = entry_counter
            heapq.heappush(heap, (-score, entry_counter, edge))

        def rebuild_heap():
            nonlocal heap, entry_versions, additions_since_rebuild, rebuild_count
            sampled_edges = self._sample_edges(H, remaining_edges)
            metrics = self._evaluate_candidate_set(
                G,
                H,
                sampled_edges,
                weight_key,
                is_weighted,
            )
            heap = []
            entry_versions = {}
            for edge in sampled_edges:
                push_entry(edge, metrics[edge]["score"])
            additions_since_rebuild = 0
            rebuild_count += 1
            if self.verbose:
                print(
                    f"[JointDilConLazy] rebuild={rebuild_count} "
                    f"sampled={len(sampled_edges)}/{len(remaining_edges)} "
                    f"edges={H.number_of_edges()}/{target_edges}"
                )

        while H.number_of_edges() < target_edges and remaining_edges:
            if not heap or (
                self.lazy_rebuild_interval > 0
                and additions_since_rebuild >= self.lazy_rebuild_interval
            ):
                rebuild_heap()
                if not heap:
                    break

            active_limit = min(self.lazy_top_k, len(remaining_edges))
            active_edges = []
            while heap and len(active_edges) < active_limit:
                _, version, edge = heapq.heappop(heap)
                if edge not in remaining_edges:
                    continue
                if entry_versions.get(edge) != version:
                    continue
                active_edges.append(edge)

            if not active_edges:
                rebuild_heap()
                if not heap:
                    break
                continue

            active_metrics = self._evaluate_candidate_set(
                G,
                H,
                active_edges,
                weight_key,
                is_weighted,
            )
            best_edge = max(active_edges, key=lambda edge: active_metrics[edge]["score"])
            best_metrics = active_metrics[best_edge]

            H.add_edge(*best_edge, **(G.get_edge_data(*best_edge) or {}))
            remaining_edges.remove(best_edge)
            additions_since_rebuild += 1

            for edge in active_edges:
                if edge == best_edge or edge not in remaining_edges:
                    continue
                push_entry(edge, active_metrics[edge]["score"])

            if self.verbose:
                dil_str = "inf" if math.isinf(best_metrics["dil"]) else f"{best_metrics['dil']:.4f}"
                print(
                    f"[JointDilConLazy] add_edge={best_edge} "
                    f"dil={dil_str} "
                    f"eConPath={best_metrics['eConPath']:.4f} "
                    f"vConPath={best_metrics['vConPath']:.4f} "
                    f"active={len(active_edges)} "
                    f"edges={H.number_of_edges()}/{target_edges}"
                )

        return H
