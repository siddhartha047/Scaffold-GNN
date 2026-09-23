from concurrent.futures import ThreadPoolExecutor
import math
import time

import networkx as nx
import numpy as np

from .clustering import assign_edges_to_clusters
from .common import ScaffoldBaseSparsifier
from .parallel_utils import evaluate_groups, split_workers
from . import tree_score as ts


class ScaffoldFastSparsifier(ScaffoldBaseSparsifier):
    """SCAFFOLD-Fast: one exact LCA tree-score pass followed by global top-k."""

    algorithm_name = "scaffold_fast"
    display_name = "SCAFFOLD-Fast"

    def __init__(
        self,
        *args,
        fast_mode="fast",
        fast_load_lambda=1.0,
        backend="networkx",
        fast_score="tree_exact",
        metis_edge_sample_size=0,
        crossing_policy="balanced_owner",
        support_bridge="auto",
        support_bridge_max_candidates=1000000,
        progress="auto",
        metis_recompute=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fast_mode = str(fast_mode)
        if self.fast_mode not in ("fast", "quality"):
            raise ValueError("fast_mode must be 'fast' or 'quality'.")
        self.fast_load_lambda = float(fast_load_lambda)
        self.backend = str(backend)
        if self.backend not in ("networkx", "tensor"):
            raise ValueError("backend must be 'networkx' or 'tensor'.")
        self.fast_score = str(fast_score)
        if self.fast_score not in (
            "exact", "tree_distance", "tree_exact", "tree_exact_loop"
        ):
            raise ValueError(
                "fast_score must be 'exact', 'tree_distance', 'tree_exact', or "
                "'tree_exact_loop'."
            )
        self.metis_edge_sample_size = max(0, int(metis_edge_sample_size))
        self.crossing_policy = str(crossing_policy)
        if self.crossing_policy not in ("balanced_owner", "drop"):
            raise ValueError("crossing_policy must be 'balanced_owner' or 'drop'.")
        self.support_bridge = str(support_bridge)
        if self.support_bridge not in ("auto", "true", "false"):
            raise ValueError("support_bridge must be one of {'auto', 'true', 'false'}.")
        self.support_bridge_max_candidates = max(0, int(support_bridge_max_candidates))
        self.progress = str(progress)
        if self.progress not in ("auto", "true", "false"):
            raise ValueError("progress must be one of {'auto', 'true', 'false'}.")
        self.metis_recompute = bool(metis_recompute)

    def delta_for_target(self, num_edges):
        delta = self.delta
        if self.target_ratio is not None:
            delta = max(0.001, min(1.0, float(self.target_ratio)))
        return delta * int(num_edges) - 1e-12

    def sparsify(self, data_or_graph):
        if self.backend == "tensor":
            from .tensor_fast import TensorScaffoldFastBackend

            backend = getattr(self, "_tensor_backend", None)
            if backend is None:
                backend = TensorScaffoldFastBackend(self)
                self._tensor_backend = backend
            return backend.sparsify(data_or_graph)
        return super().sparsify(data_or_graph)

    def _grow(self, G, base_support, delta):
        """Score every candidate once against the support forest, then top-k."""
        start_time = time.perf_counter()
        H = base_support.copy()
        target_edges = math.ceil(delta * G.number_of_edges() - 1e-12)
        remaining_budget = max(0, target_edges - H.number_of_edges())
        if remaining_budget <= 0 or H.number_of_edges() >= G.number_of_edges():
            self._record_stats(
                start_time, 1, 0, 0, H.number_of_edges(), target_edges, 0
            )
            return H
        if not nx.is_forest(H):
            raise ValueError(
                "scaffold_fast requires a forest backbone for its LCA scorer"
            )

        nodes = list(G.nodes())
        node_id = {node: index for index, node in enumerate(nodes)}
        edge_rows = list(G.edges(data=True))
        src = np.asarray([node_id[u] for u, _, _ in edge_rows], dtype=np.int64)
        dst = np.asarray([node_id[v] for _, v, _ in edge_rows], dtype=np.int64)
        weights = np.asarray(
            [float(data.get("weight", 1.0)) for _, _, data in edge_rows],
            dtype=np.float64,
        )
        support_edges = {self._canon_edge(u, v) for u, v in H.edges()}
        support_mask = np.asarray(
            [self._canon_edge(u, v) in support_edges for u, v, _ in edge_rows],
            dtype=bool,
        )
        # The single scoring pass is essentially all of Fast's runtime, and it
        # is parallel over candidates. Before this it was the one variant whose
        # advertised parallelism did not exist.
        scored = ts.scaffold_tree_scores(
            len(nodes),
            src,
            dst,
            support_mask,
            weights,
            alpha=self.alpha,
            edge_beta=self.edge_beta,
            node_beta=self.node_beta,
            edge_norm_p=self.edge_norm_p,
            node_norm_q=self.node_norm_q,
            weighted_paths=bool(getattr(self, "weighted_paths", False)),
            workers=self.parallel_workers,
        )
        candidates = np.flatnonzero(~support_mask).astype(np.int64, copy=False)
        limit = min(int(remaining_budget), int(candidates.size))
        if limit:
            order = np.lexsort((candidates, -scored["score"][candidates]))
            chosen = candidates[order[:limit]]
            H.add_edges_from(edge_rows[int(index)] for index in chosen)
        else:
            chosen = np.empty(0, dtype=np.int64)

        self._record_stats(
            start_time,
            1,
            int(candidates.size),
            int(candidates.size),
            H.number_of_edges(),
            target_edges,
            1 if candidates.size else 0,
            added_total=int(chosen.size),
        )
        self.last_cluster_stats.update({
            "selection": "topk",
            "score_scope": "complete_candidate_set",
            "mandatory_candidates": int(scored["mandatory"].sum()),
            "total_stretch": float(scored["total_stretch"]),
        })
        print(
            f"[{self.display_name} summary] selection=topk "
            f"scored={candidates.size} edges_added={chosen.size} "
            f"final_edges={H.number_of_edges()}/{target_edges} "
            f"time={self.last_cluster_stats['sparsification_time_sec']:.3f}s"
        )
        return H

    def _grow_batch(self, G, base_support, delta):
        """Original sampled per-cluster growth, used by SCAFFOLD-Batch."""
        start_time = time.perf_counter()
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
        if H.number_of_edges() >= target_edges or not remaining_edges:
            self._record_stats(start_time, 0, 0, 0, H.number_of_edges(), target_edges, 0)
            return H

        node_to_cluster = self._build_clusters(G)
        cluster_ids = sorted(set(node_to_cluster.values()))
        edge_cluster, _, remaining_by_cluster = assign_edges_to_clusters(
            remaining_edges,
            node_to_cluster,
        )

        support_load = {self._canon_edge(u, v): 0.0 for u, v in H.edges()}
        rounds = 0
        added_total = 0
        sampled_total = 0
        scored_total = 0

        def active_cluster_edges(cluster_id):
            return remaining_by_cluster.get(cluster_id, set()).intersection(remaining_edges)

        def sample_cluster_edges(cluster_id):
            candidates = active_cluster_edges(cluster_id)
            return self._sample_edges(H, candidates) if candidates else []

        # Without this the cluster pool ran NetworkX path searches, which hold
        # the GIL: the workers= knob bought nothing. The compiled BFS releases
        # it, so the pool is real parallelism -- the same fix Greedy and Heap
        # already carry.
        path_backend = self._maybe_path_backend(H, is_weighted)
        self._path_workers_used = 1

        def evaluate_group(cluster_id, group):
            return self._evaluate_candidates(
                G,
                H,
                group,
                weight_key,
                is_weighted,
                support_load,
                path_backend=path_backend,
            )

        executor = ThreadPoolExecutor(max_workers=self.parallel_workers) if self.parallel_workers > 1 else None
        try:
            while H.number_of_edges() < target_edges and remaining_edges:
                rounds += 1
                remaining_budget = target_edges - H.number_of_edges()
                grouped_samples = {}
                for cluster_id in cluster_ids:
                    sampled_edges = sample_cluster_edges(cluster_id)
                    if not sampled_edges:
                        continue
                    grouped_samples[cluster_id] = sampled_edges
                    sampled_total += len(sampled_edges)

                if not grouped_samples:
                    break

                outer, inner = split_workers(
                    len(grouped_samples), self.parallel_workers
                )
                self._path_workers = inner
                grouped_metrics = evaluate_groups(
                    grouped_samples,
                    evaluate_group,
                    max_workers=outer,
                    executor=executor,
                )

                for cluster_id, metrics in grouped_metrics.items():
                    scored_total += len(metrics)
                    if self.fast_mode == "quality":
                        self._update_support_load(support_load, metrics)

                selected = []
                for cluster_id in cluster_ids:
                    metrics = grouped_metrics.get(cluster_id, {})
                    if not metrics:
                        continue
                    ordered = sorted(
                        metrics,
                        key=lambda edge: (metrics[edge]["score"], repr(edge)),
                        reverse=True,
                    )
                    added_from_cluster = 0
                    for edge in ordered:
                        if edge not in remaining_edges:
                            continue
                        selected.append(edge)
                        added_from_cluster += 1
                        if len(selected) >= remaining_budget:
                            break
                        if added_from_cluster >= self.cluster_add_per_round:
                            break
                    if len(selected) >= remaining_budget:
                        break

                if not selected:
                    break

                added = 0
                for edge in selected:
                    if edge not in remaining_edges:
                        continue
                    H.add_edge(*edge, **(G.get_edge_data(*edge) or {}))
                    if path_backend is not None:
                        path_backend.add_edge(*edge)
                    remaining_edges.remove(edge)
                    remaining_by_cluster[edge_cluster[edge]].discard(edge)
                    support_load[self._canon_edge(*edge)] = 0.0
                    added += 1
                added_total += added
                if added == 0:
                    break
        finally:
            if executor is not None:
                executor.shutdown()

        self._record_stats(
            start_time,
            len(cluster_ids),
            sampled_total,
            scored_total,
            H.number_of_edges(),
            target_edges,
            rounds,
            added_total=added_total,
        )
        print(
            f"[{self.display_name} summary] clusters={len(cluster_ids)} "
            f"mode={self.fast_mode} method={self.cluster_method} "
            f"cache_path={self.cluster_cache_dir} sample_size={self.sample_size} "
            f"add_per_cluster={self.cluster_add_per_round} workers={self.parallel_workers} "
            f"support_cache_hit={self.last_cluster_stats['init_support_cache_hit']} "
            f"sampled={sampled_total} scored={scored_total} rounds={rounds} "
            f"edges_added={added_total} final_edges={H.number_of_edges()}/{target_edges} "
            f"time={self.last_cluster_stats['sparsification_time_sec']:.3f}s"
        )
        return H

    def _evaluate_candidates(
        self,
        G,
        H,
        candidate_edges,
        weight_key,
        is_weighted,
        _support_load,
        path_backend=None,
    ):
        """Evaluate the full SCAFFOLD objective within one sampled batch."""
        candidate_edges = list(dict.fromkeys(candidate_edges))
        if not candidate_edges:
            return {}

        paths_by_edge = self._paths_for_candidates(
            H,
            candidate_edges,
            weight_key,
            is_weighted,
            path_backend=path_backend,
            workers=getattr(self, "_path_workers", 1),
        )

        metrics = {}
        edge_con = {}
        node_con = {}
        for edge in candidate_edges:
            u, v = edge
            path = paths_by_edge.get((u, v))
            if path is None:
                metrics[edge] = {
                    "edge": edge,
                    "disconnected": True,
                    "dil": float("inf"),
                    "path_edges": [],
                    "path_nodes": [],
                    "eConPath": 0.0,
                    "vConPath": 0.0,
                    "score": float("inf"),
                }
                continue

            if is_weighted:
                dist = sum(self._edge_weight(H, path[i], path[i + 1]) for i in range(len(path) - 1))
            else:
                dist = len(path) - 1
            path_edges = [
                self._canon_edge(path[i], path[i + 1]) for i in range(len(path) - 1)
            ]
            path_nodes = list(path[1:-1])
            dilation = dist / max(self._edge_weight(G, u, v), self._eps)
            metrics[edge] = {
                "edge": edge,
                "disconnected": False,
                "dil": dilation,
                "path_edges": path_edges,
                "path_nodes": path_nodes,
            }
            for path_edge in path_edges:
                edge_con[path_edge] = edge_con.get(path_edge, 0) + 1
            for node in path_nodes:
                node_con[node] = node_con.get(node, 0) + 1

        for stats in metrics.values():
            if stats["disconnected"]:
                continue
            stats["eConPath"] = self._norm_congestion(
                [edge_con.get(path_edge, 0) for path_edge in stats["path_edges"]],
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
        e_max = max(
            (stats["eConPath"] for stats in metrics.values()),
            default=self._eps,
        )
        v_max = max(
            (stats["vConPath"] for stats in metrics.values()),
            default=self._eps,
        )
        for stats in metrics.values():
            if stats["disconnected"]:
                continue
            stats["score"] = self._score_candidate(
                stats["dil"],
                stats["eConPath"],
                stats["vConPath"],
                d_max,
                e_max,
                v_max,
            )
        return metrics

    def _update_support_load(self, support_load, metrics):
        for stats in metrics.values():
            for edge_key in stats.get("path_edges", []):
                support_load[edge_key] = support_load.get(edge_key, 0.0) + 1.0

    def _record_stats(
        self,
        start_time,
        cluster_count,
        sampled_total,
        scored_total,
        final_edges,
        target_edges,
        rounds,
        added_total=0,
    ):
        self.last_cluster_stats = {
            "algorithm": self.algorithm_name,
            "fast_mode": self.fast_mode,
            "init_support": str(self.init_support),
            "init_support_cacheable": bool(self._last_init_support_cache.get("cacheable", False)),
            "init_support_cache_hit": bool(self._last_init_support_cache.get("hit", False)),
            "init_support_time_sec": float(self._last_init_support_cache.get("time_sec", 0.0)),
            "cluster_count": cluster_count,
            "cluster_method": self.cluster_method,
            "cache_path": self.cluster_cache_dir,
            "sample_size": self.sample_size,
            "cluster_add_per_round": self.cluster_add_per_round,
            "parallel_workers": self.parallel_workers,
            "sampled_candidates": sampled_total,
            "scored_candidates": scored_total,
            "rounds": rounds,
            "edges_added": added_total,
            "target_edges": target_edges,
            "final_edges": final_edges,
            "sparsification_time_sec": time.perf_counter() - start_time,
        }
