import math
import time

import networkx as nx
from torch_geometric.data import Data

from .common import ScaffoldBaseSparsifier


class ScaffoldGreedySparsifier(ScaffoldBaseSparsifier):
    """Greedy SCAFFOLD loop for Karate and other small graph experiments."""

    def sparsify(self, data_or_graph):
        G = self._ensure_nx(data_or_graph)

        delta = self.delta
        if self.target_ratio is not None:
            delta = max(0.001, min(1.0, float(self.target_ratio)))
            if self.verbose:
                print(
                    f"[SCAFFOLD-Greedy] Auto-tuned delta={delta:.4f} "
                    f"for target_ratio={self.target_ratio}"
                )

        target_edges = math.ceil(delta * G.number_of_edges() - 1e-12)
        base_support = self._prepare_init_support(G, target_edges)
        H = self._grow(G, base_support, delta)
        self._record_support_budget_stats()
        if self.swap_refine and self.swap_max_passes > 0:
            before_edges = {self._canon_edge(u, v) for u, v in H.edges()}
            H = self._refine_with_swaps(G, H)
            after_edges = {self._canon_edge(u, v) for u, v in H.edges()}
            changed_edges = len(before_edges - after_edges)
            self.last_cluster_stats["swap_refine_enabled"] = True
            self.last_cluster_stats["swap_changed_edges"] = changed_edges
            print(
                f"[SCAFFOLD-Greedy swap summary] changed_edges={changed_edges} "
                f"final_edges={H.number_of_edges()}"
            )
        else:
            self.last_cluster_stats["swap_refine_enabled"] = False
            self.last_cluster_stats["swap_changed_edges"] = 0

        if isinstance(data_or_graph, Data):
            return self._ensure_pyg(H, original_data=data_or_graph)
        return H

    def _grow(self, G, base_support, delta):
        start_time = time.perf_counter()
        weight_key = "weight" if nx.is_weighted(G) else None
        is_weighted = nx.is_weighted(G)

        H = base_support.copy()
        H.add_nodes_from(G.nodes(data=True))
        H_edge_set = {self._canon_edge(u, v) for u, v in H.edges()}
        remaining_edges = {
            self._canon_edge(u, v)
            for u, v in G.edges()
            if self._canon_edge(u, v) not in H_edge_set
        }
        target_edges = math.ceil(delta * G.number_of_edges() - 1e-12)
        path_backend = self._maybe_path_backend(H, is_weighted)
        self._path_workers_used = 1

        rounds = 0
        added_total = 0
        scored_total = 0
        scored_by_round = []

        while H.number_of_edges() < target_edges and remaining_edges:
            metrics = self._evaluate_all_candidates(
                G,
                H,
                remaining_edges,
                weight_key,
                is_weighted,
                path_backend=path_backend,
            )
            if not metrics:
                break

            rounds += 1
            scored_total += len(metrics)
            scored_by_round.append(len(metrics))
            best_edge = max(
                metrics,
                key=lambda edge: (metrics[edge]["score"], repr(edge)),
            )
            best_metrics = metrics[best_edge]

            H.add_edge(*best_edge, **(G.get_edge_data(*best_edge) or {}))
            if path_backend is not None:
                path_backend.add_edge(*best_edge)
            remaining_edges.remove(best_edge)
            added_total += 1

            if self.verbose:
                dil = best_metrics["dil"]
                dil_str = "inf" if math.isinf(dil) else f"{dil:.4f}"
                print(
                    f"[SCAFFOLD-Greedy] round={rounds} add_edge={best_edge} "
                    f"score={best_metrics['score']:.6f} dil={dil_str} "
                    f"eConPath={best_metrics['eConPath']:.4f} "
                    f"vConPath={best_metrics['vConPath']:.4f} "
                    f"scored={len(metrics)} "
                    f"edges={H.number_of_edges()}/{target_edges}"
                )

        self.last_cluster_stats = {
            "algorithm": "scaffold_greedy",
            "init_support": str(self.init_support),
            "init_support_cacheable": bool(self._last_init_support_cache.get("cacheable", False)),
            "init_support_cache_hit": bool(self._last_init_support_cache.get("hit", False)),
            "init_support_time_sec": float(self._last_init_support_cache.get("time_sec", 0.0)),
            "cluster_count": 0,
            "cluster_method": None,
            "sample_size": 0,
            "rounds": rounds,
            "edges_added": added_total,
            "scored_candidates": scored_total,
            "scored_candidates_by_round": scored_by_round,
            "parallel_workers": self.parallel_workers,
            "workers_used": self._path_workers_used,
            "parallelism": "candidate_source_searches",
            "target_edges": target_edges,
            "final_edges": H.number_of_edges(),
            "sparsification_time_sec": time.perf_counter() - start_time,
        }
        print(
            f"[SCAFFOLD-Greedy summary] scored={scored_total} rounds={rounds} "
            f"support_cache_hit={self.last_cluster_stats['init_support_cache_hit']} "
            f"edges_added={added_total} final_edges={H.number_of_edges()}/{target_edges} "
            f"time={self.last_cluster_stats['sparsification_time_sec']:.3f}s"
        )
        return H

    def _evaluate_all_candidates(
        self,
        G,
        H,
        candidate_edges,
        weight_key,
        is_weighted,
        path_backend=None,
    ):
        candidate_edges = sorted(dict.fromkeys(candidate_edges), key=repr)
        if not candidate_edges:
            return {}

        paths_by_edge = self._paths_for_candidates(
            H,
            candidate_edges,
            weight_key,
            is_weighted,
            path_backend=path_backend,
        )
        edge_con = {}
        node_con = {}
        metrics = {}

        for edge in candidate_edges:
            u, v = edge
            path = paths_by_edge.get(edge)
            if path is None:
                metrics[edge] = {
                    "edge": edge,
                    "disconnected": True,
                    "dil": float("inf"),
                    "path_edges": [],
                    "path_nodes": [],
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
            metrics[edge] = {
                "edge": edge,
                "disconnected": False,
                "dil": dist / max(self._edge_weight(G, u, v), self._eps),
                "path_edges": path_edges,
                "path_nodes": path_nodes,
            }
            for path_edge in path_edges:
                edge_con[path_edge] = edge_con.get(path_edge, 0) + 1
            for node in path_nodes:
                node_con[node] = node_con.get(node, 0) + 1

        for stats in metrics.values():
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

