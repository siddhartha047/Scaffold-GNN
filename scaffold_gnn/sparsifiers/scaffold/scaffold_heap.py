from collections import defaultdict
import heapq
import math
import time

import networkx as nx

from .clustering import assign_edges_to_clusters
from .common import ScaffoldBaseSparsifier
from .parallel_utils import evaluate_groups, split_workers
from .path_backend import UnweightedPathBackend


class ScaffoldHeapSparsifier(ScaffoldBaseSparsifier):
    """SCAFFOLD-Heap: cluster-local heaps with per-cluster top-r growth."""

    def _score_from_terms(self, dilation, eConPath, vConPath):
        if math.isinf(dilation):
            return float("inf")
        return dilation * (
            1.0
            + self.edge_beta * math.log1p(eConPath)
            + self.node_beta * math.log1p(vConPath)
        )

    def _candidate_path_stats_fast(self, G, H, edge, weight_key, is_weighted, path_backend=None, paths_by_edge=None):
        if path_backend is None:
            return self._candidate_path_stats(G, H, edge, weight_key, is_weighted)

        u, v = edge
        path = (path_backend.shortest_path(u, v) if paths_by_edge is None
                else paths_by_edge.get(edge))
        if path is None:
            return {
                "edge": edge,
                "disconnected": True,
                "dil": float("inf"),
                "path_edges": [],
                "path_nodes": [],
            }

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

    def _evaluate_candidate_set(self, G, H, candidate_edges, weight_key, is_weighted, path_backend=None):
        candidate_edges = list(dict.fromkeys(candidate_edges))
        if not candidate_edges:
            return {}

        paths_by_edge = None
        if path_backend is not None:
            # One BFS per source instead of one per candidate. When few
            # clusters are active, spare workers can search sources in parallel.
            grouped_targets = defaultdict(set)
            for u, v in candidate_edges:
                grouped_targets[u].add(v)
            by_source = path_backend.paths_for_sources(
                grouped_targets, getattr(self, "_path_workers", 1)
            )
            paths_by_edge = {
                (u, v): path
                for u, paths in by_source.items()
                for v, path in paths.items()
            }

        if self.edge_beta == 0 and self.node_beta == 0:
            metrics = {}
            for edge in candidate_edges:
                stats = self._candidate_path_stats_fast(
                    G,
                    H,
                    edge,
                    weight_key,
                    is_weighted,
                    path_backend=path_backend,
                    paths_by_edge=paths_by_edge,
                )
                stats["eConPath"] = 0.0
                stats["vConPath"] = 0.0
                stats["score"] = stats["dil"]
                metrics[edge] = stats
            return metrics

        edge_con = {}
        node_con = {}
        metrics = {}
        for edge in candidate_edges:
            stats = self._candidate_path_stats_fast(
                G,
                H,
                edge,
                weight_key,
                is_weighted,
                path_backend=path_backend,
                paths_by_edge=paths_by_edge,
            )
            metrics[edge] = stats
            if stats["disconnected"]:
                continue
            for edge_key in stats["path_edges"]:
                edge_con[edge_key] = edge_con.get(edge_key, 0) + 1
            for node in stats["path_nodes"]:
                node_con[node] = node_con.get(node, 0) + 1

        for stats in metrics.values():
            stats["eConPath"] = max(
                (edge_con.get(edge_key, 0) for edge_key in stats["path_edges"]),
                default=0.0,
            )
            stats["vConPath"] = max(
                (node_con.get(node, 0) for node in stats["path_nodes"]),
                default=0.0,
            )
            stats["score"] = self._score_from_terms(
                stats["dil"],
                stats["eConPath"],
                stats["vConPath"],
            )
        return metrics

    def _grow(self, G, base_support, delta):
        start_time = time.perf_counter()
        weight_key = "weight" if nx.is_weighted(G) else None
        is_weighted = nx.is_weighted(G)

        H = base_support.copy()
        path_backend = None if is_weighted else UnweightedPathBackend.maybe_build(H)
        self._path_parallel_available = (
            path_backend is not None and path_backend.parallel_available
        )
        self._score_workers_used = 1
        H_edge_set = {self._canon_edge(u, v) for u, v in H.edges()}
        remaining_edges = {
            self._canon_edge(u, v)
            for u, v in G.edges()
            if self._canon_edge(u, v) not in H_edge_set
        }
        target_edges = math.ceil(delta * G.number_of_edges() - 1e-12)
        if H.number_of_edges() >= target_edges or not remaining_edges:
            self._record_stats(start_time, 0, 0, H.number_of_edges(), target_edges, 0)
            return H

        node_to_cluster = self._build_clusters(G)
        cluster_ids = sorted(set(node_to_cluster.values()))
        edge_cluster, _, remaining_by_cluster = assign_edges_to_clusters(
            remaining_edges,
            node_to_cluster,
        )

        heaps = {cluster_id: [] for cluster_id in cluster_ids}
        entry_versions = {}
        entry_counter = 0
        additions_since_rebuild = defaultdict(int)
        rebuild_count = defaultdict(int)
        candidate_paths = {}
        path_edge_index = defaultdict(set)
        touch_node_index = defaultdict(set)
        dirty_edges = set()
        rounds = 0
        added_total = 0

        def unindex_candidate(edge):
            old = candidate_paths.pop(edge, None)
            if old is None:
                return
            for path_edge in old["path_edges"]:
                path_edge_index[path_edge].discard(edge)
            for node in old["touch_nodes"]:
                touch_node_index[node].discard(edge)

        def index_candidate(edge, stats):
            unindex_candidate(edge)
            path_edges = set(stats.get("path_edges", []))
            touch_nodes = set(edge)
            touch_nodes.update(stats.get("path_nodes", []))
            candidate_paths[edge] = {
                "path_edges": path_edges,
                "touch_nodes": touch_nodes,
            }
            for path_edge in path_edges:
                path_edge_index[path_edge].add(edge)
            for node in touch_nodes:
                touch_node_index[node].add(edge)

        def push_entry(edge, score):
            nonlocal entry_counter
            if edge not in remaining_edges:
                return
            entry_counter += 1
            entry_versions[edge] = entry_counter
            heapq.heappush(heaps[edge_cluster[edge]], (-score, entry_counter, edge))

        def score_groups(grouped):
            active = [group for group in grouped.values() if group]
            outer, inner = split_workers(len(active), self.parallel_workers)
            # Fixed until this scoring pool joins; keep the evaluator's
            # existing signature for subclasses that supply their own scores.
            self._path_workers = inner
            if self._path_parallel_available:
                widths = sorted(
                    (min(inner, len({u for u, _ in group})) for group in active),
                    reverse=True,
                )
                self._score_workers_used = max(
                    self._score_workers_used, sum(widths[:outer])
                )

            def evaluate_group(cluster_id, group):
                return self._evaluate_candidate_set(
                    G, H, group, weight_key, is_weighted,
                    path_backend=path_backend,
                )

            return evaluate_groups(
                grouped, evaluate_group, max_workers=outer,
            )

        def refresh_edges(edges, push=True):
            grouped = defaultdict(list)
            for edge in edges:
                if edge in remaining_edges:
                    grouped[edge_cluster[edge]].append(edge)
            grouped_metrics = score_groups(grouped)
            refreshed = {}
            for metrics in grouped_metrics.values():
                for edge, stats in metrics.items():
                    index_candidate(edge, stats)
                    dirty_edges.discard(edge)
                    refreshed[edge] = stats
                    if push:
                        push_entry(edge, stats["score"])
            return refreshed

        def sample_cluster_edges(cluster_id):
            candidates = remaining_by_cluster.get(cluster_id, set()).intersection(remaining_edges)
            return self._sample_edges(H, candidates)

        def rebuild_clusters(ids):
            sampled_edges = []
            for cluster_id in ids:
                heaps[cluster_id] = []
                sampled_edges.extend(sample_cluster_edges(cluster_id))
            # All clusters read the same support. Sampling and heap insertion
            # retain cluster order, independent of task completion order.
            refresh_edges(sampled_edges, push=True)
            for cluster_id in ids:
                additions_since_rebuild[cluster_id] = 0
                rebuild_count[cluster_id] += 1

        def rebuild_cluster(cluster_id):
            rebuild_clusters([cluster_id])

        def pop_active(cluster_id):
            active = []
            heap = heaps[cluster_id]
            active_limit = max(self.lazy_top_k, self.cluster_add_per_round)
            while heap and len(active) < active_limit:
                _, version, edge = heapq.heappop(heap)
                if edge not in remaining_edges:
                    continue
                if entry_versions.get(edge) != version:
                    continue
                active.append(edge)
            return active

        def local_dirty_candidates(selected):
            impact_edges = set()
            impact_nodes = set()
            for edge, stats in selected:
                impact_edges.update(stats.get("path_edges", []))
                impact_nodes.update(edge)
                impact_nodes.update(stats.get("path_nodes", []))

            if self.local_update_radius > 0 and impact_nodes:
                expanded = set(impact_nodes)
                for node in list(impact_nodes):
                    if node not in H:
                        continue
                    if path_backend is not None:
                        expanded.update(path_backend.radius_nodes(node, self.local_update_radius))
                    else:
                        lengths = nx.single_source_shortest_path_length(
                            H,
                            node,
                            cutoff=self.local_update_radius,
                        )
                        expanded.update(lengths.keys())
                impact_nodes = expanded

            dirty = set()
            for path_edge in impact_edges:
                dirty.update(path_edge_index.get(path_edge, set()))
            for node in impact_nodes:
                dirty.update(touch_node_index.get(node, set()))
            dirty.intersection_update(remaining_edges)
            if self.dirty_limit > 0 and len(dirty) > self.dirty_limit:
                dirty = set(self._rng.sample(list(dirty), self.dirty_limit))
            return dirty

        rebuild_clusters([cid for cid in cluster_ids if remaining_by_cluster.get(cid)])

        while H.number_of_edges() < target_edges and remaining_edges:
            rounds += 1
            remaining_budget = target_edges - H.number_of_edges()
            active_by_cluster = {}
            for cluster_id in cluster_ids:
                if not remaining_by_cluster.get(cluster_id, set()).intersection(remaining_edges):
                    continue
                if not heaps[cluster_id] or (
                    self.lazy_rebuild_interval > 0
                    and additions_since_rebuild[cluster_id] >= self.lazy_rebuild_interval
                ):
                    rebuild_cluster(cluster_id)
                active_edges = pop_active(cluster_id)
                if not active_edges:
                    rebuild_cluster(cluster_id)
                    active_edges = pop_active(cluster_id)
                if active_edges:
                    active_by_cluster[cluster_id] = active_edges

            if not active_by_cluster:
                break

            grouped_metrics = score_groups(active_by_cluster)
            for cluster_id, metrics in grouped_metrics.items():
                for edge, stats in metrics.items():
                    index_candidate(edge, stats)
                    dirty_edges.discard(edge)

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
                    selected.append({
                        "cluster_id": cluster_id,
                        "edge": edge,
                        "stats": metrics[edge],
                    })
                    added_from_cluster += 1
                    if len(selected) >= remaining_budget:
                        break
                    if added_from_cluster >= self.cluster_add_per_round:
                        break
                if len(selected) >= remaining_budget:
                    break
            if not selected:
                break

            selected_edges = {proposal["edge"] for proposal in selected}
            selected_stats = []
            for proposal in selected:
                edge = proposal["edge"]
                if edge not in remaining_edges:
                    continue
                H.add_edge(*edge, **(G.get_edge_data(*edge) or {}))
                if path_backend is not None:
                    path_backend.add_edge(*edge)
                remaining_edges.remove(edge)
                remaining_by_cluster[edge_cluster[edge]].discard(edge)
                unindex_candidate(edge)
                dirty_edges.discard(edge)
                additions_since_rebuild[edge_cluster[edge]] += 1
                selected_stats.append((edge, proposal["stats"]))
            added_total += len(selected_stats)

            dirty_edges.update(local_dirty_candidates(selected_stats))
            for cluster_id, active_edges in active_by_cluster.items():
                metrics = grouped_metrics.get(cluster_id, {})
                for edge in active_edges:
                    if edge in selected_edges or edge not in remaining_edges or edge in dirty_edges:
                        continue
                    if edge in metrics:
                        push_entry(edge, metrics[edge]["score"])
            if dirty_edges:
                refresh_edges(set(dirty_edges), push=True)

        self._record_stats(
            start_time,
            len(cluster_ids),
            sum(rebuild_count.values()),
            H.number_of_edges(),
            target_edges,
            rounds,
            added_total=added_total,
        )
        print(
            f"[SCAFFOLD-Heap summary] clusters={len(cluster_ids)} method={self.cluster_method} "
            f"cache_path={self.cluster_cache_dir} heap_rebuilds={sum(rebuild_count.values())} "
            f"support_cache_hit={self.last_cluster_stats['init_support_cache_hit']} "
            f"rounds={rounds} edges_added={added_total} "
            f"final_edges={H.number_of_edges()}/{target_edges} "
            f"time={self.last_cluster_stats['sparsification_time_sec']:.3f}s"
        )
        return H

    def _record_stats(
        self,
        start_time,
        cluster_count,
        heap_rebuilds,
        final_edges,
        target_edges,
        rounds,
        added_total=0,
    ):
        self.last_cluster_stats = {
            "algorithm": "scaffold_heap",
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
            "workers_used": self._score_workers_used,
            "parallelism": "cluster_path_searches",
            "heap_rebuilds": heap_rebuilds,
            "rounds": rounds,
            "edges_added": added_total,
            "target_edges": target_edges,
            "final_edges": final_edges,
            "sparsification_time_sec": time.perf_counter() - start_time,
        }
