from collections import defaultdict
import heapq
import math

import networkx as nx

from .joint_dilation_congestion import JointDilationCongestionSparsifier
from .scaffold.clustering import build_node_clusters


class ClusteredLazyJointDilationCongestionSparsifier(JointDilationCongestionSparsifier):
    """Cluster-local lazy joint sparsifier.

    This experimental variant keeps a separate lazy heap per node cluster. Each
    growth round refreshes the local top-k candidates in each cluster, accepts a
    set of non-conflicting proposals, and refreshes only cached candidates whose
    previous replacement path touches the changed local region.
    """

    def __init__(
        self,
        *args,
        cluster_count=8,
        cluster_method="bfs",
        parallel_clusters=0,
        cluster_cache_dir=None,
        local_update_radius=1,
        dirty_limit=0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cluster_count = max(1, int(cluster_count))
        self.cluster_method = str(cluster_method)
        self.parallel_clusters = max(0, int(parallel_clusters))
        self.cluster_cache_dir = cluster_cache_dir
        self.local_update_radius = max(0, int(local_update_radius))
        self.dirty_limit = max(0, int(dirty_limit))

    def _build_clusters(self, G):
        return build_node_clusters(
            G,
            method=self.cluster_method,
            cluster_count=self.cluster_count,
            seed=self.seed,
            weight_key="weight" if nx.is_weighted(G) else None,
            cluster_cache_dir=self.cluster_cache_dir,
        )

    def _grow(self, G, base_support, delta):
        return self._grow_cluster_lazy(G, base_support, delta)

    def _grow_cluster_lazy(self, G, base_support, delta):
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
            return H

        node_to_cluster = self._build_clusters(G)
        cluster_ids = sorted(set(node_to_cluster.values()))
        edge_cluster = {}
        edge_affinity = {}
        remaining_by_cluster = defaultdict(set)
        cluster_load = defaultdict(int)

        for edge in sorted(remaining_edges):
            u, v = edge
            affinity = {
                node_to_cluster.get(u, 0),
                node_to_cluster.get(v, 0),
            }
            cluster_id = min(affinity, key=lambda cid: (cluster_load[cid], cid))
            edge_cluster[edge] = cluster_id
            edge_affinity[edge] = affinity
            remaining_by_cluster[cluster_id].add(edge)
            cluster_load[cluster_id] += 1

        heaps = {cluster_id: [] for cluster_id in cluster_ids}
        entry_versions = {}
        entry_counter = 0
        additions_since_rebuild = defaultdict(int)
        rebuild_count = defaultdict(int)

        metrics_cache = {}
        path_edge_index = defaultdict(set)
        touch_node_index = defaultdict(set)
        candidate_paths = {}
        dirty_edges = set()

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
            touch_nodes = set(edge)
            touch_nodes.update(stats.get("path_nodes", []))
            path_edges = set(stats.get("path_edges", []))
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
            cluster_id = edge_cluster[edge]
            entry_counter += 1
            entry_versions[edge] = entry_counter
            heapq.heappush(heaps[cluster_id], (-score, entry_counter, edge))

        def refresh_edges(edges, push=True):
            grouped = defaultdict(list)
            for edge in edges:
                if edge in remaining_edges:
                    grouped[edge_cluster[edge]].append(edge)

            refreshed = {}
            for cluster_id, group in grouped.items():
                metrics = self._evaluate_candidate_set(
                    G,
                    H,
                    group,
                    weight_key,
                    is_weighted,
                )
                for edge, stats in metrics.items():
                    metrics_cache[edge] = stats
                    index_candidate(edge, stats)
                    dirty_edges.discard(edge)
                    refreshed[edge] = stats
                    if push:
                        push_entry(edge, stats["score"])
            return refreshed

        def sample_cluster_edges(cluster_id):
            candidates = remaining_by_cluster[cluster_id].intersection(remaining_edges)
            return self._sample_edges(H, candidates)

        def rebuild_cluster(cluster_id):
            heaps[cluster_id] = []
            sampled_edges = sample_cluster_edges(cluster_id)
            refresh_edges(sampled_edges, push=True)
            additions_since_rebuild[cluster_id] = 0
            rebuild_count[cluster_id] += 1
            if self.verbose:
                print(
                    f"[ClusterJointLazy] rebuild cluster={cluster_id} "
                    f"sampled={len(sampled_edges)}/{len(remaining_by_cluster[cluster_id])} "
                    f"edges={H.number_of_edges()}/{target_edges}"
                )

        def pop_active(cluster_id):
            active = []
            heap = heaps[cluster_id]
            while heap and len(active) < self.lazy_top_k:
                _, version, edge = heapq.heappop(heap)
                if edge not in remaining_edges:
                    continue
                if edge_cluster.get(edge) != cluster_id:
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
                expanded_nodes = set(impact_nodes)
                for node in list(impact_nodes):
                    if node not in H:
                        continue
                    lengths = nx.single_source_shortest_path_length(
                        H,
                        node,
                        cutoff=self.local_update_radius,
                    )
                    expanded_nodes.update(lengths.keys())
                impact_nodes = expanded_nodes

            dirty = set()
            for path_edge in impact_edges:
                dirty.update(path_edge_index.get(path_edge, set()))
            for node in impact_nodes:
                dirty.update(touch_node_index.get(node, set()))

            dirty.intersection_update(remaining_edges)
            if self.dirty_limit > 0 and len(dirty) > self.dirty_limit:
                dirty = set(self._rng.sample(list(dirty), self.dirty_limit))
            return dirty

        for cluster_id in cluster_ids:
            if remaining_by_cluster[cluster_id]:
                rebuild_cluster(cluster_id)

        while H.number_of_edges() < target_edges and remaining_edges:
            remaining_budget = target_edges - H.number_of_edges()
            proposals = []
            active_metrics_by_cluster = {}
            active_edges_by_cluster = {}

            for cluster_id in cluster_ids:
                if not remaining_by_cluster[cluster_id].intersection(remaining_edges):
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
                if not active_edges:
                    continue

                active_metrics = refresh_edges(active_edges, push=False)
                if not active_metrics:
                    continue
                best_edge = max(
                    active_metrics,
                    key=lambda edge: active_metrics[edge]["score"],
                )
                proposals.append(
                    {
                        "cluster_id": cluster_id,
                        "edge": best_edge,
                        "stats": active_metrics[best_edge],
                    }
                )
                active_metrics_by_cluster[cluster_id] = active_metrics
                active_edges_by_cluster[cluster_id] = active_edges

            if not proposals:
                break

            proposals.sort(key=lambda item: item["stats"]["score"], reverse=True)
            selected = []
            used_affinity = set()
            used_path_edges = set()
            used_touch_nodes = set()
            max_parallel = self.parallel_clusters or len(proposals)
            max_parallel = min(max_parallel, remaining_budget)

            for proposal in proposals:
                edge = proposal["edge"]
                stats = proposal["stats"]
                affinity = edge_affinity[edge]
                path_edges = set(stats.get("path_edges", []))
                touch_nodes = set(edge)
                touch_nodes.update(stats.get("path_nodes", []))

                if affinity.intersection(used_affinity):
                    continue
                if path_edges.intersection(used_path_edges):
                    continue
                if touch_nodes.intersection(used_touch_nodes):
                    continue

                selected.append(proposal)
                used_affinity.update(affinity)
                used_path_edges.update(path_edges)
                used_touch_nodes.update(touch_nodes)
                if len(selected) >= max_parallel:
                    break

            if not selected:
                selected = proposals[:1]

            selected_edges = {proposal["edge"] for proposal in selected}
            selected_stats = []
            for proposal in selected:
                edge = proposal["edge"]
                if edge not in remaining_edges:
                    continue
                H.add_edge(*edge, **(G.get_edge_data(*edge) or {}))
                remaining_edges.remove(edge)
                remaining_by_cluster[edge_cluster[edge]].discard(edge)
                unindex_candidate(edge)
                metrics_cache.pop(edge, None)
                dirty_edges.discard(edge)
                additions_since_rebuild[edge_cluster[edge]] += 1
                selected_stats.append((edge, proposal["stats"]))

            dirty_edges.update(local_dirty_candidates(selected_stats))
            dirty_count = len(dirty_edges)

            for cluster_id, active_edges in active_edges_by_cluster.items():
                active_metrics = active_metrics_by_cluster[cluster_id]
                for edge in active_edges:
                    if edge in selected_edges or edge not in remaining_edges:
                        continue
                    if edge in dirty_edges:
                        continue
                    push_entry(edge, active_metrics[edge]["score"])

            if dirty_edges:
                to_refresh = set(dirty_edges)
                refresh_edges(to_refresh, push=True)

            if self.verbose:
                print(
                    f"[ClusterJointLazy] added={len(selected_stats)} "
                    f"dirty={dirty_count} "
                    f"edges={H.number_of_edges()}/{target_edges}"
                )

        return H
