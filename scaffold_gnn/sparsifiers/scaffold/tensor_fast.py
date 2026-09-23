"""ClusterData tensor backend shared by SCAFFOLD-Fast and SCAFFOLD-Batch.

The tensor backend uses PyG ``ClusterData`` to compute METIS node partitions,
then assigns every original edge to exactly one local scaffold job. Crossing
edges are kept by default through balanced endpoint ownership, so selected
local edges map back to a unique global edge id without duplicates.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData

from . import tree_score as ts


try:
    from numba import njit
except Exception:  # pragma: no cover - fallback for environments without numba.
    njit = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime.
    tqdm = None


_CLUSTERDATA_SAFE_GLOBALS_REGISTERED = False


def _register_clusterdata_safe_globals():
    """Allow PyTorch weights-only loading of PyG ClusterData cache files."""
    global _CLUSTERDATA_SAFE_GLOBALS_REGISTERED
    if _CLUSTERDATA_SAFE_GLOBALS_REGISTERED:
        return
    try:
        from torch.serialization import add_safe_globals
        from torch_geometric.loader.cluster import Partition

        add_safe_globals([Partition])
    except Exception:
        pass
    _CLUSTERDATA_SAFE_GLOBALS_REGISTERED = True


if njit is not None:

    @njit(cache=True, nogil=True)
    def _tree_distances_numba(src, dst, depth, root, up):
        out = np.empty(src.shape[0], dtype=np.int32)
        levels = up.shape[0]
        for i in range(src.shape[0]):
            a = int(src[i])
            b = int(dst[i])
            if root[a] != root[b]:
                out[i] = depth[a] + depth[b] + 1
                continue
            orig_a = a
            orig_b = b
            if depth[a] < depth[b]:
                tmp = a
                a = b
                b = tmp
            diff = depth[a] - depth[b]
            bit = 0
            while diff > 0:
                if diff & 1:
                    a = up[bit, a]
                diff >>= 1
                bit += 1
            if a != b:
                for lvl in range(levels - 1, -1, -1):
                    if up[lvl, a] != up[lvl, b]:
                        a = up[lvl, a]
                        b = up[lvl, b]
                a = up[0, a]
            out[i] = depth[orig_a] + depth[orig_b] - 2 * depth[a]
        return out

    @njit(cache=True, nogil=True)
    def _assign_edge_owners_numba(src, dst, node_cluster, cluster_count, drop_crossing):
        owners = np.full(src.shape[0], -1, dtype=np.int32)
        internal_counts = np.zeros(cluster_count, dtype=np.int64)
        owned_crossing_counts = np.zeros(cluster_count, dtype=np.int64)
        loads = np.zeros(cluster_count, dtype=np.int64)

        for i in range(src.shape[0]):
            cu = int(node_cluster[int(src[i])])
            cv = int(node_cluster[int(dst[i])])
            if cu == cv:
                owners[i] = cu
                internal_counts[cu] += 1
                loads[cu] += 1

        if drop_crossing:
            return owners, internal_counts, owned_crossing_counts, loads

        for i in range(src.shape[0]):
            cu = int(node_cluster[int(src[i])])
            cv = int(node_cluster[int(dst[i])])
            if cu == cv:
                continue
            if loads[cu] <= loads[cv]:
                owner = cu
            else:
                owner = cv
            owners[i] = owner
            owned_crossing_counts[owner] += 1
            loads[owner] += 1
        return owners, internal_counts, owned_crossing_counts, loads

    @njit(cache=True, nogil=True)
    def _dsu_find(parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    @njit(cache=True, nogil=True)
    def _strided_spanning_forest_mask_numba(
        num_nodes, src, dst, offset, stride, max_edges
    ):
        parent = np.arange(num_nodes, dtype=np.int64)
        rank = np.zeros(num_nodes, dtype=np.int8)
        mask = np.zeros(src.shape[0], dtype=np.bool_)
        added = 0
        m = src.shape[0]
        limit = min(max(0, int(max_edges)), max(0, int(num_nodes) - 1))
        if limit <= 0:
            return mask
        for step in range(m):
            i = (offset + step * stride) % m
            u = int(src[i])
            v = int(dst[i])
            if u == v:
                continue
            ru = _dsu_find(parent, u)
            rv = _dsu_find(parent, v)
            if ru == rv:
                continue
            if rank[ru] < rank[rv]:
                parent[ru] = rv
            elif rank[ru] > rank[rv]:
                parent[rv] = ru
            else:
                parent[rv] = ru
                rank[ru] += 1
            mask[i] = True
            added += 1
            if added >= limit:
                break
        return mask

    @njit(cache=True, nogil=True)
    def _component_count_numba(num_nodes, src, dst):
        parent = np.arange(num_nodes, dtype=np.int64)
        rank = np.zeros(num_nodes, dtype=np.int8)
        components = int(num_nodes)
        for i in range(src.shape[0]):
            u = int(src[i])
            v = int(dst[i])
            if u == v:
                continue
            ru = _dsu_find(parent, u)
            rv = _dsu_find(parent, v)
            if ru == rv:
                continue
            if rank[ru] < rank[rv]:
                parent[ru] = rv
            elif rank[ru] > rank[rv]:
                parent[rv] = ru
            else:
                parent[rv] = ru
                rank[ru] += 1
            components -= 1
        return components

    @njit(cache=True, nogil=True)
    def _build_tree_index_numba(num_nodes, tree_src, tree_dst, tree_edge_idx, levels):
        m = tree_src.shape[0]
        # Compute CSR adjacency (each edge appears twice, once per endpoint).
        deg = np.zeros(num_nodes + 1, dtype=np.int64)
        for i in range(m):
            deg[int(tree_src[i]) + 1] += 1
            deg[int(tree_dst[i]) + 1] += 1
        for i in range(1, num_nodes + 1):
            deg[i] += deg[i - 1]
        rowptr = deg
        nbr = np.empty(2 * m, dtype=np.int64)
        nbr_edge = np.empty(2 * m, dtype=np.int64)
        cursor = np.empty(num_nodes, dtype=np.int64)
        for i in range(num_nodes):
            cursor[i] = rowptr[i]
        for i in range(m):
            u = int(tree_src[i])
            v = int(tree_dst[i])
            e = int(tree_edge_idx[i])
            pos_u = cursor[u]
            nbr[pos_u] = v
            nbr_edge[pos_u] = e
            cursor[u] = pos_u + 1
            pos_v = cursor[v]
            nbr[pos_v] = u
            nbr_edge[pos_v] = e
            cursor[v] = pos_v + 1

        parent = np.arange(num_nodes)
        parent_edge = np.full(num_nodes, -1, dtype=np.int64)
        depth = np.zeros(num_nodes, dtype=np.int32)
        root = np.arange(num_nodes)
        seen = np.zeros(num_nodes, dtype=np.bool_)

        stack = np.empty(num_nodes, dtype=np.int64)

        for start in range(num_nodes):
            if seen[start]:
                continue
            seen[start] = True
            root[start] = start
            parent[start] = start
            depth[start] = 0
            top = 0
            stack[top] = start
            top += 1
            while top > 0:
                top -= 1
                node = int(stack[top])
                row_end = int(rowptr[node + 1])
                for pos in range(int(rowptr[node]), row_end):
                    n = int(nbr[pos])
                    if seen[n]:
                        continue
                    seen[n] = True
                    parent[n] = node
                    parent_edge[n] = int(nbr_edge[pos])
                    depth[n] = depth[node] + 1
                    root[n] = start
                    stack[top] = n
                    top += 1

        up = np.empty((levels, num_nodes), dtype=np.int64)
        for i in range(num_nodes):
            up[0, i] = parent[i]
        for lvl in range(1, levels):
            prev = up[lvl - 1]
            for i in range(num_nodes):
                up[lvl, i] = prev[int(prev[i])]
        return depth, root, up, parent, parent_edge

    @njit(cache=True, nogil=True)
    def _bridge_support_edges_numba(
        num_nodes,
        src,
        dst,
        support_edge_ids,
        candidate_edge_ids,
        offset,
        stride,
        target_components,
        max_edges,
    ):
        parent = np.arange(num_nodes, dtype=np.int64)
        rank = np.zeros(num_nodes, dtype=np.int8)
        components = int(num_nodes)

        for i in range(support_edge_ids.shape[0]):
            edge_id = int(support_edge_ids[i])
            u = int(src[edge_id])
            v = int(dst[edge_id])
            if u == v:
                continue
            ru = _dsu_find(parent, u)
            rv = _dsu_find(parent, v)
            if ru == rv:
                continue
            if rank[ru] < rank[rv]:
                parent[ru] = rv
            elif rank[ru] > rank[rv]:
                parent[rv] = ru
            else:
                parent[rv] = ru
                rank[ru] += 1
            components -= 1
        initial_components = components

        bridges = np.empty(candidate_edge_ids.shape[0], dtype=np.int64)
        added = 0
        m = int(candidate_edge_ids.shape[0])
        edge_limit = max(0, int(max_edges))
        if m == 0 or edge_limit <= 0:
            return bridges[:0], initial_components, components

        for step in range(m):
            i = (int(offset) + step * int(stride)) % m
            edge_id = int(candidate_edge_ids[i])
            u = int(src[edge_id])
            v = int(dst[edge_id])
            if u == v:
                continue
            ru = _dsu_find(parent, u)
            rv = _dsu_find(parent, v)
            if ru == rv:
                continue
            if rank[ru] < rank[rv]:
                parent[ru] = rv
            elif rank[ru] > rank[rv]:
                parent[rv] = ru
            else:
                parent[rv] = ru
                rank[ru] += 1
            bridges[added] = edge_id
            added += 1
            components -= 1
            if added >= edge_limit or components <= int(target_components):
                break
        return bridges[:added], initial_components, components


class TensorScaffoldFastBackend:
    def __init__(self, owner):
        self.owner = owner
        self._label = f"{getattr(owner, 'display_name', 'SCAFFOLD-Fast')} tensor"
        self._algorithm = (
            f"{getattr(owner, 'algorithm_name', 'scaffold_fast')}"
            "_tensor_clusterdata_owner"
        )
        if owner.fast_score not in ("tree_distance", "tree_exact", "tree_exact_loop"):
            raise ValueError(
                f"{self._label} supports scaffold_fast_score in "
                "{'tree_distance', 'tree_exact', 'tree_exact_loop'}."
            )
        self._plan_cache = None

    def sparsify(self, data):
        if not isinstance(data, Data):
            raise TypeError(f"{self._label} expects a torch_geometric.data.Data input.")
        requested_cluster_count = max(1, int(self.owner.cluster_count))
        if requested_cluster_count > 1 and self.owner.cluster_method != "metis":
            raise ValueError(f"{self._label} uses PyG ClusterData and requires cluster_method='metis'.")

        start_time = time.perf_counter()
        stage_times = {}

        def mark_stage(name, stage_start):
            now = time.perf_counter()
            stage_times[name] = now - stage_start
            return now

        stage_start = time.perf_counter()
        num_nodes = int(data.num_nodes)
        src, dst = self._canonical_undirected_edges(data)
        self._current_support_scores = self._support_edge_scores(data, src, dst)
        num_edges = int(src.shape[0])
        target_edges = max(1, min(num_edges, int(math.ceil(self.owner.delta_for_target(num_edges)))))
        stage_start = mark_stage("canonical_edges", stage_start)

        crossing_policy = str(getattr(self.owner, "crossing_policy", "balanced_owner"))
        max_workers = max(1, int(self.owner.parallel_workers))

        plan_key = (
            int(num_nodes),
            int(num_edges),
            int(requested_cluster_count),
            str(crossing_policy),
            str(getattr(self.owner, "support_weight_method", "uniform")),
            bool(getattr(self.owner, "metis_recompute", False)),
        )
        cached_plan = self._plan_cache if isinstance(self._plan_cache, dict) else None
        if cached_plan is not None and cached_plan.get("key") == plan_key:
            reused = self._reuse_cached_plan(
                data,
                cached_plan,
                src,
                dst,
                num_nodes,
                num_edges,
                target_edges,
                crossing_policy,
                stage_times,
                start_time,
            )
            if reused is not None:
                return reused

        if requested_cluster_count <= 1 or num_nodes <= 1:
            cluster_count = 1
            stage_times["clusterdata"] = 0.0
            stage_times["node_partition_materialization"] = 0.0
            node_cluster = np.zeros(int(num_nodes), dtype=np.int32)
            partition_node_counts = np.asarray([int(num_nodes)], dtype=np.int64)
            owners = np.zeros(int(num_edges), dtype=np.int32)
            internal_counts = np.asarray([int(num_edges)], dtype=np.int64)
            owned_crossing_counts = np.asarray([0], dtype=np.int64)
            owner_loads = np.asarray([int(num_edges)], dtype=np.int64)
            owner_cache_path = None
            owner_cache_hit = False
            stage_times["edge_owner_assignment"] = 0.0
            stage_times["edge_owner_cache_load"] = 0.0
            print(
                f"[{self._label}] single cluster ready: "
                f"clusters=1 time={stage_times['clusterdata']:.3f}s",
                flush=True,
            )
        else:
            cluster_data = self._build_cluster_data(num_nodes, src, dst)
            cluster_count = len(cluster_data)
            stage_start = mark_stage("clusterdata", stage_start)
            print(
                f"[{self._label}] ClusterData ready: clusters={cluster_count} "
                f"time={stage_times['clusterdata']:.3f}s",
                flush=True,
            )

            node_cluster, partition_node_counts = self._node_clusters_from_cluster_data(
                cluster_data,
                num_nodes,
                cluster_count,
            )
            stage_start = mark_stage("node_partition_materialization", stage_start)

            owner_cache_path = self._edge_owner_cache_path(num_nodes, num_edges, cluster_count, crossing_policy)
            owner_cache_hit = False
            cache_start = time.perf_counter()
            if bool(getattr(self.owner, "metis_recompute", False)):
                self._remove_cache_file(owner_cache_path, "edge-owner")
                cached_owners = None
            else:
                cached_owners = self._load_edge_owner_cache(owner_cache_path, num_edges, cluster_count)
            if cached_owners is not None:
                owners, internal_counts, owned_crossing_counts, owner_loads = cached_owners
                owner_cache_hit = True
                stage_times["edge_owner_cache_load"] = time.perf_counter() - cache_start
                stage_times["edge_owner_assignment"] = 0.0
                stage_start = time.perf_counter()
            else:
                assign_start = time.perf_counter()
                owners, internal_counts, owned_crossing_counts, owner_loads = self._assign_edge_owners(
                    src,
                    dst,
                    node_cluster,
                    cluster_count,
                    crossing_policy,
                )
                stage_times["edge_owner_assignment"] = time.perf_counter() - assign_start
                write_start = time.perf_counter()
                self._write_edge_owner_cache(
                    owner_cache_path,
                    owners,
                    internal_counts,
                    owned_crossing_counts,
                    owner_loads,
                )
                stage_times["edge_owner_cache_write"] = time.perf_counter() - write_start
                stage_start = time.perf_counter()
        workers = min(max_workers, max(1, cluster_count))
        internal_edges = int(internal_counts.sum())
        crossing_edges = int(num_edges - internal_edges)
        owned_crossing_edges = int(owned_crossing_counts.sum())
        dropped_crossing_edges = int(crossing_edges - owned_crossing_edges)
        available_edges = int((owners >= 0).sum())
        support_cache_key = self._tensor_support_cache_key(
            num_nodes,
            num_edges,
            cluster_count,
            crossing_policy,
            target_edges,
            available_edges,
            owner_cache_path,
        )
        support_cache_hit = False
        support_budget_stats = None

        jobs = self._materialize_owned_jobs(
            src,
            dst,
            owners,
            node_cluster,
            partition_node_counts,
            internal_counts,
            owned_crossing_counts,
            workers,
        )
        self._assign_support_budgets(jobs, target_edges)
        stage_start = mark_stage("owned_edge_materialization", stage_start)
        print(
            f"[{self._label}] edge owners ready: policy={crossing_policy} "
            f"internal={internal_edges} crossing={crossing_edges} "
            f"owned_crossing={owned_crossing_edges} dropped_crossing={dropped_crossing_edges} "
            f"workers={workers} time={stage_times['owned_edge_materialization']:.3f}s",
            flush=True,
        )

        if available_edges <= target_edges:
            output_start = time.perf_counter()
            selected_parts = [job["edge_ids"] for job in jobs if job["edge_ids"].size]
            if selected_parts:
                selected_edge_ids = np.unique(np.concatenate(selected_parts).astype(np.int64, copy=False))
            else:
                selected_edge_ids = np.empty(0, dtype=np.int64)
            out = self._build_output_data(data, num_nodes, src, dst, selected_edge_ids)
            stage_times["target_allocation"] = 0.0
            stage_times["local_support"] = 0.0
            stage_times["local_growth"] = 0.0
            stage_times["output_materialization"] = time.perf_counter() - output_start
            target_feasible = bool(available_edges == target_edges)
            total_time = time.perf_counter() - start_time
            edge_balance = self._edge_owner_balance(owner_loads)
            partition_rows = self._partition_rows(jobs, None)
            self.owner.last_cluster_stats = {
                "algorithm": self._algorithm,
                "fast_mode": self.owner.fast_mode,
                "fast_score": self.owner.fast_score,
                "score_scope": (
                    "sampled_batch"
                    if getattr(self.owner, "algorithm_name", "") == "scaffold_batch"
                    else "complete_candidate_set"
                ),
                "crossing_policy": crossing_policy,
                "cluster_count": int(cluster_count),
                "cluster_method": self.owner.cluster_method,
                "cache_path": self.owner.cluster_cache_dir,
                "edge_owner_cache_path": str(owner_cache_path) if owner_cache_path else None,
                "edge_owner_cache_hit": owner_cache_hit,
                "metis_recompute": bool(getattr(self.owner, "metis_recompute", False)),
                "sample_size": self.owner.sample_size,
                "cluster_add_per_round": self.owner.cluster_add_per_round,
                "parallel_workers": self.owner.parallel_workers,
                "workers_used": int(workers),
                "sampled_candidates": 0,
                "rounds": 0,
                "total_rounds": 0,
                "edges_added": 0,
                "local_support_edges": 0,
                "support_budget_mode": str(
                    getattr(self.owner, "support_budget_mode", "early_stop")
                ),
                "support_edges_before_trim": 0,
                "support_edges_trimmed": 0,
                "support_edges_after_trim": 0,
                "support_bridge_enabled": False,
                "support_bridge_mode": str(getattr(self.owner, "support_bridge", "auto")),
                "support_bridge_skip_reason": "fast_exit",
                "bridge_support_edges": 0,
                "support_cache_hit": False,
                "original_components": -1,
                "support_components_before_bridge": -1,
                "support_components_after_bridge": -1,
                "internal_edges": internal_edges,
                "crossing_edges": crossing_edges,
                "owned_crossing_edges": owned_crossing_edges,
                "dropped_crossing_edges": dropped_crossing_edges,
                "available_edges": available_edges,
                "target_feasible": target_feasible,
                "support_floor_edges": 0,
                "target_edges": target_edges,
                "final_edges": int(selected_edge_ids.shape[0]),
                "edge_owner_load_min": edge_balance["min"],
                "edge_owner_load_median": edge_balance["median"],
                "edge_owner_load_max": edge_balance["max"],
                "edge_owner_balance_ratio": edge_balance["balance_ratio"],
                "fast_exit_reason": (
                    "target_equals_available_edges"
                    if target_feasible
                    else "target_exceeds_available_edges"
                ),
                "clusterdata_time_sec": stage_times.get("clusterdata", 0.0),
                "clusterloader_materialization_time_sec": stage_times.get("owned_edge_materialization", 0.0),
                "local_support_time_sec": stage_times.get("local_support", 0.0),
                "local_growth_time_sec": stage_times.get("local_growth", 0.0),
                "output_materialization_time_sec": stage_times.get("output_materialization", 0.0),
                "sparsification_time_sec": total_time,
                "timing_sec": stage_times,
                "partition_edge_distribution": partition_rows,
                "local_profile": {
                    "workers_used": int(workers),
                    "edge_count_min": int(min((job["edge_count"] for job in jobs), default=0)),
                    "edge_count_max": int(max((job["edge_count"] for job in jobs), default=0)),
                    "target_count_min": 0,
                    "target_count_max": 0,
                },
            }
            print(
                f"[{self._label} ClusterData summary] clusters={cluster_count} "
                f"mode={self.owner.fast_mode} score={self.owner.fast_score} method={self.owner.cluster_method} "
                f"crossing_policy={crossing_policy} "
                f"cache_path={self.owner.cluster_cache_dir} sample_size={self.owner.sample_size} "
                f"add_per_cluster={self.owner.cluster_add_per_round} workers={self.owner.parallel_workers} "
                f"workers_used={workers} internal={internal_edges} crossing={crossing_edges} "
                f"owned_crossing={owned_crossing_edges} dropped_crossing={dropped_crossing_edges} "
                f"support=0 sampled=0 max_rounds=0 edges_added=0 "
                f"final_edges={selected_edge_ids.shape[0]}/{target_edges} "
                f"target_feasible={target_feasible} fast_exit="
                f"{self.owner.last_cluster_stats['fast_exit_reason']} time={total_time:.3f}s"
            )
            return out

        bridge_stats = self._apply_tensor_support_cache(jobs, support_cache_key)
        if bridge_stats is not None:
            support_cache_hit = True
            cached_support_edges = int(sum(job["support_count"] for job in jobs))
            support_budget_stats = self._support_budget_stats(
                cached_support_edges,
                cached_support_edges,
            )
            stage_times["local_support"] = 0.0
            stage_times["support_bridge"] = 0.0
            print(
                f"[{self._label}] local support cache hit: "
                f"support_edges={sum(job['support_count'] for job in jobs)}",
                flush=True,
            )
        else:
            support_start = time.perf_counter()
            full_support_cache_hit = self._apply_tensor_full_support_cache(
                jobs,
                support_cache_key,
            )
            if full_support_cache_hit:
                support_cache_hit = True
            else:
                jobs = self._build_local_supports(jobs, workers)
                self._store_tensor_full_support_cache(support_cache_key, jobs)
            support_budget_stats = self._trim_local_supports_to_budget(
                jobs,
                min(target_edges, available_edges),
            )
            stage_times["local_support"] = time.perf_counter() - support_start
            pre_bridge_support_floor = int(sum(job["support_count"] for job in jobs))
            pre_bridge_desired_final_edges = min(target_edges, available_edges)
            bridge_budget = max(
                0, pre_bridge_desired_final_edges - pre_bridge_support_floor
            )
            bridge_required = bridge_budget > 0
            bridge_start = time.perf_counter()
            bridge_stats = self._add_crossing_support_bridges(
                jobs,
                src,
                dst,
                owners,
                node_cluster,
                crossing_policy,
                num_nodes,
                bridge_required,
                owned_crossing_edges,
                max_edges=bridge_budget,
            )
            stage_times["support_bridge"] = time.perf_counter() - bridge_start
            self._store_tensor_support_cache(support_cache_key, jobs, bridge_stats)
        if bridge_stats["support_bridge_enabled"]:
            bridge_component_msg = (
                f"components={bridge_stats['support_components_before_bridge']}->"
                f"{bridge_stats['support_components_after_bridge']} "
                f"target_components={bridge_stats['original_components']}"
            )
        else:
            bridge_component_msg = f"components=bridge_disabled:{bridge_stats['support_bridge_skip_reason']}"
        print(
            f"[{self._label}] local support built: support_edges="
            f"{sum(job['support_count'] for job in jobs)} "
            f"budget_mode={support_budget_stats['support_budget_mode']} "
            f"before_trim={support_budget_stats['support_edges_before_trim']} "
            f"trimmed={support_budget_stats['support_edges_trimmed']} "
            f"bridge_edges={bridge_stats['bridge_support_edges']} "
            f"{bridge_component_msg} "
            f"time={stage_times['local_support'] + stage_times['support_bridge']:.3f}s",
            flush=True,
        )

        alloc_start = time.perf_counter()
        support_floor = int(sum(job["support_count"] for job in jobs))
        desired_final_edges = min(target_edges, available_edges)
        if support_floor > desired_final_edges:
            raise RuntimeError(
                "Budget-aware Scaffold support exceeded the requested target: "
                f"support={support_floor} target={desired_final_edges}"
            )
        target_feasible = bool(available_edges >= target_edges)
        extra_budget = max(0, int(desired_final_edges - support_floor))
        capacities = np.asarray(
            [job["edge_count"] - job["support_count"] for job in jobs],
            dtype=np.int64,
        )
        extra_targets = self._allocate_budget(capacities, extra_budget)
        for job, extra in zip(jobs, extra_targets.tolist()):
            job["target_edges"] = int(job["support_count"] + extra)
        stage_times["target_allocation"] = time.perf_counter() - alloc_start

        growth_start = time.perf_counter()
        results = self._grow_local_jobs(jobs, workers)
        stage_times["local_growth"] = time.perf_counter() - growth_start
        print(
            f"[{self._label}] local growth complete: "
            f"time={stage_times['local_growth']:.3f}s",
            flush=True,
        )

        output_start = time.perf_counter()
        selected_parts = [result["selected_edge_ids"] for result in results if result["selected_edge_ids"].size]
        if selected_parts:
            selected_edge_ids = np.unique(np.concatenate(selected_parts).astype(np.int64, copy=False))
        else:
            selected_edge_ids = np.empty(0, dtype=np.int64)

        out = self._build_output_data(data, num_nodes, src, dst, selected_edge_ids)
        stage_times["output_materialization"] = time.perf_counter() - output_start

        sampled_total = int(sum(result["sampled_candidates"] for result in results))
        added_total = int(sum(result["edges_added"] for result in results))
        swap_total = int(sum(result.get("swaps_accepted", 0) for result in results))
        swap_candidates_total = int(sum(result.get("swap_candidates", 0) for result in results))
        swap_passes_total = int(sum(result.get("swap_passes", 0) for result in results))
        max_rounds = int(max((result["rounds"] for result in results), default=0))
        total_rounds = int(sum(result["rounds"] for result in results))
        support_total = int(sum(result["support_edges"] for result in results))
        local_profile = self._summarize_local_profiles(results)
        local_profile["workers_used"] = int(workers)
        edge_balance = self._edge_owner_balance(owner_loads)
        partition_rows = self._partition_rows(jobs, results)

        total_time = time.perf_counter() - start_time
        self.owner.last_cluster_stats = {
            "algorithm": self._algorithm,
            "fast_mode": self.owner.fast_mode,
            "fast_score": self.owner.fast_score,
            "score_scope": (
                "sampled_batch"
                if getattr(self.owner, "algorithm_name", "") == "scaffold_batch"
                else "complete_candidate_set"
            ),
            "crossing_policy": crossing_policy,
            "cluster_count": int(cluster_count),
            "cluster_method": self.owner.cluster_method,
            "cache_path": self.owner.cluster_cache_dir,
            "edge_owner_cache_path": str(owner_cache_path) if owner_cache_path else None,
            "edge_owner_cache_hit": owner_cache_hit,
            "metis_recompute": bool(getattr(self.owner, "metis_recompute", False)),
            "sample_size": self.owner.sample_size,
            "cluster_add_per_round": self.owner.cluster_add_per_round,
            "parallel_workers": self.owner.parallel_workers,
            "workers_used": int(workers),
            "sampled_candidates": sampled_total,
            "rounds": max_rounds,
            "total_rounds": total_rounds,
            "edges_added": added_total,
            "swaps_accepted": swap_total,
            "swap_candidates": swap_candidates_total,
            "swap_passes": swap_passes_total,
            "local_support_edges": support_total,
            **support_budget_stats,
            "support_bridge_enabled": bool(bridge_stats["support_bridge_enabled"]),
            "support_bridge_mode": str(bridge_stats["support_bridge_mode"]),
            "support_bridge_skip_reason": str(bridge_stats["support_bridge_skip_reason"]),
            "bridge_support_edges": int(bridge_stats["bridge_support_edges"]),
            "support_cache_hit": bool(support_cache_hit),
            "original_components": int(bridge_stats["original_components"]),
            "support_components_before_bridge": int(bridge_stats["support_components_before_bridge"]),
            "support_components_after_bridge": int(bridge_stats["support_components_after_bridge"]),
            "internal_edges": internal_edges,
            "crossing_edges": crossing_edges,
            "owned_crossing_edges": owned_crossing_edges,
            "dropped_crossing_edges": dropped_crossing_edges,
            "available_edges": available_edges,
            "target_feasible": target_feasible,
            "support_floor_edges": support_floor,
            "target_edges": target_edges,
            "final_edges": int(selected_edge_ids.shape[0]),
            "edge_owner_load_min": edge_balance["min"],
            "edge_owner_load_median": edge_balance["median"],
            "edge_owner_load_max": edge_balance["max"],
            "edge_owner_balance_ratio": edge_balance["balance_ratio"],
            "clusterdata_time_sec": stage_times.get("clusterdata", 0.0),
            "clusterloader_materialization_time_sec": stage_times.get("owned_edge_materialization", 0.0),
            "local_support_time_sec": stage_times.get("local_support", 0.0),
            "local_growth_time_sec": stage_times.get("local_growth", 0.0),
            "output_materialization_time_sec": stage_times.get("output_materialization", 0.0),
            "sparsification_time_sec": total_time,
            "timing_sec": stage_times,
            "partition_edge_distribution": partition_rows,
            "local_profile": local_profile,
        }
        print(
            f"[{self._label} ClusterData summary] clusters={cluster_count} "
            f"mode={self.owner.fast_mode} score={self.owner.fast_score} method={self.owner.cluster_method} "
            f"crossing_policy={crossing_policy} "
            f"cache_path={self.owner.cluster_cache_dir} sample_size={self.owner.sample_size} "
            f"add_per_cluster={self.owner.cluster_add_per_round} workers={self.owner.parallel_workers} "
            f"workers_used={workers} internal={internal_edges} crossing={crossing_edges} "
            f"owned_crossing={owned_crossing_edges} dropped_crossing={dropped_crossing_edges} "
            f"support={support_total} bridge_mode={bridge_stats['support_bridge_mode']} "
            f"budget_mode={support_budget_stats['support_budget_mode']} "
            f"before_trim={support_budget_stats['support_edges_before_trim']} "
            f"trimmed={support_budget_stats['support_edges_trimmed']} "
            f"bridge_support={bridge_stats['bridge_support_edges']} "
            f"support_cache_hit={support_cache_hit} "
            f"sampled={sampled_total} max_rounds={max_rounds} "
            f"edges_added={added_total} swaps={swap_total} "
            f"final_edges={selected_edge_ids.shape[0]}/{target_edges} "
            f"target_feasible={target_feasible} time={total_time:.3f}s"
        )
        self._store_plan_cache(
            plan_key,
            src,
            dst,
            node_cluster,
            partition_node_counts,
            owners,
            internal_counts,
            owned_crossing_counts,
            owner_loads,
            jobs,
            support_cache_key,
            bridge_stats,
            support_cache_hit,
            owner_cache_hit,
            owner_cache_path,
            cluster_count,
            workers,
        )
        return out

    def _store_plan_cache(
        self,
        plan_key,
        src,
        dst,
        node_cluster,
        partition_node_counts,
        owners,
        internal_counts,
        owned_crossing_counts,
        owner_loads,
        jobs,
        support_cache_key,
        bridge_stats,
        support_cache_hit,
        owner_cache_hit,
        owner_cache_path,
        cluster_count,
        workers,
    ):
        """Persist plan-invariant work so subsequent sparsify() calls skip it.

        Only the graph-structural pieces are cached — anything that depends on
        the RNG seed (support masks come from ``_apply_tensor_support_cache``;
        growth results) is recomputed each call.
        """
        job_structs = []
        for job in jobs:
            job_structs.append(
                {
                    "cluster_id": int(job["cluster_id"]),
                    "num_nodes": int(job["num_nodes"]),
                    "partition_node_count": int(job.get("partition_node_count", 0)),
                    "internal_edges": int(job.get("internal_edges", 0)),
                    "owned_crossing_edges": int(job.get("owned_crossing_edges", 0)),
                    "edge_ids": np.asarray(job["edge_ids"], dtype=np.int64),
                    "src": np.asarray(job["src"], dtype=np.int64),
                    "dst": np.asarray(job["dst"], dtype=np.int64),
                    "edge_count": int(job["edge_count"]),
                    "support_eligible_mask": np.asarray(
                        job.get("support_eligible_mask", np.zeros(job["edge_count"], dtype=bool)),
                        dtype=bool,
                    ),
                }
            )
        self._plan_cache = {
            "key": plan_key,
            "src": np.asarray(src, dtype=np.int64),
            "dst": np.asarray(dst, dtype=np.int64),
            "node_cluster": np.asarray(node_cluster, dtype=np.int32),
            "partition_node_counts": np.asarray(partition_node_counts, dtype=np.int64),
            "owners": np.asarray(owners, dtype=np.int32),
            "internal_counts": np.asarray(internal_counts, dtype=np.int64),
            "owned_crossing_counts": np.asarray(owned_crossing_counts, dtype=np.int64),
            "owner_loads": np.asarray(owner_loads, dtype=np.int64),
            "jobs": job_structs,
            "support_cache_key": support_cache_key,
            "bridge_stats": dict(bridge_stats),
            "support_cache_hit": bool(support_cache_hit),
            "owner_cache_hit": bool(owner_cache_hit),
            "owner_cache_path": owner_cache_path,
            "cluster_count": int(cluster_count),
            "workers": int(workers),
        }

    def _reuse_cached_plan(
        self,
        data,
        cached_plan,
        src,
        dst,
        num_nodes,
        num_edges,
        target_edges,
        crossing_policy,
        stage_times,
        start_time,
    ):
        """Run the seed-dependent portion of sparsify() using a cached plan."""
        # Structural check: canonical edges must match the cached plan exactly.
        cached_src = cached_plan["src"]
        cached_dst = cached_plan["dst"]
        if cached_src.shape != src.shape or cached_dst.shape != dst.shape:
            return None
        if not (np.array_equal(cached_src, src) and np.array_equal(cached_dst, dst)):
            return None

        stage_times["clusterdata"] = 0.0
        stage_times["node_partition_materialization"] = 0.0
        stage_times["edge_owner_cache_load"] = 0.0
        stage_times["edge_owner_assignment"] = 0.0
        stage_times["owned_edge_materialization"] = 0.0

        node_cluster = cached_plan["node_cluster"]
        partition_node_counts = cached_plan["partition_node_counts"]
        owners = cached_plan["owners"]
        internal_counts = cached_plan["internal_counts"]
        owned_crossing_counts = cached_plan["owned_crossing_counts"]
        owner_loads = cached_plan["owner_loads"]
        cluster_count = int(cached_plan["cluster_count"])
        workers = int(cached_plan["workers"])
        owner_cache_hit = bool(cached_plan.get("owner_cache_hit", False))
        owner_cache_path = cached_plan.get("owner_cache_path")

        jobs = [
            {
                "cluster_id": int(struct["cluster_id"]),
                "num_nodes": int(struct["num_nodes"]),
                "partition_node_count": int(struct["partition_node_count"]),
                "internal_edges": int(struct["internal_edges"]),
                "owned_crossing_edges": int(struct["owned_crossing_edges"]),
                "edge_ids": struct["edge_ids"],
                "src": struct["src"],
                "dst": struct["dst"],
                "edge_count": int(struct["edge_count"]),
                "support_mask": np.zeros(int(struct["edge_count"]), dtype=bool),
                "support_eligible_mask": struct["support_eligible_mask"],
                "support_scores": (
                    None
                    if self._current_support_scores is None
                    else self._current_support_scores[struct["edge_ids"]]
                ),
                "support_count": 0,
                "support_budget_edges": 0,
                "target_edges": 0,
            }
            for struct in cached_plan["jobs"]
        ]
        self._assign_support_budgets(jobs, target_edges)
        internal_edges = int(internal_counts.sum())
        crossing_edges = int(num_edges - internal_edges)
        owned_crossing_edges = int(owned_crossing_counts.sum())
        dropped_crossing_edges = int(crossing_edges - owned_crossing_edges)
        available_edges = int((owners >= 0).sum())
        print(
            f"[{self._label}] plan cache hit: reusing partitions/owners/jobs "
            f"clusters={cluster_count} owned_edges={available_edges}",
            flush=True,
        )

        support_cache_key = self._tensor_support_cache_key(
            num_nodes,
            num_edges,
            cluster_count,
            crossing_policy,
            target_edges,
            available_edges,
            owner_cache_path,
        )
        support_cache_hit = False
        support_budget_stats = None
        bridge_stats = self._apply_tensor_support_cache(jobs, support_cache_key)
        if bridge_stats is not None:
            support_cache_hit = True
            cached_support_edges = int(sum(job["support_count"] for job in jobs))
            support_budget_stats = self._support_budget_stats(
                cached_support_edges,
                cached_support_edges,
            )
            stage_times["local_support"] = 0.0
            stage_times["support_bridge"] = 0.0
            print(
                f"[{self._label}] local support cache hit: "
                f"support_edges={sum(job['support_count'] for job in jobs)}",
                flush=True,
            )
        else:
            support_start = time.perf_counter()
            full_support_cache_hit = self._apply_tensor_full_support_cache(
                jobs,
                support_cache_key,
            )
            if full_support_cache_hit:
                support_cache_hit = True
            else:
                jobs = self._build_local_supports(jobs, workers)
                self._store_tensor_full_support_cache(support_cache_key, jobs)
            support_budget_stats = self._trim_local_supports_to_budget(
                jobs,
                min(target_edges, available_edges),
            )
            stage_times["local_support"] = time.perf_counter() - support_start
            pre_bridge_support_floor = int(sum(job["support_count"] for job in jobs))
            pre_bridge_desired_final_edges = min(target_edges, available_edges)
            bridge_budget = max(
                0, pre_bridge_desired_final_edges - pre_bridge_support_floor
            )
            bridge_required = bridge_budget > 0
            bridge_start = time.perf_counter()
            bridge_stats = self._add_crossing_support_bridges(
                jobs,
                src,
                dst,
                owners,
                node_cluster,
                crossing_policy,
                num_nodes,
                bridge_required,
                owned_crossing_edges,
                max_edges=bridge_budget,
            )
            stage_times["support_bridge"] = time.perf_counter() - bridge_start
            self._store_tensor_support_cache(support_cache_key, jobs, bridge_stats)

        alloc_start = time.perf_counter()
        support_floor = int(sum(job["support_count"] for job in jobs))
        desired_final_edges = min(target_edges, available_edges)
        if support_floor > desired_final_edges:
            raise RuntimeError(
                "Budget-aware Scaffold support exceeded the requested target: "
                f"support={support_floor} target={desired_final_edges}"
            )
        target_feasible = bool(available_edges >= target_edges)
        extra_budget = max(0, int(desired_final_edges - support_floor))
        capacities = np.asarray(
            [job["edge_count"] - job["support_count"] for job in jobs],
            dtype=np.int64,
        )
        extra_targets = self._allocate_budget(capacities, extra_budget)
        for job, extra in zip(jobs, extra_targets.tolist()):
            job["target_edges"] = int(job["support_count"] + extra)
        stage_times["target_allocation"] = time.perf_counter() - alloc_start

        growth_start = time.perf_counter()
        results = self._grow_local_jobs(jobs, workers)
        stage_times["local_growth"] = time.perf_counter() - growth_start
        print(
            f"[{self._label}] local growth complete: "
            f"time={stage_times['local_growth']:.3f}s",
            flush=True,
        )

        output_start = time.perf_counter()
        selected_parts = [result["selected_edge_ids"] for result in results if result["selected_edge_ids"].size]
        if selected_parts:
            selected_edge_ids = np.unique(np.concatenate(selected_parts).astype(np.int64, copy=False))
        else:
            selected_edge_ids = np.empty(0, dtype=np.int64)
        out = self._build_output_data(data, num_nodes, src, dst, selected_edge_ids)
        stage_times["output_materialization"] = time.perf_counter() - output_start

        sampled_total = int(sum(result["sampled_candidates"] for result in results))
        added_total = int(sum(result["edges_added"] for result in results))
        swap_total = int(sum(result.get("swaps_accepted", 0) for result in results))
        swap_candidates_total = int(sum(result.get("swap_candidates", 0) for result in results))
        swap_passes_total = int(sum(result.get("swap_passes", 0) for result in results))
        max_rounds = int(max((result["rounds"] for result in results), default=0))
        total_rounds = int(sum(result["rounds"] for result in results))
        support_total = int(sum(result["support_edges"] for result in results))
        local_profile = self._summarize_local_profiles(results)
        local_profile["workers_used"] = int(workers)
        edge_balance = self._edge_owner_balance(owner_loads)
        partition_rows = self._partition_rows(jobs, results)
        total_time = time.perf_counter() - start_time
        self.owner.last_cluster_stats = {
            "algorithm": self._algorithm,
            "fast_mode": self.owner.fast_mode,
            "fast_score": self.owner.fast_score,
            "score_scope": (
                "sampled_batch"
                if getattr(self.owner, "algorithm_name", "") == "scaffold_batch"
                else "complete_candidate_set"
            ),
            "crossing_policy": crossing_policy,
            "cluster_count": int(cluster_count),
            "cluster_method": self.owner.cluster_method,
            "cache_path": self.owner.cluster_cache_dir,
            "edge_owner_cache_path": str(owner_cache_path) if owner_cache_path else None,
            "edge_owner_cache_hit": owner_cache_hit,
            "metis_recompute": bool(getattr(self.owner, "metis_recompute", False)),
            "sample_size": self.owner.sample_size,
            "cluster_add_per_round": self.owner.cluster_add_per_round,
            "parallel_workers": self.owner.parallel_workers,
            "workers_used": int(workers),
            "plan_cache_hit": True,
            "sampled_candidates": sampled_total,
            "rounds": max_rounds,
            "total_rounds": total_rounds,
            "edges_added": added_total,
            "swaps_accepted": swap_total,
            "swap_candidates": swap_candidates_total,
            "swap_passes": swap_passes_total,
            "local_support_edges": support_total,
            **support_budget_stats,
            "support_bridge_enabled": bool(bridge_stats["support_bridge_enabled"]),
            "support_bridge_mode": str(bridge_stats["support_bridge_mode"]),
            "support_bridge_skip_reason": str(bridge_stats["support_bridge_skip_reason"]),
            "bridge_support_edges": int(bridge_stats["bridge_support_edges"]),
            "support_cache_hit": bool(support_cache_hit),
            "original_components": int(bridge_stats["original_components"]),
            "support_components_before_bridge": int(bridge_stats["support_components_before_bridge"]),
            "support_components_after_bridge": int(bridge_stats["support_components_after_bridge"]),
            "internal_edges": internal_edges,
            "crossing_edges": crossing_edges,
            "owned_crossing_edges": owned_crossing_edges,
            "dropped_crossing_edges": dropped_crossing_edges,
            "available_edges": available_edges,
            "target_feasible": target_feasible,
            "support_floor_edges": support_floor,
            "target_edges": target_edges,
            "final_edges": int(selected_edge_ids.shape[0]),
            "edge_owner_load_min": edge_balance["min"],
            "edge_owner_load_median": edge_balance["median"],
            "edge_owner_load_max": edge_balance["max"],
            "edge_owner_balance_ratio": edge_balance["balance_ratio"],
            "clusterdata_time_sec": stage_times.get("clusterdata", 0.0),
            "clusterloader_materialization_time_sec": stage_times.get("owned_edge_materialization", 0.0),
            "local_support_time_sec": stage_times.get("local_support", 0.0),
            "local_growth_time_sec": stage_times.get("local_growth", 0.0),
            "output_materialization_time_sec": stage_times.get("output_materialization", 0.0),
            "sparsification_time_sec": total_time,
            "timing_sec": stage_times,
            "partition_edge_distribution": partition_rows,
            "local_profile": local_profile,
        }
        print(
            f"[{self._label} ClusterData summary] clusters={cluster_count} "
            f"mode={self.owner.fast_mode} score={self.owner.fast_score} method={self.owner.cluster_method} "
            f"crossing_policy={crossing_policy} "
            f"cache_path={self.owner.cluster_cache_dir} sample_size={self.owner.sample_size} "
            f"add_per_cluster={self.owner.cluster_add_per_round} workers={self.owner.parallel_workers} "
            f"workers_used={workers} internal={internal_edges} crossing={crossing_edges} "
            f"owned_crossing={owned_crossing_edges} dropped_crossing={dropped_crossing_edges} "
            f"support={support_total} bridge_mode={bridge_stats['support_bridge_mode']} "
            f"budget_mode={support_budget_stats['support_budget_mode']} "
            f"before_trim={support_budget_stats['support_edges_before_trim']} "
            f"trimmed={support_budget_stats['support_edges_trimmed']} "
            f"bridge_support={bridge_stats['bridge_support_edges']} "
            f"support_cache_hit={support_cache_hit} plan_cache_hit=True "
            f"sampled={sampled_total} max_rounds={max_rounds} "
            f"edges_added={added_total} swaps={swap_total} "
            f"final_edges={selected_edge_ids.shape[0]}/{target_edges} "
            f"target_feasible={target_feasible} time={total_time:.3f}s"
        )
        return out

    def _progress_enabled(self):
        setting = str(getattr(self.owner, "progress", "auto")).lower()
        if setting == "false":
            return False
        if tqdm is None:
            return False
        if setting == "true":
            return True
        return True

    def _progress_bar(self, *args, **kwargs):
        if not self._progress_enabled():
            return None
        total = kwargs.get("total")
        if total is None and args:
            try:
                total = len(args[0])
            except Exception:
                total = None
        if str(getattr(self.owner, "progress", "auto")).lower() == "auto" and total is not None and int(total) < 8:
            return None
        kwargs.setdefault("dynamic_ncols", True)
        kwargs.setdefault("mininterval", 0.5)
        kwargs.setdefault("smoothing", 0.1)
        kwargs.setdefault("file", sys.stderr)
        return tqdm(*args, **kwargs)

    def _build_output_data(self, data, num_nodes, src, dst, selected_edge_ids):
        out_src = np.concatenate((src[selected_edge_ids], dst[selected_edge_ids]))
        out_dst = np.concatenate((dst[selected_edge_ids], src[selected_edge_ids]))
        edge_index = torch.empty((2, out_src.shape[0]), dtype=torch.long)
        edge_index[0] = torch.from_numpy(out_src.astype(np.int64, copy=False))
        edge_index[1] = torch.from_numpy(out_dst.astype(np.int64, copy=False))

        out = Data(
            x=data.x,
            edge_index=edge_index,
            y=data.y,
            num_nodes=num_nodes,
        )
        out.edge_index_is_symmetric_unique = True
        out.num_undirected_edges = int(selected_edge_ids.shape[0])
        return out

    def _support_edge_scores(self, data, src, dst):
        method = str(getattr(self.owner, "support_weight_method", "uniform"))
        if method == "uniform":
            return None
        scores = self.owner._compute_feature_edge_scores(
            getattr(data, "x", None),
            torch.from_numpy(np.asarray(src, dtype=np.int64)),
            torch.from_numpy(np.asarray(dst, dtype=np.int64)),
        )
        return scores.numpy().astype(np.float32, copy=False)

    def _canonical_undirected_edges(self, data):
        edge_index = data.edge_index.detach().cpu()
        src = edge_index[0]
        dst = edge_index[1]
        if bool(getattr(data, "edge_index_is_undirected_unique", False)):
            return src.numpy().astype(np.int64, copy=False), dst.numpy().astype(np.int64, copy=False)
        if bool(getattr(data, "edge_index_is_symmetric_unique", False)):
            mask = src < dst
            src = src[mask]
            dst = dst[mask]
            return src.numpy().astype(np.int64, copy=False), dst.numpy().astype(np.int64, copy=False)

        mask = src != dst
        src = src[mask]
        dst = dst[mask]
        lo = torch.minimum(src, dst)
        hi = torch.maximum(src, dst)
        pairs = torch.stack((lo, hi), dim=0)
        pairs = torch.unique(pairs, dim=1)
        return pairs[0].numpy().astype(np.int64, copy=False), pairs[1].numpy().astype(np.int64, copy=False)

    def _build_cluster_data(self, num_nodes, src, dst):
        cluster_count = min(max(1, int(self.owner.cluster_count)), int(num_nodes))
        edge_ids = torch.arange(src.shape[0], dtype=torch.long)
        edge_index = torch.empty((2, src.shape[0] * 2), dtype=torch.long)
        edge_index[0, : src.shape[0]] = torch.from_numpy(src.astype(np.int64, copy=False))
        edge_index[1, : src.shape[0]] = torch.from_numpy(dst.astype(np.int64, copy=False))
        edge_index[0, src.shape[0] :] = edge_index[1, : src.shape[0]]
        edge_index[1, src.shape[0] :] = edge_index[0, : src.shape[0]]
        data = Data(
            edge_index=edge_index,
            edge_id=torch.cat((edge_ids, edge_ids), dim=0),
            node_id=torch.arange(num_nodes, dtype=torch.long),
            num_nodes=num_nodes,
        )

        kwargs = {
            "num_parts": cluster_count,
            "recursive": False,
            "save_dir": None,
            "log": False,
        }
        if self.owner.cluster_cache_dir:
            cache_dir = Path(self.owner.cluster_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            kwargs["save_dir"] = str(cache_dir)
            if "filename" in inspect.signature(ClusterData).parameters:
                kwargs["filename"] = self._cluster_data_cache_filename(cluster_count, num_nodes, src.shape[0])
            if bool(getattr(self.owner, "metis_recompute", False)):
                for path in self._cluster_data_cache_paths(cache_dir, cluster_count, num_nodes, src.shape[0]):
                    self._remove_cache_file(path, "ClusterData")
        signature = inspect.signature(ClusterData)
        if "keep_inter_cluster_edges" in signature.parameters:
            kwargs["keep_inter_cluster_edges"] = False
        if "sparse_format" in signature.parameters:
            kwargs["sparse_format"] = "csr"
        _register_clusterdata_safe_globals()
        return ClusterData(data, **kwargs)

    def _cluster_data_cache_filename(self, cluster_count, num_nodes, num_edges):
        return f"clusterloader_parts_{int(cluster_count)}_n{int(num_nodes)}_m{int(num_edges)}.pt"

    def _cluster_data_cache_paths(self, cache_dir, cluster_count, num_nodes, num_edges):
        filename = self._cluster_data_cache_filename(cluster_count, num_nodes, num_edges)
        cache_dir = Path(cache_dir)
        return (
            cache_dir / filename,
            cache_dir / f"part_{int(cluster_count)}" / filename,
        )

    def _remove_cache_file(self, path, label):
        if path is None:
            return False
        path = Path(path)
        if not path.exists():
            return False
        try:
            path.unlink()
            print(f"[{self._label}] removed {label} cache: {path}", flush=True)
            return True
        except Exception as exc:
            print(f"[{self._label}] could not remove {label} cache {path}: {exc}", flush=True)
            return False

    def _edge_owner_cache_path(self, num_nodes, num_edges, cluster_count, crossing_policy):
        if not self.owner.cluster_cache_dir:
            return None
        cache_dir = Path(self.owner.cluster_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_policy = str(crossing_policy).replace("/", "_")
        return cache_dir / (
            f"edge_owner_v1_{safe_policy}_parts_{int(cluster_count)}_"
            f"n{int(num_nodes)}_m{int(num_edges)}.npz"
        )

    def _load_edge_owner_cache(self, path, num_edges, cluster_count):
        if path is None or not Path(path).exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as payload:
                owners = payload["owners"].astype(np.int32, copy=False)
                internal_counts = payload["internal_counts"].astype(np.int64, copy=False)
                owned_crossing_counts = payload["owned_crossing_counts"].astype(np.int64, copy=False)
                owner_loads = payload["owner_loads"].astype(np.int64, copy=False)
        except Exception as exc:
            print(f"[{self._label}] ignoring unreadable edge-owner cache {path}: {exc}", flush=True)
            return None
        if owners.shape[0] != int(num_edges):
            return None
        if internal_counts.shape[0] != int(cluster_count):
            return None
        if owned_crossing_counts.shape[0] != int(cluster_count):
            return None
        if owner_loads.shape[0] != int(cluster_count):
            return None
        return owners, internal_counts, owned_crossing_counts, owner_loads

    def _write_edge_owner_cache(self, path, owners, internal_counts, owned_crossing_counts, owner_loads):
        if path is None:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp-{int(time.time_ns())}.npz")
        try:
            np.savez(
                tmp_path,
                owners=owners.astype(np.int32, copy=False),
                internal_counts=internal_counts.astype(np.int64, copy=False),
                owned_crossing_counts=owned_crossing_counts.astype(np.int64, copy=False),
                owner_loads=owner_loads.astype(np.int64, copy=False),
            )
            tmp_path.replace(path)
        except Exception as exc:
            print(f"[{self._label}] could not write edge-owner cache {path}: {exc}", flush=True)
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _node_clusters_from_cluster_data(self, cluster_data, num_nodes, cluster_count):
        partition = getattr(cluster_data, "partition", None)
        if partition is not None and hasattr(partition, "node_perm") and hasattr(partition, "partptr"):
            node_perm = partition.node_perm.detach().cpu().numpy().astype(np.int64, copy=False)
            partptr = partition.partptr.detach().cpu().numpy().astype(np.int64, copy=False)
            node_cluster = np.full(int(num_nodes), -1, dtype=np.int32)
            node_counts = np.zeros(int(cluster_count), dtype=np.int64)
            for cluster_id in range(int(cluster_count)):
                start = int(partptr[cluster_id])
                end = int(partptr[cluster_id + 1])
                nodes = node_perm[start:end]
                node_cluster[nodes] = int(cluster_id)
                node_counts[cluster_id] = int(nodes.shape[0])
            if np.any(node_cluster < 0):
                raise RuntimeError("ClusterData partition did not assign every node.")
            return node_cluster, node_counts

        node_cluster = np.full(int(num_nodes), -1, dtype=np.int32)
        node_counts = np.zeros(int(cluster_count), dtype=np.int64)
        for cluster_id in range(int(cluster_count)):
            batch = cluster_data[int(cluster_id)]
            node_id = getattr(batch, "node_id", None)
            if node_id is None:
                raise RuntimeError("ClusterData batch is missing node_id; cannot recover node partitions.")
            nodes = node_id.detach().cpu().numpy().astype(np.int64, copy=False)
            node_cluster[nodes] = int(cluster_id)
            node_counts[cluster_id] = int(nodes.shape[0])
        if np.any(node_cluster < 0):
            raise RuntimeError("ClusterData batches did not assign every node.")
        return node_cluster, node_counts

    def _assign_edge_owners(self, src, dst, node_cluster, cluster_count, crossing_policy):
        if crossing_policy not in ("balanced_owner", "drop"):
            raise ValueError("scaffold_crossing_policy must be one of {'balanced_owner', 'drop'}.")
        drop_crossing = crossing_policy == "drop"
        if njit is not None:
            return _assign_edge_owners_numba(
                src.astype(np.int64, copy=False),
                dst.astype(np.int64, copy=False),
                node_cluster.astype(np.int32, copy=False),
                int(cluster_count),
                bool(drop_crossing),
            )
        return self._assign_edge_owners_python(src, dst, node_cluster, cluster_count, drop_crossing)

    def _assign_edge_owners_python(self, src, dst, node_cluster, cluster_count, drop_crossing):
        owners = np.full(src.shape[0], -1, dtype=np.int32)
        internal_counts = np.zeros(int(cluster_count), dtype=np.int64)
        owned_crossing_counts = np.zeros(int(cluster_count), dtype=np.int64)
        loads = np.zeros(int(cluster_count), dtype=np.int64)

        for edge_id, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
            cu = int(node_cluster[int(u)])
            cv = int(node_cluster[int(v)])
            if cu == cv:
                owners[edge_id] = cu
                internal_counts[cu] += 1
                loads[cu] += 1

        if drop_crossing:
            return owners, internal_counts, owned_crossing_counts, loads

        for edge_id, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
            cu = int(node_cluster[int(u)])
            cv = int(node_cluster[int(v)])
            if cu == cv:
                continue
            owner = cu if loads[cu] <= loads[cv] else cv
            owners[edge_id] = owner
            owned_crossing_counts[owner] += 1
            loads[owner] += 1
        return owners, internal_counts, owned_crossing_counts, loads

    def _materialize_owned_jobs(
        self,
        src,
        dst,
        owners,
        node_cluster,
        partition_node_counts,
        internal_counts,
        owned_crossing_counts,
        workers,
    ):
        cluster_count = int(partition_node_counts.shape[0])
        valid_edge_ids = np.flatnonzero(owners >= 0).astype(np.int64, copy=False)
        if valid_edge_ids.size:
            order = np.argsort(owners[valid_edge_ids], kind="stable")
            sorted_edge_ids = valid_edge_ids[order]
            sorted_owners = owners[sorted_edge_ids]
            boundaries = np.searchsorted(sorted_owners, np.arange(cluster_count + 1), side="left")
        else:
            sorted_edge_ids = np.empty(0, dtype=np.int64)
            boundaries = np.zeros(cluster_count + 1, dtype=np.int64)

        if workers <= 1:
            return [
                self._build_owned_job(
                    cluster_id,
                    sorted_edge_ids[boundaries[cluster_id] : boundaries[cluster_id + 1]],
                    src,
                    dst,
                    node_cluster,
                    partition_node_counts,
                    internal_counts,
                    owned_crossing_counts,
                )
                for cluster_id in range(cluster_count)
            ]

        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    self._build_owned_job,
                    cluster_id,
                    sorted_edge_ids[boundaries[cluster_id] : boundaries[cluster_id + 1]],
                    src,
                    dst,
                    node_cluster,
                    partition_node_counts,
                    internal_counts,
                    owned_crossing_counts,
                )
                for cluster_id in range(cluster_count)
            ]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda job: job["cluster_id"])
        return results

    def _build_owned_job(
        self,
        cluster_id,
        edge_ids,
        src,
        dst,
        node_cluster,
        partition_node_counts,
        internal_counts,
        owned_crossing_counts,
    ):
        edge_ids = np.asarray(edge_ids, dtype=np.int64)
        if edge_ids.size == 0:
            return self._empty_job(
                cluster_id,
                int(partition_node_counts[cluster_id]),
                int(internal_counts[cluster_id]),
                int(owned_crossing_counts[cluster_id]),
            )

        endpoints = np.concatenate((src[edge_ids], dst[edge_ids])).astype(np.int64, copy=False)
        _, inverse = np.unique(endpoints, return_inverse=True)
        local_src = inverse[: edge_ids.shape[0]].astype(np.int64, copy=False)
        local_dst = inverse[edge_ids.shape[0] :].astype(np.int64, copy=False)
        support_eligible_mask = (
            (node_cluster[src[edge_ids]] == int(cluster_id))
            & (node_cluster[dst[edge_ids]] == int(cluster_id))
        )
        return {
            "cluster_id": int(cluster_id),
            "num_nodes": int(inverse.max()) + 1 if inverse.size else 0,
            "partition_node_count": int(partition_node_counts[cluster_id]),
            "internal_edges": int(internal_counts[cluster_id]),
            "owned_crossing_edges": int(owned_crossing_counts[cluster_id]),
            "edge_ids": edge_ids.astype(np.int64, copy=False),
            "src": local_src,
            "dst": local_dst,
            "edge_count": int(edge_ids.shape[0]),
            "support_mask": np.zeros(edge_ids.shape[0], dtype=bool),
            "support_eligible_mask": support_eligible_mask.astype(bool, copy=False),
            "support_scores": (
                None
                if self._current_support_scores is None
                else self._current_support_scores[edge_ids]
            ),
            "support_count": 0,
            "support_budget_edges": 0,
            "target_edges": 0,
        }

    def _empty_job(self, cluster_id, partition_node_count=0, internal_edges=0, owned_crossing_edges=0):
        return {
            "cluster_id": int(cluster_id),
            "num_nodes": 0,
            "partition_node_count": int(partition_node_count),
            "internal_edges": int(internal_edges),
            "owned_crossing_edges": int(owned_crossing_edges),
            "edge_ids": np.empty(0, dtype=np.int64),
            "src": np.empty(0, dtype=np.int64),
            "dst": np.empty(0, dtype=np.int64),
            "edge_count": 0,
            "support_mask": np.empty(0, dtype=bool),
            "support_eligible_mask": np.empty(0, dtype=bool),
            "support_scores": None,
            "support_count": 0,
            "support_budget_edges": 0,
            "target_edges": 0,
        }

    def _assign_support_budgets(self, jobs, target_edges):
        """Allocate either the target cap or full local-support capacity."""
        capacities = np.asarray(
            [
                int(np.asarray(job.get("support_eligible_mask"), dtype=bool).sum())
                for job in jobs
            ],
            dtype=np.int64,
        )
        if str(getattr(self.owner, "support_budget_mode", "early_stop")) == "full_then_random_trim":
            budget = int(capacities.sum())
        else:
            budget = min(max(0, int(target_edges)), int(capacities.sum()))
        allocations = self._allocate_budget(capacities, budget)
        for job, allocation in zip(jobs, allocations.tolist()):
            job["support_budget_edges"] = int(allocation)

    def _build_local_supports(self, jobs, workers):
        if not jobs:
            return jobs
        active_support_jobs = sum(
            int(job.get("support_budget_edges", 0)) > 0 for job in jobs
        )
        outer_parallel = int(workers) > 1 and active_support_jobs > 1
        kernel_workers = 1 if outer_parallel else max(
            1, int(self.owner.parallel_workers)
        )
        for job in jobs:
            job["support_kernel_workers"] = kernel_workers
        if workers <= 1:
            results = []
            support_edges = 0
            total_edges = 0
            bar = self._progress_bar(
                total=len(jobs),
                desc=f"SCAFFOLD support trees ({len(jobs)} parts)",
                unit="part",
                leave=True,
            )
            try:
                for job in jobs:
                    result = self._build_local_support(job)
                    results.append(result)
                    support_edges += int(result.get("support_count", 0))
                    total_edges += int(result.get("edge_count", 0))
                    if bar is not None:
                        bar.update(1)
                        bar.set_postfix(
                            support_edges=support_edges,
                            owned_edges=total_edges,
                            refresh=False,
                        )
            finally:
                if bar is not None:
                    bar.close()
            return results
        results = []
        support_edges = 0
        total_edges = 0
        bar = self._progress_bar(
            total=len(jobs),
            desc=f"SCAFFOLD support trees ({len(jobs)} parts)",
            unit="part",
            leave=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self._build_local_support, job) for job in jobs]
            try:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    support_edges += int(result.get("support_count", 0))
                    total_edges += int(result.get("edge_count", 0))
                    if bar is not None:
                        bar.update(1)
                        bar.set_postfix(
                            support_edges=support_edges,
                            owned_edges=total_edges,
                            refresh=False,
                        )
            finally:
                if bar is not None:
                    bar.close()
        results.sort(key=lambda job: job["cluster_id"])
        return results

    def _build_local_support(self, job):
        start = time.perf_counter()
        if job["edge_count"] == 0:
            job["support_sec"] = 0.0
            return job
        support_eligible_mask = np.asarray(job.get("support_eligible_mask"), dtype=bool)
        if support_eligible_mask.size == 0:
            support_mask = np.zeros(job["edge_count"], dtype=bool)
        else:
            support_mask = np.zeros(job["edge_count"], dtype=bool)
            eligible_idx = np.flatnonzero(support_eligible_mask).astype(np.int64, copy=False)
            support_budget = min(
                int(eligible_idx.size),
                max(0, int(job.get("support_budget_edges", eligible_idx.size))),
            )
            if eligible_idx.size and support_budget > 0:
                eligible_support = self._build_support_mask(
                    job["num_nodes"],
                    job["src"][eligible_idx],
                    job["dst"][eligible_idx],
                    job["cluster_id"],
                    scores=(
                        None
                        if job.get("support_scores") is None
                        else np.asarray(job["support_scores"])[eligible_idx]
                    ),
                    max_edges=support_budget,
                    parallel_workers=job.get("support_kernel_workers", 1),
                )
                support_mask[eligible_idx] = eligible_support
        job["support_mask"] = support_mask
        job["support_count"] = int(support_mask.sum())
        job["support_sec"] = time.perf_counter() - start
        return job

    def _support_budget_stats(self, before_trim, after_trim):
        before_trim = int(before_trim)
        after_trim = int(after_trim)
        return {
            "support_budget_mode": str(
                getattr(self.owner, "support_budget_mode", "early_stop")
            ),
            "support_edges_before_trim": before_trim,
            "support_edges_trimmed": max(0, before_trim - after_trim),
            "support_edges_after_trim": after_trim,
        }

    def _trim_local_supports_to_budget(self, jobs, target_edges):
        """Uniformly trim the completed local supports to one global budget."""
        support_edge_ids = self._support_edge_ids(jobs)
        before_trim = int(support_edge_ids.shape[0])
        mode = str(getattr(self.owner, "support_budget_mode", "early_stop"))
        target_edges = max(0, int(target_edges))
        if mode != "full_then_random_trim" or before_trim <= target_edges:
            return self._support_budget_stats(before_trim, before_trim)

        generator = torch.Generator(device="cpu")
        active_seed = self.owner._active_seed()
        if active_seed is None:
            generator.seed()
        else:
            generator.manual_seed(int(active_seed))
        torch.multinomial(
            torch.ones(1, dtype=torch.double),
            num_samples=1,
            replacement=False,
            generator=generator,
        )
        drop_count = before_trim - target_edges
        retained_positions = torch.randperm(
            before_trim,
            generator=generator,
        )[drop_count:].numpy()
        retained_edge_ids = np.sort(support_edge_ids[retained_positions])

        for job in jobs:
            edge_ids = np.asarray(job.get("edge_ids"), dtype=np.int64)
            existing_mask = np.asarray(job.get("support_mask"), dtype=bool)
            if edge_ids.size == 0 or retained_edge_ids.size == 0:
                support_mask = np.zeros(edge_ids.shape[0], dtype=bool)
            else:
                positions = np.searchsorted(retained_edge_ids, edge_ids)
                safe_positions = np.minimum(positions, retained_edge_ids.shape[0] - 1)
                retained_mask = (
                    (positions < retained_edge_ids.shape[0])
                    & (retained_edge_ids[safe_positions] == edge_ids)
                )
                support_mask = existing_mask & retained_mask
            job["support_mask"] = support_mask.astype(bool, copy=False)
            job["support_count"] = int(job["support_mask"].sum())

        after_trim = int(sum(job["support_count"] for job in jobs))
        return self._support_budget_stats(before_trim, after_trim)

    def _add_crossing_support_bridges(
        self,
        jobs,
        src,
        dst,
        owners,
        node_cluster,
        crossing_policy,
        num_nodes,
        bridge_required,
        bridge_candidate_count,
        max_edges=None,
    ):
        if max_edges is not None and int(max_edges) <= 0:
            return {
                "support_bridge_enabled": False,
                "support_bridge_mode": str(getattr(self.owner, "support_bridge", "auto")),
                "support_bridge_skip_reason": "target_budget_exhausted",
                "bridge_support_edges": 0,
                "original_components": -1,
                "support_components_before_bridge": -1,
                "support_components_after_bridge": -1,
            }
        bridge_enabled, skip_reason = self._support_bridge_decision(
            crossing_policy,
            bridge_required,
            bridge_candidate_count,
        )
        if not bridge_enabled:
            return {
                "support_bridge_enabled": False,
                "support_bridge_mode": str(getattr(self.owner, "support_bridge", "auto")),
                "support_bridge_skip_reason": skip_reason,
                "bridge_support_edges": 0,
                "original_components": -1,
                "support_components_before_bridge": -1,
                "support_components_after_bridge": -1,
            }
        original_components = self._component_count(num_nodes, src, dst)
        support_edge_ids = self._support_edge_ids(jobs)
        if owners.size == 0:
            return {
                "support_bridge_enabled": True,
                "support_bridge_mode": str(getattr(self.owner, "support_bridge", "auto")),
                "support_bridge_skip_reason": "",
                "bridge_support_edges": 0,
                "original_components": int(original_components),
                "support_components_before_bridge": int(num_nodes),
                "support_components_after_bridge": int(num_nodes),
            }

        support_global = np.zeros(src.shape[0], dtype=bool)
        if support_edge_ids.size:
            support_global[support_edge_ids] = True
        crossing_mask = node_cluster[src] != node_cluster[dst]
        candidate_edge_ids = np.flatnonzero((owners >= 0) & crossing_mask & (~support_global)).astype(
            np.int64,
            copy=False,
        )
        offset, stride = self._bridge_iteration_params(candidate_edge_ids.shape[0])
        bridge_edge_ids, support_components_before, support_components_after = self._bridge_support_edges(
            num_nodes,
            src,
            dst,
            support_edge_ids,
            candidate_edge_ids,
            offset,
            stride,
            original_components,
            max_edges=max_edges,
        )
        self._mark_support_edges(jobs, owners, bridge_edge_ids)
        return {
            "support_bridge_enabled": True,
            "support_bridge_mode": str(getattr(self.owner, "support_bridge", "auto")),
            "support_bridge_skip_reason": "",
            "bridge_support_edges": int(bridge_edge_ids.shape[0]),
            "original_components": int(original_components),
            "support_components_before_bridge": int(support_components_before),
            "support_components_after_bridge": int(support_components_after),
        }

    def _support_bridge_decision(self, crossing_policy, bridge_required, bridge_candidate_count):
        setting = str(getattr(self.owner, "support_bridge", "auto")).lower()
        if setting == "false":
            return False, "disabled"
        if str(crossing_policy) != "balanced_owner":
            return False, "crossing_policy"
        if setting == "true":
            return True, ""
        if int(bridge_candidate_count) <= 0:
            return False, "no_candidates"
        if not bool(bridge_required):
            return False, "not_required"
        max_candidates = int(getattr(self.owner, "support_bridge_max_candidates", 1000000))
        if max_candidates > 0 and int(bridge_candidate_count) > max_candidates:
            return False, f"candidate_cap:{int(bridge_candidate_count)}>{max_candidates}"
        return True, ""

    def _support_edge_ids(self, jobs):
        parts = [
            job["edge_ids"][np.asarray(job["support_mask"], dtype=bool)]
            for job in jobs
            if int(job.get("support_count", 0)) > 0
        ]
        if not parts:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(parts).astype(np.int64, copy=False)

    def _tensor_support_cache_key(
        self,
        num_nodes,
        num_edges,
        cluster_count,
        crossing_policy,
        target_edges,
        available_edges,
        owner_cache_path,
    ):
        return (
            "tensor_edge_ids",
            str(self.owner.init_support),
            int(num_nodes),
            int(num_edges),
            int(cluster_count),
            str(crossing_policy),
            int(target_edges),
            int(available_edges),
            (
                int(self.owner.fast_tree_buckets)
                if str(self.owner.init_support) in ("fast_mst", "fast_maxst")
                else None
            ),
            str(getattr(self.owner, "support_bridge", "auto")),
            int(getattr(self.owner, "support_bridge_max_candidates", 1000000)),
            str(getattr(self.owner, "support_budget_mode", "early_stop")),
            str(getattr(self.owner, "support_weight_method", "uniform")),
            str(owner_cache_path) if owner_cache_path is not None else None,
        )

    def _tensor_support_cacheable(self):
        if str(getattr(self.owner, "support_budget_mode", "early_stop")) != "early_stop":
            return False
        if str(self.owner.init_support) not in (
            "mst",
            "maxst",
            "fast_mst",
            "fast_maxst",
        ):
            return False
        if bool(getattr(self.owner, "metis_recompute", False)):
            return False
        return True

    def _tensor_full_support_cacheable(self):
        if str(getattr(self.owner, "support_budget_mode", "early_stop")) != "full_then_random_trim":
            return False
        if str(self.owner.init_support) not in (
            "mst",
            "maxst",
            "fast_mst",
            "fast_maxst",
        ):
            return False
        return not bool(getattr(self.owner, "metis_recompute", False))

    def _set_job_support_from_edge_ids(self, jobs, support_edge_ids):
        support_edge_ids = np.asarray(support_edge_ids, dtype=np.int64)
        if support_edge_ids.size:
            support_edge_ids = np.sort(support_edge_ids.astype(np.int64, copy=False))
        for job in jobs:
            edge_ids = np.asarray(job.get("edge_ids"), dtype=np.int64)
            if edge_ids.size == 0 or support_edge_ids.size == 0:
                support_mask = np.zeros(edge_ids.shape[0], dtype=bool)
            else:
                positions = np.searchsorted(support_edge_ids, edge_ids)
                safe_positions = np.minimum(positions, support_edge_ids.shape[0] - 1)
                support_mask = (
                    (positions < support_edge_ids.shape[0])
                    & (support_edge_ids[safe_positions] == edge_ids)
                )
            job["support_mask"] = support_mask.astype(bool, copy=False)
            job["support_count"] = int(job["support_mask"].sum())
            job["support_sec"] = 0.0

    def _apply_tensor_full_support_cache(self, jobs, cache_key):
        if not self._tensor_full_support_cacheable():
            return False
        cache = getattr(self.owner, "_tensor_full_support_cache", None)
        if (
            not isinstance(cache, dict)
            or cache.get("kind") != "tensor_full_support_edge_ids"
            or cache.get("key") != cache_key
        ):
            return False
        self._set_job_support_from_edge_ids(jobs, cache.get("support_edge_ids"))
        return True

    def _store_tensor_full_support_cache(self, cache_key, jobs):
        if not self._tensor_full_support_cacheable():
            return
        support_edge_ids = self._support_edge_ids(jobs)
        self.owner._tensor_full_support_cache = {
            "kind": "tensor_full_support_edge_ids",
            "init_support": str(self.owner.init_support),
            "key": cache_key,
            "support_edge_ids": np.sort(
                support_edge_ids.astype(np.int64, copy=True)
            ),
        }

    def _apply_tensor_support_cache(self, jobs, cache_key):
        if not self._tensor_support_cacheable():
            return None
        cache = getattr(self.owner, "_tensor_deterministic_support_cache", None)
        if (
            not isinstance(cache, dict)
            or cache.get("kind") != "tensor_edge_ids"
            or cache.get("key") != cache_key
        ):
            return None

        self._set_job_support_from_edge_ids(jobs, cache.get("support_edge_ids"))

        bridge_stats = dict(cache.get("bridge_stats", {}))
        if not bridge_stats:
            bridge_stats = {
                "support_bridge_enabled": False,
                "support_bridge_mode": str(getattr(self.owner, "support_bridge", "auto")),
                "support_bridge_skip_reason": "cache",
                "bridge_support_edges": 0,
                "original_components": -1,
                "support_components_before_bridge": -1,
                "support_components_after_bridge": -1,
            }
        return bridge_stats

    def _store_tensor_support_cache(self, cache_key, jobs, bridge_stats):
        if not self._tensor_support_cacheable():
            return
        support_edge_ids = self._support_edge_ids(jobs)
        if support_edge_ids.size:
            support_edge_ids = np.sort(support_edge_ids.astype(np.int64, copy=True))
        else:
            support_edge_ids = np.empty(0, dtype=np.int64)
        self.owner._tensor_deterministic_support_cache = {
            "kind": "tensor_edge_ids",
            "init_support": str(self.owner.init_support),
            "key": cache_key,
            "support_edge_ids": support_edge_ids,
            "bridge_stats": dict(bridge_stats),
        }

    def _bridge_iteration_params(self, candidate_count):
        m = int(candidate_count)
        if m <= 1:
            return 0, 1
        if self.owner.init_support not in ("randst", "fast_randst"):
            return 0, 1
        base_seed = 0 if self.owner._active_seed() is None else int(self.owner._active_seed())
        seed = int(np.random.SeedSequence([base_seed & 0xFFFFFFFF, 0x5CAFF01D]).generate_state(1)[0])
        rng = np.random.default_rng(seed)
        offset = int(rng.integers(0, m))
        stride = int(rng.integers(1, m))
        if stride % 2 == 0:
            stride += 1
        while math.gcd(stride, m) != 1:
            stride += 2
            if stride >= m:
                stride = 1
                break
        return int(offset), int(stride)

    def _bridge_support_edges(
        self,
        num_nodes,
        src,
        dst,
        support_edge_ids,
        candidate_edge_ids,
        offset,
        stride,
        target_components,
        max_edges=None,
    ):
        edge_limit = int(candidate_edge_ids.shape[0]) if max_edges is None else max(
            0, int(max_edges)
        )
        if njit is not None:
            return _bridge_support_edges_numba(
                int(num_nodes),
                src,
                dst,
                support_edge_ids,
                candidate_edge_ids,
                int(offset),
                int(stride),
                int(target_components),
                int(edge_limit),
            )
        return self._bridge_support_edges_python(
            int(num_nodes),
            src,
            dst,
            support_edge_ids,
            candidate_edge_ids,
            int(offset),
            int(stride),
            int(target_components),
            int(edge_limit),
        )

    def _bridge_support_edges_python(
        self,
        num_nodes,
        src,
        dst,
        support_edge_ids,
        candidate_edge_ids,
        offset,
        stride,
        target_components,
        max_edges,
    ):
        parent = np.arange(int(num_nodes), dtype=np.int64)
        rank = np.zeros(int(num_nodes), dtype=np.int8)
        components = int(num_nodes)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return int(x)

        def union(u, v):
            nonlocal components
            if u == v:
                return False
            ru = find(int(u))
            rv = find(int(v))
            if ru == rv:
                return False
            if rank[ru] < rank[rv]:
                parent[ru] = rv
            elif rank[ru] > rank[rv]:
                parent[rv] = ru
            else:
                parent[rv] = ru
                rank[ru] += 1
            components -= 1
            return True

        for edge_id in support_edge_ids.tolist():
            union(src[int(edge_id)], dst[int(edge_id)])
        initial_components = int(components)

        bridges = []
        m = int(candidate_edge_ids.shape[0])
        edge_limit = max(0, int(max_edges))
        if edge_limit <= 0:
            return np.empty(0, dtype=np.int64), initial_components, int(components)
        for step in range(m):
            i = (int(offset) + step * int(stride)) % m
            edge_id = int(candidate_edge_ids[i])
            if union(src[edge_id], dst[edge_id]):
                bridges.append(edge_id)
                if len(bridges) >= edge_limit or components <= int(target_components):
                    break
        return np.asarray(bridges, dtype=np.int64), initial_components, int(components)

    def _component_count(self, num_nodes, src, dst):
        cache_key = (int(num_nodes), int(src.shape[0]))
        cache = getattr(self.owner, "_tensor_component_count_cache", None)
        if isinstance(cache, dict) and cache_key in cache:
            return int(cache[cache_key])
        if njit is not None:
            components = int(_component_count_numba(int(num_nodes), src, dst))
        else:
            components = self._component_count_from_arrays_python(int(num_nodes), src, dst)
        if not isinstance(cache, dict):
            cache = {}
            self.owner._tensor_component_count_cache = cache
        cache[cache_key] = int(components)
        return int(components)

    def _component_count_from_edge_ids(self, num_nodes, src, dst, edge_ids):
        if edge_ids.size == 0:
            return int(num_nodes)
        if njit is not None:
            _bridges, components, _after = _bridge_support_edges_numba(
                int(num_nodes),
                src,
                dst,
                edge_ids,
                np.empty(0, dtype=np.int64),
                0,
                1,
                0,
                0,
            )
            return int(components)
        return int(
            self._bridge_support_edges_python(
                int(num_nodes),
                src,
                dst,
                edge_ids,
                np.empty(0, dtype=np.int64),
                0,
                1,
                0,
                0,
            )[1]
        )

    def _component_count_from_arrays_python(self, num_nodes, src, dst):
        parent = np.arange(int(num_nodes), dtype=np.int64)
        rank = np.zeros(int(num_nodes), dtype=np.int8)
        components = int(num_nodes)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return int(x)

        for u, v in zip(src.tolist(), dst.tolist()):
            if u == v:
                continue
            ru = find(int(u))
            rv = find(int(v))
            if ru == rv:
                continue
            if rank[ru] < rank[rv]:
                parent[ru] = rv
            elif rank[ru] > rank[rv]:
                parent[rv] = ru
            else:
                parent[rv] = ru
                rank[ru] += 1
            components -= 1
        return int(components)

    def _mark_support_edges(self, jobs, owners, edge_ids):
        if edge_ids.size == 0:
            return
        touched = set()
        for edge_id in edge_ids.tolist():
            edge_id = int(edge_id)
            owner = int(owners[edge_id])
            if owner < 0 or owner >= len(jobs):
                continue
            job = jobs[owner]
            local_pos = int(np.searchsorted(job["edge_ids"], edge_id))
            if local_pos >= job["edge_ids"].shape[0] or int(job["edge_ids"][local_pos]) != edge_id:
                matches = np.flatnonzero(job["edge_ids"] == edge_id)
                if matches.size == 0:
                    continue
                local_pos = int(matches[0])
            job["support_mask"][local_pos] = True
            touched.add(owner)
        for owner in touched:
            jobs[owner]["support_count"] = int(np.asarray(jobs[owner]["support_mask"], dtype=bool).sum())

    def _build_support_mask(
        self,
        num_nodes,
        src,
        dst,
        cluster_id,
        scores=None,
        max_edges=None,
        parallel_workers=None,
    ):
        if self.owner.init_support not in (
            "maxst",
            "mst",
            "fast_maxst",
            "fast_mst",
            "randst",
            "fast_randst",
        ):
            raise ValueError(
                f"{self._label} supports init_support in "
                "{'maxst', 'mst', 'fast-maxst', 'fast-mst', "
                "'randst', 'fast-randst'}."
            )
        if self.owner.init_support == "fast_randst":
            base_seed = 0 if self.owner._active_seed() is None else int(self.owner._active_seed())
            seed = int(np.random.SeedSequence([base_seed & 0xFFFFFFFF, int(cluster_id)]).generate_state(1)[0])
            return self._build_fast_random_support_mask(
                num_nodes, src, dst, seed, max_edges=max_edges
            )

        if self.owner.init_support == "randst":
            from .spanning_tree import build_random_spanning_forest_mask

            base_seed = 0 if self.owner._active_seed() is None else int(self.owner._active_seed())
            seed = int(np.random.SeedSequence([base_seed & 0xFFFFFFFF, int(cluster_id)]).generate_state(1)[0])
            return build_random_spanning_forest_mask(
                num_nodes,
                src,
                dst,
                seed=seed,
                max_edges=max_edges,
            )

        if self.owner.init_support in ("fast_maxst", "fast_mst"):
            from .spanning_tree import build_fast_weighted_spanning_forest_mask

            return build_fast_weighted_spanning_forest_mask(
                num_nodes,
                src,
                dst,
                scores=scores,
                maximum=self.owner.init_support == "fast_maxst",
                bucket_count=self.owner.fast_tree_buckets,
                max_edges=max_edges,
                parallel_workers=parallel_workers,
            )

        if scores is not None:
            from .spanning_tree import build_weighted_spanning_forest_mask

            return build_weighted_spanning_forest_mask(
                num_nodes,
                src,
                dst,
                scores,
                maximum=self.owner.init_support == "maxst",
                max_edges=max_edges,
            )

        from .spanning_tree import build_spanning_forest_mask

        return build_spanning_forest_mask(
            num_nodes,
            src,
            dst,
            max_edges=max_edges,
        )

    def _build_fast_random_support_mask(
        self, num_nodes, src, dst, seed, max_edges=None
    ):
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        m = int(src.shape[0])
        if m == 0:
            return np.empty(0, dtype=bool)
        limit = min(
            max(0, int(num_nodes) - 1),
            m if max_edges is None else max(0, int(max_edges)),
        )
        if limit <= 0:
            return np.zeros(m, dtype=bool)
        if m == 1:
            return self._strided_spanning_forest_mask_python(
                int(num_nodes), src, dst, 0, 1, max_edges=limit
            )
        rng = np.random.default_rng(seed)
        offset = int(rng.integers(0, m))
        stride = int(rng.integers(1, m))
        if stride % 2 == 0:
            stride += 1
        while math.gcd(stride, m) != 1:
            stride += 2
            if stride >= m:
                stride = 1
                break
        if njit is not None:
            return _strided_spanning_forest_mask_numba(
                int(num_nodes),
                src,
                dst,
                int(offset),
                int(stride),
                int(limit),
            )
        return self._strided_spanning_forest_mask_python(
            int(num_nodes),
            src,
            dst,
            int(offset),
            int(stride),
            max_edges=limit,
        )

    def _strided_spanning_forest_mask_python(
        self, num_nodes, src, dst, offset, stride, max_edges=None
    ):
        parent = np.arange(int(num_nodes), dtype=np.int64)
        rank = np.zeros(int(num_nodes), dtype=np.int8)
        mask = np.zeros(src.shape[0], dtype=bool)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        added = 0
        m = int(src.shape[0])
        limit = min(
            max(0, int(num_nodes) - 1),
            m if max_edges is None else max(0, int(max_edges)),
        )
        if limit <= 0:
            return mask
        for step in range(m):
            i = (int(offset) + step * int(stride)) % m
            u = int(src[i])
            v = int(dst[i])
            if u == v:
                continue
            ru = find(u)
            rv = find(v)
            if ru == rv:
                continue
            if rank[ru] < rank[rv]:
                parent[ru] = rv
            elif rank[ru] > rank[rv]:
                parent[rv] = ru
            else:
                parent[rv] = ru
                rank[ru] += 1
            mask[i] = True
            added += 1
            if added >= limit:
                break
        return mask

    def _allocate_budget(self, capacities, budget):
        capacities = np.asarray(capacities, dtype=np.int64)
        budget = max(0, min(int(budget), int(capacities.sum())))
        allocation = np.zeros(capacities.shape[0], dtype=np.int64)
        if budget <= 0 or capacities.sum() <= 0:
            return allocation

        raw = (capacities.astype(np.float64) * float(budget)) / float(capacities.sum())
        allocation = np.floor(raw).astype(np.int64)
        allocation = np.minimum(allocation, capacities)
        remaining = budget - int(allocation.sum())
        if remaining <= 0:
            return allocation

        frac = raw - np.floor(raw)
        order = np.lexsort((np.arange(capacities.shape[0]), -frac))
        for cluster_id in order.tolist():
            if remaining <= 0:
                break
            if allocation[cluster_id] >= capacities[cluster_id]:
                continue
            allocation[cluster_id] += 1
            remaining -= 1
        return allocation

    def _grow_local_jobs(self, jobs, workers):
        if not jobs:
            return []
        # Each cluster task owns its inner scoring budget. Do not depend on
        # an environment override: library callers need the same bound.
        outer = min(max(1, int(workers)), len(jobs))
        for job in jobs:
            job["score_kernel_workers"] = max(1, int(workers) // outer)
        if workers <= 1:
            results = []
            added_edges = 0
            swaps_accepted = 0
            selected_edges = 0
            sampled_candidates = 0
            rounds = 0
            target_edges = int(sum(int(job.get("target_edges", 0)) for job in jobs))
            bar = self._progress_bar(
                total=len(jobs),
                desc=f"SCAFFOLD local growth ({len(jobs)} parts)",
                unit="part",
                leave=True,
            )
            try:
                for job in jobs:
                    result = self._grow_local_job(job)
                    results.append(result)
                    added_edges += int(result.get("edges_added", 0))
                    swaps_accepted += int(result.get("swaps_accepted", 0))
                    selected_edges += int(result.get("selected_edges", 0))
                    sampled_candidates += int(result.get("sampled_candidates", 0))
                    rounds += int(result.get("rounds", 0))
                    if bar is not None:
                        bar.update(1)
                        bar.set_postfix(
                            edges_added=added_edges,
                            swaps=swaps_accepted,
                            selected=f"{selected_edges}/{target_edges}",
                            sampled=sampled_candidates,
                            rounds=rounds,
                            refresh=False,
                        )
            finally:
                if bar is not None:
                    bar.close()
            return results
        results = []
        added_edges = 0
        swaps_accepted = 0
        selected_edges = 0
        sampled_candidates = 0
        rounds = 0
        target_edges = int(sum(int(job.get("target_edges", 0)) for job in jobs))
        bar = self._progress_bar(
            total=len(jobs),
            desc=f"SCAFFOLD local growth ({len(jobs)} parts)",
            unit="part",
            leave=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self._grow_local_job, job) for job in jobs]
            try:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    added_edges += int(result.get("edges_added", 0))
                    swaps_accepted += int(result.get("swaps_accepted", 0))
                    selected_edges += int(result.get("selected_edges", 0))
                    sampled_candidates += int(result.get("sampled_candidates", 0))
                    rounds += int(result.get("rounds", 0))
                    if bar is not None:
                        bar.update(1)
                        bar.set_postfix(
                            edges_added=added_edges,
                            swaps=swaps_accepted,
                            selected=f"{selected_edges}/{target_edges}",
                            sampled=sampled_candidates,
                            rounds=rounds,
                            refresh=False,
                        )
            finally:
                if bar is not None:
                    bar.close()
        results.sort(key=lambda result: result["cluster_id"])
        return results

    def _grow_local_job(self, job):
        start = time.perf_counter()
        edge_count = int(job["edge_count"])
        if edge_count == 0:
            return self._empty_result(job, 0.0)

        selected_mask = np.asarray(job["support_mask"], dtype=bool).copy()
        selected_count = int(selected_mask.sum())
        target_edges = min(edge_count, max(selected_count, int(job.get("target_edges", selected_count))))
        sampled_total = 0
        added_total = 0
        rounds = 0

        fast_score = str(getattr(self.owner, "fast_score", ""))
        tree_exact = fast_score == "tree_exact"
        # tree_exact_loop keeps the sampled round loop verbatim and swaps only
        # the ranking, so it needs the tree-prefix index rather than the
        # hop-distance one, but not the one-shot top-k below.
        tree_exact_loop = fast_score == "tree_exact_loop"

        tree_start = time.perf_counter()
        if tree_exact or tree_exact_loop:
            tree_index = ts.build_tree_index(
                int(job["num_nodes"]),
                job["src"][selected_mask],
                job["dst"][selected_mask],
            )
        else:
            support_idx = np.flatnonzero(selected_mask).astype(np.int64, copy=False)
            depth, root, up, _, _ = self._build_tree_index(
                int(job["num_nodes"]),
                job["src"][support_idx],
                job["dst"][support_idx],
                support_idx,
            )
        tree_index_sec = time.perf_counter() - tree_start

        if tree_exact:
            scored, added, passes = self._select_tree_exact(
                job, selected_mask, target_edges, tree_index
            )
            sampled_total += scored
            added_total += added
            selected_count += added
            rounds = passes
            selected_degree = self._selected_degrees(
                job["num_nodes"], job["src"], job["dst"], selected_mask
            )
            return self._finish_local_job(
                job, selected_mask, selected_degree, start, tree_index_sec,
                sampled_total, added_total, rounds, edge_count, target_edges,
            )

        selected_degree = self._selected_degrees(job["num_nodes"], job["src"], job["dst"], selected_mask)
        per_round_add_cap = max(1, int(self.owner.cluster_add_per_round))

        # The one thing that separates tree_exact_loop from tree_distance: the
        # ranking inside each round. Everything else -- the degree-weighted
        # sample, the per-round degree update, the round count -- is identical,
        # so an A/B between the two isolates the objective from the selection.
        exact_scores = None
        batch_scoped_exact = bool(
            tree_exact_loop
            and getattr(self.owner, "algorithm_name", "scaffold_fast")
            == "scaffold_batch"
        )
        if tree_exact_loop and not batch_scoped_exact:
            exact_scores = self._exact_candidate_scores(
                job, selected_mask, tree_index
            )

        while selected_count < target_edges:
            missing_idx = np.flatnonzero(~selected_mask).astype(np.int64, copy=False)
            if missing_idx.size == 0:
                break
            rounds += 1
            sample_idx = self._sample_cluster_indices(
                missing_idx,
                self._cluster_rng(rounds, job["cluster_id"]),
                selected_degree,
                job["src"],
                job["dst"],
            )
            if sample_idx.size == 0:
                break
            sampled_total += int(sample_idx.shape[0])
            if batch_scoped_exact:
                scores = self._exact_candidate_scores(
                    job,
                    selected_mask,
                    tree_index,
                    candidate_ids=sample_idx,
                )[sample_idx]
            elif exact_scores is not None:
                scores = exact_scores[sample_idx]
            else:
                scores = self._tree_distances(job["src"][sample_idx], job["dst"][sample_idx], depth, root, up)
            order = np.lexsort((sample_idx, -scores))
            limit = min(per_round_add_cap, int(target_edges - selected_count), int(order.size))
            if limit <= 0:
                break
            chosen = sample_idx[order[:limit]]
            selected_mask[chosen] = True
            selected_count += int(chosen.shape[0])
            added_total += int(chosen.shape[0])
            np.add.at(selected_degree, job["src"][chosen], 1)
            np.add.at(selected_degree, job["dst"][chosen], 1)

        return self._finish_local_job(
            job, selected_mask, selected_degree, start, tree_index_sec,
            sampled_total, added_total, rounds, edge_count, target_edges,
        )

    def _select_tree_exact(self, job, selected_mask, target_edges, tree_index):
        """One exact scoring pass plus top-k, replacing the sampled round loop.

        The growth phase never rebuilds the tree index, so the ``tree_distance``
        score is static across rounds and the sampled loop only ever approximated
        a ranking that can be computed exactly in a single pass. This scores every
        remaining candidate with the *full* SCAFFOLD objective -- dilation plus
        both congestion terms -- at roughly the cost the old loop paid for hop
        distance alone, because congestion is a tree-prefix computation rather
        than a per-candidate search.

        Returns ``(candidates_scored, edges_added, passes)``.
        """
        budget = int(target_edges) - int(selected_mask.sum())
        candidates = np.flatnonzero(~selected_mask).astype(np.int64, copy=False)
        if budget <= 0 or candidates.size == 0:
            return 0, 0, 0

        scores = self._exact_candidate_scores(job, selected_mask, tree_index)
        limit = min(int(budget), int(candidates.size))
        # Highest score first; ties break on local edge id so runs are reproducible.
        # Disconnected candidates carry +inf and are therefore taken first, which
        # matches the greedy variants treating them as mandatory bridges.
        order = np.lexsort((candidates, -scores[candidates]))
        chosen = candidates[order[:limit]]
        selected_mask[chosen] = True
        return int(candidates.size), int(chosen.size), 1

    def _exact_candidate_scores(
        self,
        job,
        selected_mask,
        tree_index,
        candidate_ids=None,
    ):
        """Full SCAFFOLD score for every candidate or one sampled batch.

        Without ``candidate_ids``, one tree-prefix pass scores the full job and
        is safe to cache. With ids, only that batch contributes to congestion,
        which is the canonical SCAFFOLD-Batch behavior.
        """
        owner = self.owner
        candidate_mask = None
        if candidate_ids is not None:
            candidate_mask = np.zeros(selected_mask.shape[0], dtype=bool)
            candidate_mask[np.asarray(candidate_ids, dtype=np.int64)] = True
        return ts.scaffold_tree_scores(
            int(job["num_nodes"]),
            job["src"],
            job["dst"],
            selected_mask,
            job.get("support_scores"),
            alpha=owner.alpha,
            edge_beta=owner.edge_beta,
            node_beta=owner.node_beta,
            edge_norm_p=owner.edge_norm_p,
            node_norm_q=owner.node_norm_q,
            tree_index=tree_index,
            candidate_mask=candidate_mask,
            weighted_paths=bool(getattr(owner, "weighted_paths", False)),
            workers=int(job.get("score_kernel_workers", 1)),
        )["score"]

    def _finish_local_job(
        self, job, selected_mask, selected_degree, start, tree_index_sec,
        sampled_total, added_total, rounds, edge_count, target_edges,
    ):
        swap_stats = self._refine_local_swaps(job, selected_mask, selected_degree)
        selected_local_idx = np.flatnonzero(selected_mask).astype(np.int64, copy=False)
        total_sec = time.perf_counter() - start
        return {
            "cluster_id": int(job["cluster_id"]),
            "selected_edge_ids": job["edge_ids"][selected_local_idx].astype(np.int64, copy=False),
            "selected_edges": int(selected_local_idx.shape[0]),
            "support_edges": int(job["support_count"]),
            "sampled_candidates": int(sampled_total),
            "edges_added": int(added_total),
            "rounds": int(rounds),
            "swaps_accepted": int(swap_stats["swaps_accepted"]),
            "swap_candidates": int(swap_stats["swap_candidates"]),
            "swap_passes": int(swap_stats["swap_passes"]),
            "swap_sec": float(swap_stats["swap_sec"]),
            "tree_index_sec": float(tree_index_sec),
            "support_sec": float(job.get("support_sec", 0.0)),
            "grow_sec": float(max(0.0, total_sec - tree_index_sec - swap_stats["swap_sec"])),
            "total_sec": float(total_sec + job.get("support_sec", 0.0)),
            "edge_count": edge_count,
            "partition_node_count": int(job.get("partition_node_count", 0)),
            "internal_edges": int(job.get("internal_edges", 0)),
            "owned_crossing_edges": int(job.get("owned_crossing_edges", 0)),
            "target_edges": int(target_edges),
        }

    def _empty_result(self, job, total_sec):
        return {
            "cluster_id": int(job["cluster_id"]),
            "selected_edge_ids": np.empty(0, dtype=np.int64),
            "selected_edges": 0,
            "support_edges": 0,
            "sampled_candidates": 0,
            "edges_added": 0,
            "rounds": 0,
            "swaps_accepted": 0,
            "swap_candidates": 0,
            "swap_passes": 0,
            "swap_sec": 0.0,
            "tree_index_sec": 0.0,
            "support_sec": float(job.get("support_sec", 0.0)),
            "grow_sec": float(total_sec),
            "total_sec": float(total_sec + job.get("support_sec", 0.0)),
            "edge_count": 0,
            "partition_node_count": int(job.get("partition_node_count", 0)),
            "internal_edges": int(job.get("internal_edges", 0)),
            "owned_crossing_edges": int(job.get("owned_crossing_edges", 0)),
            "target_edges": 0,
        }

    def _refine_local_swaps(self, job, selected_mask, selected_degree):
        start = time.perf_counter()
        if not bool(getattr(self.owner, "swap_refine", False)):
            return {
                "swaps_accepted": 0,
                "swap_candidates": 0,
                "swap_passes": 0,
                "swap_sec": 0.0,
            }
        max_swaps = max(0, int(getattr(self.owner, "swap_max_passes", 0)))
        if max_swaps <= 0 or int(job["edge_count"]) == 0:
            return {
                "swaps_accepted": 0,
                "swap_candidates": 0,
                "swap_passes": 0,
                "swap_sec": time.perf_counter() - start,
            }

        support_mask = np.asarray(job["support_mask"], dtype=bool)
        accepted = 0
        sampled_total = 0
        passes = 0
        stalled = 0
        patience = max(1, int(getattr(self.owner, "swap_no_improve_patience", 1)))

        while accepted < max_swaps and stalled < patience:
            missing_idx = np.flatnonzero(~selected_mask).astype(np.int64, copy=False)
            if missing_idx.size == 0:
                break
            selected_idx = np.flatnonzero(selected_mask).astype(np.int64, copy=False)
            if selected_idx.size == 0:
                break

            passes += 1
            depth, root, up, parent, parent_edge = self._build_tree_index(
                int(job["num_nodes"]),
                job["src"][selected_idx],
                job["dst"][selected_idx],
                selected_idx,
            )
            sample_idx = self._sample_indices(
                missing_idx,
                self._cluster_rng(10_000 + passes, job["cluster_id"]),
                selected_degree,
                job["src"],
                job["dst"],
                str(getattr(self.owner, "swap_sampling_mode", "random")),
                max(1, int(getattr(self.owner, "swap_sample_size", 256))),
            )
            if sample_idx.size == 0:
                break
            sampled_total += int(sample_idx.shape[0])

            scores = self._tree_distances(job["src"][sample_idx], job["dst"][sample_idx], depth, root, up)
            order = np.lexsort((sample_idx, -scores))
            swapped_this_pass = False
            for order_pos in order:
                add_idx = int(sample_idx[order_pos])
                if int(scores[order_pos]) <= 2:
                    continue
                path_edges = self._tree_path_edges(
                    int(job["src"][add_idx]),
                    int(job["dst"][add_idx]),
                    depth,
                    root,
                    parent,
                    parent_edge,
                )
                if not path_edges:
                    continue
                removable = [
                    int(edge_idx)
                    for edge_idx in path_edges
                    if selected_mask[int(edge_idx)] and not support_mask[int(edge_idx)]
                ]
                if not removable:
                    continue
                cycle_sample = int(getattr(self.owner, "swap_cycle_sample_size", 0))
                if cycle_sample > 0 and len(removable) > cycle_sample:
                    removable = sorted(
                        removable,
                        key=lambda idx: (
                            -(int(selected_degree[int(job["src"][idx])]) + int(selected_degree[int(job["dst"][idx])])),
                            idx,
                        ),
                    )[:cycle_sample]
                remove_idx = max(
                    removable,
                    key=lambda idx: (
                        int(selected_degree[int(job["src"][idx])]) + int(selected_degree[int(job["dst"][idx])]),
                        -idx,
                    ),
                )
                if remove_idx == add_idx:
                    continue
                selected_mask[add_idx] = True
                selected_mask[remove_idx] = False
                np.add.at(
                    selected_degree,
                    np.asarray([job["src"][add_idx], job["dst"][add_idx]], dtype=np.int64),
                    1,
                )
                np.add.at(
                    selected_degree,
                    np.asarray([job["src"][remove_idx], job["dst"][remove_idx]], dtype=np.int64),
                    -1,
                )
                accepted += 1
                swapped_this_pass = True
                break

            if swapped_this_pass:
                stalled = 0
            else:
                stalled += 1

        return {
            "swaps_accepted": int(accepted),
            "swap_candidates": int(sampled_total),
            "swap_passes": int(passes),
            "swap_sec": time.perf_counter() - start,
        }

    def _selected_degrees(self, num_nodes, src, dst, selected_mask):
        degree = np.zeros(int(num_nodes), dtype=np.int64)
        selected_idx = np.flatnonzero(selected_mask)
        if selected_idx.size:
            np.add.at(degree, src[selected_idx], 1)
            np.add.at(degree, dst[selected_idx], 1)
        return degree

    def _sample_cluster_indices(self, candidate_idx, rng, selected_degree, src, dst):
        return self._sample_indices(
            candidate_idx,
            rng,
            selected_degree,
            src,
            dst,
            str(self.owner.sampling_mode),
            max(1, int(self.owner.sample_size)),
        )

    def _sample_indices(self, candidate_idx, rng, selected_degree, src, dst, mode, sample_size):
        if candidate_idx.size == 0:
            return candidate_idx

        mode = str(mode)
        if mode == "full":
            return candidate_idx

        k = min(max(1, int(sample_size)), int(candidate_idx.size))
        if candidate_idx.size <= k:
            return candidate_idx

        if mode == "random":
            return rng.choice(candidate_idx, size=k, replace=False)

        if mode == "weighted":
            src_deg = np.maximum(selected_degree[src[candidate_idx]], 1)
            dst_deg = np.maximum(selected_degree[dst[candidate_idx]], 1)
            weights = (1.0 / src_deg) + (1.0 / dst_deg)
            weight_sum = float(weights.sum())
            if weight_sum <= 0.0 or not np.isfinite(weight_sum):
                return rng.choice(candidate_idx, size=k, replace=False)
            return rng.choice(candidate_idx, size=k, replace=False, p=weights / weight_sum)

        raise ValueError(f"Unknown sampling_mode: {mode}")

    def _cluster_rng(self, round_idx, cluster_id):
        base_seed = (0 if self.owner.seed is None else int(self.owner.seed)) & 0xFFFFFFFF
        seed_sequence = np.random.SeedSequence([base_seed, int(round_idx), int(cluster_id)])
        return np.random.default_rng(seed_sequence)

    def _build_tree_index(self, num_nodes, tree_src, tree_dst, tree_edge_idx):
        num_nodes = int(num_nodes)
        levels = max(1, int(math.ceil(math.log2(max(2, num_nodes)))) + 1)
        if njit is not None:
            tree_src = np.ascontiguousarray(tree_src, dtype=np.int64)
            tree_dst = np.ascontiguousarray(tree_dst, dtype=np.int64)
            tree_edge_idx = np.ascontiguousarray(tree_edge_idx, dtype=np.int64)
            return _build_tree_index_numba(
                num_nodes,
                tree_src,
                tree_dst,
                tree_edge_idx,
                int(levels),
            )
        adj = [[] for _ in range(num_nodes)]
        for u, v, edge_idx in zip(tree_src.tolist(), tree_dst.tolist(), tree_edge_idx.tolist()):
            u = int(u)
            v = int(v)
            edge_idx = int(edge_idx)
            adj[u].append((v, edge_idx))
            adj[v].append((u, edge_idx))

        parent = np.arange(num_nodes, dtype=np.int64)
        parent_edge = np.full(num_nodes, -1, dtype=np.int64)
        depth = np.zeros(num_nodes, dtype=np.int32)
        root = np.arange(num_nodes, dtype=np.int64)
        seen = np.zeros(num_nodes, dtype=bool)
        for start in range(num_nodes):
            if seen[start]:
                continue
            seen[start] = True
            root[start] = start
            stack = [start]
            while stack:
                node = stack.pop()
                for nbr, edge_idx in adj[node]:
                    if seen[nbr]:
                        continue
                    seen[nbr] = True
                    parent[nbr] = node
                    parent_edge[nbr] = edge_idx
                    depth[nbr] = depth[node] + 1
                    root[nbr] = start
                    stack.append(nbr)

        up = np.empty((levels, num_nodes), dtype=np.int64)
        up[0] = parent
        for lvl in range(1, levels):
            up[lvl] = up[lvl - 1][up[lvl - 1]]
        return depth, root, up, parent, parent_edge

    def _tree_path_edges(self, u, v, depth, root, parent, parent_edge):
        u = int(u)
        v = int(v)
        if u < 0 or v < 0 or u >= root.shape[0] or v >= root.shape[0]:
            return []
        if int(root[u]) != int(root[v]):
            return []

        path_edges = []
        a = u
        b = v
        while int(depth[a]) > int(depth[b]):
            edge_idx = int(parent_edge[a])
            if edge_idx >= 0:
                path_edges.append(edge_idx)
            next_a = int(parent[a])
            if next_a == a:
                break
            a = next_a
        while int(depth[b]) > int(depth[a]):
            edge_idx = int(parent_edge[b])
            if edge_idx >= 0:
                path_edges.append(edge_idx)
            next_b = int(parent[b])
            if next_b == b:
                break
            b = next_b
        while a != b:
            edge_a = int(parent_edge[a])
            edge_b = int(parent_edge[b])
            if edge_a >= 0:
                path_edges.append(edge_a)
            if edge_b >= 0:
                path_edges.append(edge_b)
            next_a = int(parent[a])
            next_b = int(parent[b])
            if next_a == a and next_b == b:
                break
            a = next_a
            b = next_b
        return path_edges

    def _tree_distances(self, src, dst, depth, root, up):
        if njit is not None:
            return _tree_distances_numba(src, dst, depth, root, up)
        out = np.empty(src.shape[0], dtype=np.int32)
        for i, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
            if root[u] != root[v]:
                out[i] = int(depth[u]) + int(depth[v]) + 1
            else:
                out[i] = abs(int(depth[u]) - int(depth[v])) + 1
        return out

    def _edge_owner_balance(self, loads):
        loads = np.asarray(loads, dtype=np.int64)
        if loads.size == 0:
            return {"min": 0, "median": 0.0, "max": 0, "balance_ratio": 0.0}
        mean_load = float(loads.mean())
        max_load = int(loads.max())
        return {
            "min": int(loads.min()),
            "median": float(np.median(loads)),
            "max": max_load,
            "balance_ratio": float(max_load / mean_load) if mean_load > 0.0 else 0.0,
        }

    def _partition_rows(self, jobs, results):
        by_cluster = {}
        if results is not None:
            by_cluster = {int(result["cluster_id"]): result for result in results}
        rows = []
        for job in sorted(jobs, key=lambda item: int(item["cluster_id"])):
            cluster_id = int(job["cluster_id"])
            result = by_cluster.get(cluster_id, {})
            selected_edges = result.get("selected_edges")
            if selected_edges is None and results is None:
                selected_edges = int(job.get("edge_count", 0))
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "node_count": int(job.get("partition_node_count", 0)),
                    "internal_edges": int(job.get("internal_edges", 0)),
                    "owned_crossing_edges": int(job.get("owned_crossing_edges", 0)),
                    "total_owned_edges": int(job.get("edge_count", 0)),
                    "support_edges": int(result.get("support_edges", job.get("support_count", 0))),
                    "target_edges": int(result.get("target_edges", job.get("target_edges", 0))),
                    "selected_edges": int(selected_edges or 0),
                }
            )
        return rows

    def _summarize_local_profiles(self, results):
        keys = ("support_sec", "tree_index_sec", "grow_sec", "swap_sec", "total_sec")
        summary = {}
        for key in keys:
            values = [float(result.get(key, 0.0)) for result in results]
            summary[f"{key}_sum"] = float(sum(values))
            summary[f"{key}_max"] = float(max(values)) if values else 0.0
        summary["swaps_accepted"] = int(sum(int(result.get("swaps_accepted", 0)) for result in results))
        summary["swap_candidates"] = int(sum(int(result.get("swap_candidates", 0)) for result in results))
        summary["swap_passes"] = int(sum(int(result.get("swap_passes", 0)) for result in results))
        edge_counts = [int(result.get("edge_count", 0)) for result in results]
        target_counts = [int(result.get("target_edges", 0)) for result in results]
        summary["edge_count_min"] = int(min(edge_counts)) if edge_counts else 0
        summary["edge_count_max"] = int(max(edge_counts)) if edge_counts else 0
        summary["target_count_min"] = int(min(target_counts)) if target_counts else 0
        summary["target_count_max"] = int(max(target_counts)) if target_counts else 0
        return summary
