"""Shared base class for SCAFFOLD variants.

Extends :class:`JointDilationCongestionSparsifier` but replaces the slow
``networkx.minimum_spanning_tree`` / ``networkx.random_spanning_tree`` calls
with the direct union-find helpers in :mod:`.spanning_tree`.
"""

from collections import defaultdict
import time

import networkx as nx

from scaffold_gnn.sparsifiers.joint_dilation_congestion import JointDilationCongestionSparsifier

from .clustering import build_node_clusters
from .parallel_utils import auto_worker_count
from .path_backend import UnweightedPathBackend
from .spanning_tree import (
    build_fast_weighted_spanning_forest_nx,
    build_random_spanning_forest_nx,
    build_spanning_forest_nx,
)


DETERMINISTIC_INIT_SUPPORTS = {"mst", "maxst", "fast_mst", "fast_maxst"}


class ScaffoldBaseSparsifier(JointDilationCongestionSparsifier):
    """Shared configuration and helpers for SCAFFOLD-Greedy / -Heap / -Fast."""

    def __init__(
        self,
        *args,
        cluster_count=64,
        cluster_method="metis",
        parallel_clusters=0,
        cluster_add_per_round=16,
        cluster_cache_dir=None,
        cluster_strategy=None,
        local_update_radius=1,
        dirty_limit=0,
        parallel_workers=0,
        resparsify_every=1,
        resparsify_reseed=True,
        support_budget_mode="full_then_random_trim",
        **kwargs,
    ):
        super().__init__(
            *args,
            support_budget_mode=support_budget_mode,
            **kwargs,
        )
        self.cluster_count = max(1, int(cluster_count))
        self.cluster_method = str(cluster_method)
        self.parallel_clusters = max(0, int(parallel_clusters))
        self.cluster_add_per_round = max(1, int(cluster_add_per_round))
        self.cluster_cache_dir = cluster_cache_dir
        self.local_update_radius = max(0, int(local_update_radius))
        self.dirty_limit = max(0, int(dirty_limit))
        self.parallel_workers = auto_worker_count(parallel_workers, cap=8)
        self.last_cluster_stats = {}
        self.resparsify_every = max(0, int(resparsify_every))
        self.resparsify_reseed = bool(resparsify_reseed)
        self._resparsify_state = None
        self._deterministic_support_cache = None
        self._last_init_support_cache = {
            "cacheable": False,
            "hit": False,
            "kind": None,
            "time_sec": 0.0,
        }

    def _build_clusters(self, G):
        weight_key = "weight" if nx.is_weighted(G) else None
        return build_node_clusters(
            G,
            method=self.cluster_method,
            cluster_count=self.cluster_count,
            seed=self.seed,
            weight_key=weight_key,
            cluster_cache_dir=self.cluster_cache_dir,
        )

    def _maybe_path_backend(self, H, is_weighted):
        """Compiled BFS helper for H, or ``None`` when it does not apply.

        Only unweighted graphs labelled ``0..n-1`` qualify; everything else
        keeps the NetworkX path routines.
        """
        if is_weighted:
            return None
        return UnweightedPathBackend.maybe_build(H)

    def _paths_for_candidates(
        self,
        H,
        candidate_edges,
        weight_key,
        is_weighted,
        path_backend=None,
        workers=None,
    ):
        """Shortest path per candidate, keyed by ``(u, v)``.

        With a ``path_backend`` this is one compiled BFS per *source* rather
        than one NetworkX search per candidate, and the searches release the
        GIL so an enclosing cluster pool actually runs them in parallel.
        Shared by Greedy and Batch, which score identical objectives.
        """
        targets_by_source = defaultdict(set)
        for u, v in candidate_edges:
            targets_by_source[u].add(v)

        paths_by_edge = {}
        if path_backend is not None:
            if workers is None:
                workers = self.parallel_workers
            by_source = path_backend.paths_for_sources(
                targets_by_source, workers=workers
            )
            used = min(int(workers), len(targets_by_source))
            if not path_backend.parallel_available:
                used = 1
            self._path_workers_used = max(getattr(self, "_path_workers_used", 1), used)
            for source, paths in by_source.items():
                for target, path in paths.items():
                    paths_by_edge[(source, target)] = path
            return paths_by_edge

        for source, targets in targets_by_source.items():
            if source not in H:
                paths = {}
            elif is_weighted:
                paths = nx.single_source_dijkstra_path(H, source, weight=weight_key)
            else:
                paths = nx.single_source_shortest_path(H, source)

            for target in targets:
                path = paths.get(target)
                if path is not None:
                    paths_by_edge[(source, target)] = path
        return paths_by_edge

    def _build_init_support(self, G, max_edges=None):
        """Fast base-support builder.

        For plain (``mst`` / ``maxst``) initializers we skip the NetworkX
        Kruskal implementation and drive our own union-find directly on ``G``.
        For ``randst`` (a.k.a. ``fast_randst``) we run Kruskal on random edge
        priorities -- structurally similar to a uniform random spanning tree
        but orders of magnitude faster than Wilson's algorithm. Higher-quality
        initializers (glst/slst/randspt/llst) still defer to the shared
        implementation from :class:`JointDilationCongestionSparsifier`.
        """
        if isinstance(self.init_support, nx.Graph):
            self._last_init_support_cache = {
                "cacheable": False,
                "hit": False,
                "kind": "provided_graph",
                "time_sec": 0.0,
            }
            return self.init_support.copy()
        if self.init_support in ("randst", "fast_randst"):
            self._last_init_support_cache = {
                "cacheable": False,
                "hit": False,
                "kind": str(self.init_support),
                "time_sec": 0.0,
            }
            return build_random_spanning_forest_nx(
                G,
                seed=self._active_seed(),
                max_edges=max_edges,
            )
        if self.init_support in DETERMINISTIC_INIT_SUPPORTS:
            cache_key = self._networkx_support_cache_key(G, max_edges=max_edges)
            cached = self._deterministic_support_cache
            if (
                isinstance(cached, dict)
                and cached.get("kind") == "networkx_graph"
                and cached.get("key") == cache_key
                and isinstance(cached.get("graph"), nx.Graph)
            ):
                self._last_init_support_cache = {
                    "cacheable": True,
                    "hit": True,
                    "kind": "networkx_graph",
                    "time_sec": 0.0,
                    "support_edges": cached["graph"].number_of_edges(),
                }
                return cached["graph"].copy()

            start = time.perf_counter()
            weight_key = "weight" if nx.is_weighted(G) else None
            maximum = self.init_support in ("maxst", "fast_maxst")
            if self.init_support in ("fast_mst", "fast_maxst"):
                support = build_fast_weighted_spanning_forest_nx(
                    G,
                    weight_key=weight_key,
                    maximum=maximum,
                    bucket_count=self.fast_tree_buckets,
                    max_edges=max_edges,
                    parallel_workers=self.parallel_workers,
                )
            else:
                support = build_spanning_forest_nx(
                    G,
                    weight_key=weight_key,
                    maximum=maximum,
                    max_edges=max_edges,
                )
            elapsed = time.perf_counter() - start
            self._deterministic_support_cache = {
                "kind": "networkx_graph",
                "key": cache_key,
                "init_support": str(self.init_support),
                "graph": support.copy(),
            }
            self._last_init_support_cache = {
                "cacheable": True,
                "hit": False,
                "kind": "networkx_graph",
                "time_sec": elapsed,
                "support_edges": support.number_of_edges(),
            }
            return support
        self._last_init_support_cache = {
            "cacheable": False,
            "hit": False,
            "kind": str(self.init_support),
            "time_sec": 0.0,
        }
        return super()._build_init_support(G, max_edges=max_edges)

    def _networkx_support_cache_key(self, G, max_edges=None):
        return (
            "networkx_graph",
            str(self.init_support),
            str(getattr(self, "support_weight_method", "uniform")),
            None if max_edges is None else int(max_edges),
            int(G.number_of_nodes()),
            int(G.number_of_edges()),
            bool(G.is_directed()),
            bool(nx.is_weighted(G)),
            (
                int(self.fast_tree_buckets)
                if str(self.init_support) in ("fast_mst", "fast_maxst")
                else None
            ),
        )

    def _is_deterministic_init_support_cacheable(self):
        return str(self.init_support) in DETERMINISTIC_INIT_SUPPORTS

    def export_deterministic_support_cache(self):
        """Return a CPU-only payload for async workers, when one is reusable."""
        tensor_cache = getattr(self, "_tensor_deterministic_support_cache", None)
        if isinstance(tensor_cache, dict) and tensor_cache.get("kind") == "tensor_edge_ids":
            try:
                import torch

                edge_ids = tensor_cache.get("support_edge_ids")
                edge_ids = torch.as_tensor(edge_ids, dtype=torch.long).detach().cpu().contiguous()
                try:
                    edge_ids.share_memory_()
                except Exception:
                    pass
                payload = {
                    "kind": "tensor_edge_ids",
                    "init_support": str(tensor_cache.get("init_support", self.init_support)),
                    "key": tensor_cache.get("key"),
                    "support_edge_ids": edge_ids,
                    "bridge_stats": dict(tensor_cache.get("bridge_stats", {})),
                }
                return payload
            except Exception:
                return None

        cache = self._deterministic_support_cache
        if (
            not isinstance(cache, dict)
            or cache.get("kind") != "networkx_graph"
            or not isinstance(cache.get("graph"), nx.Graph)
        ):
            return None

        try:
            import torch

            edges = list(cache["graph"].edges())
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            try:
                edge_index.share_memory_()
            except Exception:
                pass
            return {
                "kind": "networkx_edge_index",
                "init_support": str(cache.get("init_support", self.init_support)),
                "key": cache.get("key"),
                "num_nodes": int(cache["graph"].number_of_nodes()),
                "support_edge_index": edge_index,
            }
        except Exception:
            return None

    def load_deterministic_support_cache(self, payload):
        """Install a deterministic support cache exported by the initial run."""
        if not isinstance(payload, dict):
            return False
        if str(payload.get("init_support", self.init_support)) != str(self.init_support):
            return False

        kind = str(payload.get("kind"))
        if kind == "tensor_edge_ids":
            try:
                import numpy as np

                edge_ids = payload.get("support_edge_ids")
                if hasattr(edge_ids, "detach"):
                    edge_ids = edge_ids.detach().cpu().numpy()
                edge_ids = np.asarray(edge_ids, dtype=np.int64)
                self._tensor_deterministic_support_cache = {
                    "kind": "tensor_edge_ids",
                    "init_support": str(self.init_support),
                    "key": payload.get("key"),
                    "support_edge_ids": np.sort(edge_ids.astype(np.int64, copy=False)),
                    "bridge_stats": dict(payload.get("bridge_stats", {})),
                }
                return True
            except Exception:
                return False

        if kind == "networkx_edge_index":
            try:
                edge_index = payload.get("support_edge_index")
                if hasattr(edge_index, "detach"):
                    edge_index = edge_index.detach().cpu()
                num_nodes = int(payload.get("num_nodes", 0))
                support = nx.Graph()
                support.add_nodes_from(range(num_nodes))
                if edge_index is not None:
                    edges = edge_index.t().tolist()
                    support.add_edges_from((int(u), int(v)) for u, v in edges)
                self._deterministic_support_cache = {
                    "kind": "networkx_graph",
                    "init_support": str(self.init_support),
                    "key": payload.get("key"),
                    "graph": support,
                }
                return True
            except Exception:
                return False
        return False

    def _active_seed(self):
        """Seed used for stochastic init this call (resparsify-aware)."""
        state = self._resparsify_state
        if state is None:
            return self.seed
        return state.get("seed", self.seed)

    # ------------------------------------------------------------------
    # Resparsify: repeat sparsification with a fresh seed.
    # ------------------------------------------------------------------
    def cache_source(self, data_or_graph):
        """Remember the pre-sparsification input for later ``resparsify`` calls.

        ``main.py`` invokes this once with the ``pyg_data`` handed to
        :meth:`sparsify` so the training loop can request a fresh sparse graph
        every ``resparsify_every`` epochs without re-running any of the
        upstream data plumbing.
        """
        self._resparsify_source = data_or_graph

    def resparsify(self, seed=None):
        """Rebuild the sparse graph with a fresh seed.

        The stored source data is passed through :meth:`sparsify` again. When
        ``resparsify_reseed`` is ``True`` (the default) the internal RNGs are
        reseeded so that stochastic components (fast RandST, per-cluster
        sampling, swap search) produce a different sample. Returns whatever
        :meth:`sparsify` returned -- for scaffold variants that is either a
        :class:`torch_geometric.data.Data` (when the input was PyG) or an
        :class:`networkx.Graph`.
        """
        source = getattr(self, "_resparsify_source", None)
        if source is None:
            raise RuntimeError(
                "resparsify() requires cache_source(data) to be called first."
            )
        import random

        import torch

        active_seed = seed if seed is not None else self.seed
        prev_state = self._resparsify_state
        self._resparsify_state = {"seed": active_seed}
        if self.resparsify_reseed:
            self._rng = random.Random(active_seed)
            if active_seed is not None:
                self._torch_gen = torch.Generator()
                self._torch_gen.manual_seed(int(active_seed))
        try:
            return self.sparsify(source)
        finally:
            self._resparsify_state = prev_state
