"""SCAFFOLD-Sample: per-epoch weighted sampling from a precomputed weight vector.

All structural work happens once, offline (:mod:`.sample_weights`). Each epoch
then costs one uniform draw plus one ``searchsorted``:

* a **complete spanning forest is unioned in unconditionally**, so the emitted
  graph has exactly the connected components of ``G`` with probability 1 -- not
  in expectation, not with high probability (Invariant C);
* the remaining ``k - |B|`` edges are drawn by **systematic pi-ps** over a
  tree-locality ordering, which gives exactly ``k`` edges, exact marginal
  inclusion probabilities, and the tree-local spread that the adaptive greedy
  produces.

Output is plain unweighted topology; GCN renormalises on the sampled graph
itself, so the training contract is identical to the other ``scaffold-*``
identities. See ``Brainstrom/others/notes/2026-08-17_plan_scaffold_sampling.md`` sections
3, 4.0 and 8.4.
"""

import math
import os
import time

import numpy as np
import torch
from torch_geometric.data import Data

from . import sample_weights as sw
from . import tree_score as ts
from .common import ScaffoldBaseSparsifier
from .sampling import systematic_positions
from .spanning_tree import canonical_sample_backbone


class ScaffoldSampleSparsifier(ScaffoldBaseSparsifier):
    """Precompute-once, sample-per-epoch SCAFFOLD."""

    def __init__(
        self,
        *args,
        sample_artifact_path=None,
        sample_backbone="fixed-maxst",
        sample_scheme="systematic",
        sample_tree_count=sw.DEFAULT_TREE_COUNT,
        sample_lambda=sw.DEFAULT_AGGREGATE_LAMBDA,
        sample_weight_mode="normalized-mixture",
        sample_mix_alpha=0.5,
        sample_assert_connectivity="auto",
        sample_allow_build=True,
        sample_dataset=None,
        sample_scratch_root=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sample_artifact_path = (
            str(sample_artifact_path) if sample_artifact_path else None
        )
        # Accepts the forest spellings too (fixed-maxsf, fixed-slsf,
        # rotate-randsf, bare fixed-sf); the tree spelling stays canonical so
        # the literal comparisons below and every existing config_tag are
        # unchanged.
        self.sample_backbone = canonical_sample_backbone(sample_backbone)
        if self.sample_backbone not in ("fixed-maxst", "fixed-slst", "rotate-randst"):
            raise ValueError(
                "sample_backbone must be one of 'fixed-maxst' (= fixed-maxsf), "
                "'fixed-slst' (= fixed-slsf) or 'rotate-randst' (= rotate-randsf), "
                f"got {sample_backbone!r}"
            )
        self.sample_scheme = str(sample_scheme).strip().lower()
        if self.sample_scheme not in ("systematic", "alias"):
            raise ValueError("sample_scheme must be 'systematic' or 'alias'")
        self.sample_tree_count = max(1, int(sample_tree_count))
        self.sample_lambda = float(sample_lambda)
        self.sample_weight_mode = str(sample_weight_mode)
        self.sample_mix_alpha = float(sample_mix_alpha)
        if self.sample_weight_mode not in ('legacy', 'normalized-mixture'):
            raise ValueError('sample_weight_mode must be legacy or normalized-mixture')
        if not math.isfinite(self.sample_mix_alpha) or not 0. <= self.sample_mix_alpha <= 1.:
            raise ValueError('sample_mix_alpha must lie in [0, 1]')
        self.sample_assert_connectivity = str(sample_assert_connectivity).lower()
        self.sample_allow_build = bool(sample_allow_build)
        self.sample_dataset = sample_dataset
        self.sample_scratch_root = sample_scratch_root

        self._artifact = None
        self._artifact_key = None
        self._plan = None
        self._plan_key = None
        self._draw_index = 0

    # ------------------------------------------------------------------
    # graph plumbing
    # ------------------------------------------------------------------
    def _canonical_undirected(self, data, num_nodes):
        """Canonical ``src < dst`` undirected edge list, loops removed.

        Memoised: ``resparsify`` re-sparsifies the *same* source graph every
        epoch, so this must not be recomputed per draw. The dedup encodes each
        pair as a single int64 and uses a 1-D ``np.unique``; ``torch.unique(dim=1)``
        measured ~50x slower and would dominate the whole per-epoch budget.
        """
        edge_index = data.edge_index
        cache = getattr(self, "_edges_cache", None)
        key = (id(data), int(num_nodes), int(edge_index.shape[1]))
        if cache is not None and cache[0] == key:
            return cache[1], cache[2]

        edge_index = edge_index.detach().cpu()
        src = edge_index[0].numpy().astype(np.int64, copy=False)
        dst = edge_index[1].numpy().astype(np.int64, copy=False)
        keep = src != dst
        src, dst = src[keep], dst[keep]
        if bool(getattr(data, "edge_index_is_undirected_unique", False)):
            # The flag rules out duplicates, so the dedup can be skipped -- but
            # not the ordering. The artifact's weight and backbone arrays are
            # indexed positionally against the order precompute_scaffold_weights
            # wrote, which is always ascending ``lo * n + hi``. Callers reach
            # here through main.py's symmetric fast path, whose order is the
            # loader's: already canonical for most datasets, but not for
            # Planetoid, where returning it verbatim yields the same edge set in
            # a different sequence. graph_fingerprint rejects that (and without
            # the check it would silently misalign every edge weight), so sort
            # into the same order rather than trusting the producer's.
            lo = np.minimum(src, dst)
            hi = np.maximum(src, dst)
            order = np.argsort(lo * np.int64(num_nodes) + hi, kind="stable")
            out_src = np.ascontiguousarray(lo[order])
            out_dst = np.ascontiguousarray(hi[order])
        else:
            lo = np.minimum(src, dst)
            hi = np.maximum(src, dst)
            codes = np.unique(lo * np.int64(num_nodes) + hi)
            out_src = (codes // np.int64(num_nodes)).astype(np.int64, copy=False)
            out_dst = (codes % np.int64(num_nodes)).astype(np.int64, copy=False)
        self._edges_cache = (key, out_src, out_dst)
        return out_src, out_dst

    def _edge_weights(self, data, src, dst):
        if str(getattr(self, "support_weight_method", "uniform")) == "uniform":
            return None
        scores = self._compute_feature_edge_scores(
            getattr(data, "x", None),
            torch.from_numpy(src),
            torch.from_numpy(dst),
        )
        return scores.numpy().astype(np.float64, copy=False)

    def _resolve_artifact_path(self, dataset):
        if self.sample_artifact_path:
            return self.sample_artifact_path
        tag = sw.config_tag(
            self.sample_tree_count,
            self.sample_lambda,
            self.alpha,
            self.edge_beta,
            self.node_beta,
            self.edge_norm_p,
            self.node_norm_q,
            getattr(self, "support_weight_method", "uniform"),
            weight_mode=self.sample_weight_mode,
            # The deterministic forest defines perm and the support scores, so
            # a slst artifact must not share a cache entry with a maxst one.
            det_support=("slst" if self.sample_backbone == "fixed-slst" else "maxst"),
        )
        return sw.default_artifact_path(
            dataset or "graph", tag, self.sample_scratch_root
        )

    def _get_artifact(self, num_nodes, src, dst, weight):
        key = (int(num_nodes), int(src.size), sw.graph_fingerprint(num_nodes, src, dst))
        if self._artifact is not None and self._artifact_key == key:
            return self._artifact

        path = self._resolve_artifact_path(self.sample_dataset)
        artifact = None
        if path and os.path.isfile(path):
            try:
                candidate = sw.load_artifact(path)
                sw.validate_artifact(candidate, num_nodes, src, dst, path=path)
                if self.sample_weight_mode == 'normalized-mixture':
                    sw.validate_weight_components(candidate)
                artifact = candidate
                print(f"[ScaffoldSample] loaded artifact {path}", flush=True)
            except Exception as exc:  # stale or foreign artifact
                if not self.sample_allow_build:
                    raise
                print(
                    f"[ScaffoldSample] ignoring unusable artifact {path}: {exc}",
                    flush=True,
                )

        if artifact is None:
            if not self.sample_allow_build:
                raise FileNotFoundError(
                    "scaffold-sample artifact is missing and --scaffold_sample_allow_build "
                    f"is off: {path}. Run scripts/precompute_scaffold_weights.py first."
                )
            artifact = sw.build_artifact(
                num_nodes,
                src,
                dst,
                weight,
                tree_count=self.sample_tree_count,
                aggregate_lambda=self.sample_lambda,
                alpha=self.alpha,
                edge_beta=self.edge_beta,
                node_beta=self.node_beta,
                edge_norm_p=self.edge_norm_p,
                node_norm_q=self.node_norm_q,
                seed=self.seed or 0,
                workers=self.parallel_workers,
                verbose=self.verbose,
                det_support=("slst" if self.sample_backbone == "fixed-slst"
                             else "maxst"),
            )
            if path:
                try:
                    sw.save_artifact(path, artifact)
                    print(f"[ScaffoldSample] wrote artifact {path}", flush=True)
                except Exception as exc:  # cache is an optimisation, not a requirement
                    print(f"[ScaffoldSample] could not cache artifact: {exc}", flush=True)

        self._artifact = artifact
        self._artifact_key = key
        return artifact

    # ------------------------------------------------------------------
    # sampling plan (built once per ratio/backbone, reused every epoch)
    # ------------------------------------------------------------------
    def _backbone_ids(self, artifact, rotation):
        if self.sample_backbone in ("fixed-maxst", "fixed-slst"):
            return np.asarray(artifact["det_forest_edge_ids"], dtype=np.int64), "det"
        return sw.forest_edge_ids(artifact, rotation), f"rand{rotation}"

    def _build_plan(self, artifact, target_edges, rotation):
        m = int(artifact["num_edges"])
        pi = np.asarray(artifact["pi"], dtype=np.float64)
        perm = np.asarray(artifact["perm"], dtype=np.int64)
        mandatory = np.asarray(artifact["mandatory"], dtype=bool)
        backbone, backbone_tag = self._backbone_ids(artifact, rotation)

        forced = np.zeros(m, dtype=bool)
        forced[backbone] = True
        forced |= mandatory

        trimmed = False
        if int(forced.sum()) > target_edges:
            # Budget below the connectivity floor: no method can stay connected.
            keep = np.flatnonzero(forced)
            rng = np.random.default_rng(int(self.seed or 0))
            keep = rng.permutation(keep)[:target_edges]
            forced = np.zeros(m, dtype=bool)
            forced[keep] = True
            trimmed = True

        remaining = int(target_edges - forced.sum())
        pool_mask = ~forced
        pool_order = perm[pool_mask[perm]]  # tree-locality order, restricted to the pool
        if self.sample_weight_mode == 'normalized-mixture':
            sw.validate_weight_components(artifact)
            pool_weights = sw.normalized_mixture_weights(
                artifact['tree_frequency'], artifact['support_score'], pool_order, self.sample_mix_alpha)
        else:
            pool_weights = pi[pool_order]
        p = sw.cap_and_renormalize(pool_weights, remaining)

        return {
            "forced": np.flatnonzero(forced).astype(np.int64, copy=False),
            "pool_order": pool_order,
            "p": p,
            "cum": np.cumsum(p),
            "remaining": remaining,
            "target_edges": int(target_edges),
            "backbone_tag": backbone_tag,
            "trimmed": trimmed,
        }

    def _draw_pool(self, plan, rng):
        self._draw_workers_used = 1
        remaining = plan["remaining"]
        if remaining <= 0:
            return np.zeros(0, dtype=np.int64)
        pool_order = plan["pool_order"]

        if self.sample_scheme == "alias":
            from scaffold_gnn.sparsifiers.effective_resistance import (
                _alias_table,
                _sample_alias_until_unique,
            )

            p = plan["p"]
            total = float(p.sum())
            probs = p / total if total > 0 else np.full(p.size, 1.0 / max(1, p.size))
            threshold, alias = _alias_table(probs)
            picks, _, _ = _sample_alias_until_unique(
                threshold, alias, remaining, int(rng.integers(0, 2**31 - 1)), 1
            )
            return pool_order[picks]

        cum = plan["cum"]
        offset = float(rng.random())
        pos, self._draw_workers_used = systematic_positions(
            cum, offset, remaining, self.parallel_workers
        )
        pos = np.unique(pos)
        if pos.size < remaining:
            # Float-boundary degeneracy (many p_e == 1). Top up deterministically
            # with the highest-weight unselected pool entries so |E| stays exact.
            taken = np.zeros(cum.size, dtype=bool)
            taken[pos] = True
            spare = np.flatnonzero(~taken)
            extra = spare[np.argsort(-plan["p"][spare], kind="stable")]
            pos = np.concatenate((pos, extra[: remaining - pos.size]))
        return pool_order[pos]

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------
    def sparsify(self, data_or_graph):
        if not isinstance(data_or_graph, Data):
            raise TypeError(
                "ScaffoldSampleSparsifier expects a torch_geometric Data object"
            )
        start = time.perf_counter()
        data = data_or_graph
        num_nodes = int(data.num_nodes)
        src, dst = self._canonical_undirected(data, num_nodes)
        m = int(src.size)
        weight = self._edge_weights(data, src, dst)

        delta = self.delta if self.target_ratio is None else float(self.target_ratio)
        delta = max(0.0, min(1.0, delta))
        target_edges = int(min(m, math.ceil(delta * m - 1e-12)))

        artifact = self._get_artifact(num_nodes, src, dst, weight)
        # A fixed backbone has no rotation, so the plan -- and in particular the
        # cap_and_renormalize pass over the whole pool -- is built once and
        # reused for every epoch. Only rotate-randst rebuilds per rotation.
        rotation = (
            0
            if self.sample_backbone == "fixed-maxst"
            else self._draw_index % max(1, int(artifact["tree_count"]))
        )

        plan_key = (target_edges, self.sample_backbone, rotation, self.sample_weight_mode, self.sample_mix_alpha)
        if self._plan is None or self._plan_key != plan_key:
            self._plan = self._build_plan(artifact, target_edges, rotation)
            self._plan_key = plan_key
        plan = self._plan

        rng = np.random.default_rng(
            np.random.SeedSequence([int(self._active_seed() or 0), int(self._draw_index)])
        )
        chosen = np.concatenate((plan["forced"], self._draw_pool(plan, rng)))
        chosen = np.unique(chosen)
        if chosen.size != target_edges and m:
            raise RuntimeError(
                "scaffold-sample budget invariant failed: "
                f"selected {chosen.size} edges, expected {target_edges}"
            )

        sel_src = src[chosen]
        sel_dst = dst[chosen]

        components = None
        if self._should_assert_connectivity(m):
            components = ts.component_count(num_nodes, sel_src, sel_dst)
            expected = int(artifact["base_components"])
            if not plan["trimmed"] and components != expected:
                raise RuntimeError(
                    "scaffold-sample connectivity invariant failed: "
                    f"{components} components, expected {expected}"
                )

        out = self._build_output(data, num_nodes, sel_src, sel_dst)
        elapsed = time.perf_counter() - start
        self.last_cluster_stats = {
            "algorithm": "scaffold_sample",
            "backbone": self.sample_backbone,
            "backbone_tag": plan["backbone_tag"],
            "scheme": self.sample_scheme,
            "weight_mode": self.sample_weight_mode,
            "mix_alpha": self.sample_mix_alpha if self.sample_weight_mode == 'normalized-mixture' else None,
            "rotation": int(rotation),
            "draw_index": int(self._draw_index),
            "tree_count": int(artifact["tree_count"]),
            "delta_min": float(artifact["delta_min"]),
            "budget_trimmed": bool(plan["trimmed"]),
            "forced_edges": int(plan["forced"].size),
            "sampled_edges": int(plan["remaining"]),
            "target_edges": int(target_edges),
            "final_edges": int(chosen.size),
            "components": components,
            "base_components": int(artifact["base_components"]),
            "artifact_build_seconds": float(artifact["build_seconds"]),
            "parallel_workers": self.parallel_workers,
            "workers_used": self._draw_workers_used,
            "parallelism": ("systematic_tick_blocks"
                            if self.sample_scheme == "systematic" else "serial_alias"),
            "sparsification_time_sec": elapsed,
        }
        warn = ""
        if plan["trimmed"]:
            warn = (
                f" WARNING budget {target_edges} < connectivity floor "
                f"(delta_min={float(artifact['delta_min']):.4f}); graph is fragmented"
            )
        print(
            f"[SCAFFOLD-Sample] backbone={self.sample_backbone}/{plan['backbone_tag']} "
            f"scheme={self.sample_scheme} draw={self._draw_index} "
            f"weight_mode={self.sample_weight_mode} "
            f"mix_alpha={self.sample_mix_alpha if self.sample_weight_mode == 'normalized-mixture' else 'NA'} "
            f"forced={plan['forced'].size} sampled={plan['remaining']} "
            f"edges={chosen.size}/{target_edges} "
            f"draw_workers={self._draw_workers_used} "
            f"components={components if components is not None else 'skipped'} "
            f"time={elapsed:.3f}s{warn}",
            flush=True,
        )
        return out

    def _should_assert_connectivity(self, num_edges):
        mode = self.sample_assert_connectivity
        if mode in ("true", "1", "yes", "on"):
            return True
        if mode in ("false", "0", "no", "off"):
            return False
        return num_edges <= 5_000_000

    @staticmethod
    def _build_output(data, num_nodes, sel_src, sel_dst):
        out_src = np.concatenate((sel_src, sel_dst))
        out_dst = np.concatenate((sel_dst, sel_src))
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
        out.num_undirected_edges = int(sel_src.shape[0])
        return out

    def resparsify(self, seed=None):
        """Advance the rotation and redraw. No rebuild, no rescoring."""
        self._draw_index += 1
        return super().resparsify(seed=seed)
