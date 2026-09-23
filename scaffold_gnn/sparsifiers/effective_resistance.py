"""One-shot spectral topology sampling from precomputed ER artifacts."""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from .base import BaseSparsifier

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


DATASET_DIRECTORIES = {
    "karate": "Karate",
    "cora": "Planetoid/Cora",
    "citeseer": "Planetoid/CiteSeer",
    "pubmed": "Planetoid/PubMed",
    "amazon-computer": "Amazon/Computers",
    "amazon-photo": "Amazon/Photo",
    "coauthor-cs": "Coauthor/CS",
    "coauthor-physics": "Coauthor/Physics",
    "wikics": "wikics",
    "squirrel": "geom-gcn/squirrel",
    "chameleon": "geom-gcn/chameleon",
    "roman-empire": "Heterophilous/roman_empire",
    "amazon-ratings": "Heterophilous/amazon_ratings",
    "minesweeper": "Heterophilous/minesweeper",
    "questions": "Heterophilous/questions",
    "reddit": "Reddit",
    "ogbn-products": "ogb/ogbn_products",
    "ogbn-arxiv": "ogb/ogbn_arxiv",
    "ogbn-proteins": "ogb/ogbn_proteins",
    "pokec": "pokec",
}


def effective_resistance_artifact(data_root, dataset):
    key = str(dataset).strip().lower().replace("_", "-")
    if key not in DATASET_DIRECTORIES:
        raise ValueError(f"no effective-resistance artifact mapping for {dataset!r}")
    return Path(os.environ.get("SCAFFOLD_SPECTRAL_CACHE", "./results/cache/spectral")) / key / "effective_resistance.npz"


def effective_resistance_artifact_signature(path):
    """Return a cheap cache identity that changes when an artifact is replaced."""

    artifact = Path(path)
    stat = artifact.stat()
    graph_fingerprint = ""
    with np.load(artifact, allow_pickle=False) as values:
        if "graph_fingerprint" in values.files:
            graph_fingerprint = bytes(
                np.asarray(values["graph_fingerprint"], dtype=np.uint8).tolist()
            ).decode("ascii")
    return f"{stat.st_size}:{stat.st_mtime_ns}:{graph_fingerprint}"


def _alias_table_python(probabilities):
    """Vose alias table equivalent to the Julia spectral sampler."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    count = int(probabilities.size)
    scaled = probabilities * count
    threshold = np.ones(count, dtype=np.float64)
    alias = np.arange(count, dtype=np.int64)
    small = list(np.flatnonzero(scaled < 1.0))
    large = list(np.flatnonzero(scaled >= 1.0))
    while small and large:
        low = small.pop()
        high = large.pop()
        threshold[low] = scaled[low]
        alias[low] = high
        scaled[high] -= 1.0 - scaled[low]
        (small if scaled[high] < 1.0 else large).append(high)
    return threshold, alias


if njit is not None:
    @njit(cache=True, nogil=True)
    def _alias_table_numba(probabilities):
        count = int(probabilities.size)
        scaled = probabilities * count
        threshold = np.ones(count, dtype=np.float64)
        alias = np.arange(count, dtype=np.int64)
        small = np.empty(count, dtype=np.int64)
        large = np.empty(count, dtype=np.int64)
        small_count = 0
        large_count = 0
        for index in range(count):
            if scaled[index] < 1.0:
                small[small_count] = index
                small_count += 1
            else:
                large[large_count] = index
                large_count += 1
        while small_count > 0 and large_count > 0:
            small_count -= 1
            low = small[small_count]
            large_count -= 1
            high = large[large_count]
            threshold[low] = scaled[low]
            alias[low] = high
            scaled[high] -= 1.0 - scaled[low]
            if scaled[high] < 1.0:
                small[small_count] = high
                small_count += 1
            else:
                large[large_count] = high
                large_count += 1
        return threshold, alias


def _alias_table(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if njit is not None:
        return _alias_table_numba(probabilities)
    return _alias_table_python(probabilities)


def _sample_alias_counts(threshold, alias, draws, seed):
    rng = np.random.default_rng(seed)
    count = int(threshold.size)
    candidates = rng.integers(0, count, size=int(draws), dtype=np.int64)
    uniforms = rng.random(int(draws))
    selected = np.where(
        uniforms <= threshold[candidates], candidates, alias[candidates]
    )
    return np.unique(selected, return_counts=True)


def _sample_alias_indices(threshold, alias, draws, seed):
    """Return ordered alias draws for exact-unique ER sampling."""

    rng = np.random.default_rng(seed)
    count = int(threshold.size)
    candidates = rng.integers(0, count, size=int(draws), dtype=np.int64)
    uniforms = rng.random(int(draws))
    return np.where(
        uniforms <= threshold[candidates], candidates, alias[candidates]
    )


def _merge_sample_counts(pieces):
    indices = np.concatenate([piece[0] for piece in pieces])
    counts = np.concatenate([piece[1] for piece in pieces]).astype(
        np.int64, copy=False
    )
    order = np.argsort(indices, kind="stable")
    ordered_indices = indices[order]
    starts = np.r_[0, np.flatnonzero(
        ordered_indices[1:] != ordered_indices[:-1]
    ) + 1]
    return ordered_indices[starts], np.add.reduceat(counts[order], starts)


def _sample_alias_until_unique(threshold, alias, target_unique, seed, workers):
    """Draw with replacement until exactly ``target_unique`` edges appear.

    Draw generation is split across CPU workers. Results are consumed in a
    deterministic worker order, and the final chunk is truncated at the draw
    that discovers the requested final edge. This preserves repeated-draw
    counts for weighted GNN training while guaranteeing the topology budget.
    """

    edge_count = int(threshold.size)
    target_unique = int(target_unique)
    if target_unique < 0 or target_unique > edge_count:
        raise ValueError(
            f"target unique-edge count {target_unique} is outside [0, {edge_count}]"
        )
    if target_unique == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            0,
        )

    sample_counts = np.zeros(edge_count, dtype=np.uint32)
    unique_count = 0
    draw_count = 0
    workers = max(1, min(int(workers or 1), target_unique))
    seed_sequence = np.random.SeedSequence(int(seed))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while unique_count < target_unique:
            remaining = target_unique - unique_count
            # The observed novelty rate keeps batches large enough to utilize
            # all workers without allocating an unbounded draw array.
            novelty = (
                unique_count / draw_count if draw_count > 0 else 1.0
            )
            expected_new_rate = max(0.02, min(1.0, 1.0 - novelty))
            round_draws = int(math.ceil(1.10 * remaining / expected_new_rate))
            round_draws = max(262_144, min(round_draws, 4_194_304))
            base, remainder = divmod(round_draws, workers)
            worker_draws = [
                base + (worker < remainder) for worker in range(workers)
            ]
            worker_seeds = seed_sequence.spawn(workers)
            pieces = list(executor.map(
                _sample_alias_indices,
                [threshold] * workers,
                [alias] * workers,
                worker_draws,
                worker_seeds,
            ))
            candidates = np.concatenate(pieces)

            unique_values, first_positions, counts = np.unique(
                candidates, return_index=True, return_counts=True
            )
            new_mask = sample_counts[unique_values] == 0
            new_positions = np.sort(first_positions[new_mask])
            if new_positions.size >= remaining:
                stop = int(new_positions[remaining - 1]) + 1
                candidates = candidates[:stop]
                unique_values, counts = np.unique(
                    candidates, return_counts=True
                )

            newly_seen = int(np.count_nonzero(sample_counts[unique_values] == 0))
            sample_counts[unique_values] += counts.astype(np.uint32, copy=False)
            unique_count += newly_seen
            draw_count += int(candidates.size)

    selected = np.flatnonzero(sample_counts).astype(np.int64, copy=False)
    selected_counts = sample_counts[selected].astype(np.int64, copy=False)
    if selected.size != target_unique:
        raise RuntimeError(
            "exact effective-resistance sampler failed its unique-edge invariant: "
            f"expected {target_unique}, observed {selected.size}"
        )
    if int(selected_counts.sum()) != draw_count:
        raise RuntimeError(
            "exact effective-resistance sampler failed its draw-count invariant: "
            f"expected {draw_count}, observed {int(selected_counts.sum())}"
        )
    return selected, selected_counts, draw_count


def _validate_artifact_topology(data, src, dst, artifact):
    """Require the artifact and input to contain exactly the same node pairs."""

    edge_index = data.edge_index.detach().cpu().long()
    edge_src, edge_dst = edge_index[0], edge_index[1]
    non_loop = edge_src != edge_dst
    if bool(getattr(data, "edge_index_is_symmetric_unique", False)):
        keep = non_loop & (edge_src < edge_dst)
        low, high = edge_src[keep], edge_dst[keep]
        already_unique = True
    else:
        low = torch.minimum(edge_src[non_loop], edge_dst[non_loop])
        high = torch.maximum(edge_src[non_loop], edge_dst[non_loop])
        already_unique = bool(
            getattr(data, "edge_index_is_undirected_unique", False)
        )
    stride = int(data.num_nodes)
    source_keys = low * stride + high
    if not already_unique:
        source_keys = torch.unique(source_keys, sorted=True)
    source_keys = source_keys.numpy()
    source_keys.sort()

    artifact_low = np.minimum(src, dst)
    artifact_high = np.maximum(src, dst)
    if (
        (artifact_low < 0).any()
        or (artifact_high >= stride).any()
    ):
        raise ValueError(
            f"effective-resistance artifact has a node outside [0, {stride}): {artifact}"
        )
    if (artifact_low == artifact_high).any():
        raise ValueError(f"effective-resistance artifact contains a self-loop: {artifact}")
    artifact_keys = artifact_low * np.int64(stride) + artifact_high
    artifact_keys.sort()
    if source_keys.size != artifact_keys.size:
        raise ValueError(
            f"effective-resistance artifact edge count {artifact_keys.size} does not "
            f"match source graph edge count {source_keys.size}: {artifact}"
        )
    if not np.array_equal(source_keys, artifact_keys):
        mismatch = int(np.flatnonzero(source_keys != artifact_keys)[0])
        source_u, source_v = divmod(int(source_keys[mismatch]), stride)
        artifact_u, artifact_v = divmod(int(artifact_keys[mismatch]), stride)
        raise ValueError(
            "effective-resistance artifact topology does not match source graph; "
            f"first sorted mismatch source=({source_u}, {source_v}) "
            f"artifact=({artifact_u}, {artifact_v}): {artifact}"
        )


class EffectiveResistanceSparsifier(BaseSparsifier):
    """Sample edges by leverage score ``conductance * resistance``.

    The draw budget is ``ceil(target_ratio * m)`` with replacement, matching
    ``SpectralSparsification.jl::sample_sparsifier``. Duplicate draws collapse
    to one topology edge, so the achieved retained ratio is measured rather
    than assumed to equal the requested ratio.
    """

    def __init__(
        self,
        target_ratio,
        seed=42,
        artifact_path=None,
        artifact_signature=None,
        parallel_workers=1,
        exact_unique=False,
    ):
        self.target_ratio = float(target_ratio)
        self.seed = int(seed)
        self.artifact_path = str(artifact_path) if artifact_path else None
        self.artifact_signature = str(artifact_signature or "")
        self.parallel_workers = max(1, int(parallel_workers or 1))
        self.exact_unique = bool(exact_unique)
        if not 0.0 < self.target_ratio <= 1.0:
            raise ValueError("target_ratio must be in (0, 1]")

    def sparsify(self, data_or_graph):
        if not isinstance(data_or_graph, Data):
            raise TypeError("EffectiveResistanceSparsifier expects PyG Data")
        if not self.artifact_path:
            raise ValueError("effective-resistance artifact path is required")
        artifact = Path(self.artifact_path)
        if not artifact.is_file():
            raise FileNotFoundError(
                f"effective-resistance artifact is not ready: {artifact}"
            )

        with np.load(artifact, allow_pickle=False) as values:
            src = np.asarray(values["src"], dtype=np.int64).copy()
            dst = np.asarray(values["dst"], dtype=np.int64).copy()
            conductance = np.asarray(
                values["conductance"], dtype=np.float64
            ).copy()
            if "leverage_score" in values.files:
                leverage = np.asarray(
                    values["leverage_score"], dtype=np.float64
                ).copy()
            else:
                leverage = (
                    np.asarray(values["conductance"], dtype=np.float64)
                    * np.asarray(values["resistance"], dtype=np.float64)
                )

        edge_count = int(src.size)
        if (
            dst.size != edge_count
            or conductance.size != edge_count
            or leverage.size != edge_count
        ):
            raise ValueError(f"misaligned effective-resistance artifact: {artifact}")
        _validate_artifact_topology(data_or_graph, src, dst, artifact)
        if not np.isfinite(conductance).all() or (conductance <= 0.0).any():
            raise ValueError(
                f"effective-resistance conductance must be finite and positive: {artifact}"
            )
        if edge_count == 0:
            selected = np.empty(0, dtype=np.int64)
            sample_counts = np.empty(0, dtype=np.int64)
            sampled_weight = np.empty(0, dtype=np.float32)
            budget = 0
        else:
            leverage = np.nan_to_num(
                leverage, nan=0.0, posinf=0.0, neginf=0.0
            ).clip(min=0.0)
            total = float(leverage.sum())
            probabilities = (
                leverage / total
                if math.isfinite(total) and total > 0.0
                else np.full(edge_count, 1.0 / edge_count, dtype=np.float64)
            )
            threshold, alias = _alias_table(probabilities)
            target_edges = max(
                1, min(edge_count, int(math.ceil(self.target_ratio * edge_count)))
            )
            if self.exact_unique:
                selected, sample_counts, budget = _sample_alias_until_unique(
                    threshold,
                    alias,
                    target_edges,
                    self.seed,
                    self.parallel_workers,
                )
            else:
                budget = target_edges
                workers = min(self.parallel_workers, budget)
                base, remainder = divmod(budget, workers)
                draw_counts = [base + (worker < remainder) for worker in range(workers)]
                seeds = np.random.SeedSequence(self.seed).spawn(workers)
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    pieces = list(executor.map(
                        _sample_alias_counts,
                        [threshold] * workers,
                        [alias] * workers,
                        draw_counts,
                        seeds,
                    ))
                selected, sample_counts = _merge_sample_counts(pieces)
            sampled_weight = (
                sample_counts.astype(np.float64)
                * conductance[selected]
                / (budget * probabilities[selected])
            ).astype(np.float32)

        pairs = torch.from_numpy(np.stack((src[selected], dst[selected]), axis=0))
        edge_index = torch.cat((pairs, pairs.flip(0)), dim=1).long().contiguous()
        one_direction_weight = torch.from_numpy(sampled_weight)
        edge_weight = torch.cat(
            (one_direction_weight, one_direction_weight), dim=0
        ).float().contiguous()
        output = Data(
            x=data_or_graph.x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=data_or_graph.y,
            num_nodes=data_or_graph.num_nodes,
        )
        output.edge_index_is_symmetric_unique = True
        output.num_undirected_edges = int(selected.size)
        output.sample_draw_count = int(budget)
        output.sample_unique_edge_count = int(selected.size)
        if int(sample_counts.sum()) != int(budget):
            raise RuntimeError(
                'effective-resistance sample-count invariant failed: '
                f'expected {budget}, observed {int(sample_counts.sum())}'
            )
        weight_min = float(sampled_weight.min()) if sampled_weight.size else 0.0
        weight_max = float(sampled_weight.max()) if sampled_weight.size else 0.0
        print(
            f"[EffectiveResistance] artifact={artifact} draws="
            f"{budget} unique_edges={selected.size} weighted=true "
            f"draw_ratio={(budget / edge_count if edge_count else 0.0):.12g} "
            f"unique_ratio={(selected.size / edge_count if edge_count else 0.0):.12g} "
            f"weight_min={weight_min:.6g} weight_max={weight_max:.6g} "
            f"workers={self.parallel_workers} exact_unique={str(self.exact_unique).lower()}",
            flush=True,
        )
        return output


class EffectiveResistanceExactSparsifier(EffectiveResistanceSparsifier):
    """ER-Fix: keep drawing until the requested unique-edge ratio is exact."""

    def __init__(self, *args, **kwargs):
        kwargs["exact_unique"] = True
        super().__init__(*args, **kwargs)
