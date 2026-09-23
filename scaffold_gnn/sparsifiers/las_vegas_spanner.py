"""Adaptive Baswana--Sen and scalable CUDA Las Vegas graph spanners."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path

import networkx as nx
import torch
from torch_geometric.data import Data

from .base import BaseSparsifier


class LasVegasSpannerSparsifier(BaseSparsifier):
    """Adaptive Baswana--Sen spanner with an exact postprocessed edge budget.

    Small/medium graphs tune several odd stretch values using NetworkX's Las
    Vegas Baswana--Sen implementation and persist the closest choice. Large
    graphs use the specialized CUDA t=3 construction. The result is then
    forced to the requested edge budget: source edges are added when it is too
    small, or randomly removed when it is too large. Adding edges preserves
    the stretch certificate; user-requested trimming may intentionally
    forfeit it.
    """

    def __init__(
        self,
        stretch=3,
        target_ratio=1.0,
        seed=42,
        parallel_workers=1,
        device="auto",
        dataset="unknown",
        tuning_cache_path=None,
        networkx_max_edges=1_000_000,
    ):
        self.stretch = int(stretch)
        self.target_ratio = float(target_ratio)
        self.seed = int(seed)
        self.parallel_workers = max(1, int(parallel_workers or 1))
        self.device = str(device)
        self.dataset = str(dataset).strip().lower().replace("_", "-")
        self.tuning_cache_path = (
            str(tuning_cache_path) if tuning_cache_path else None
        )
        self.networkx_max_edges = max(0, int(networkx_max_edges))
        if self.stretch < 3 or self.stretch % 2 == 0:
            raise ValueError("Las Vegas spanner stretch must be an odd integer >= 3")
        if not 0.0 < self.target_ratio <= 1.0:
            raise ValueError("target_ratio must be in (0, 1]")

    @staticmethod
    def _topology_fingerprint(pairs, num_nodes):
        digest = hashlib.sha256()
        digest.update(f"{int(num_nodes)}:{int(pairs.size(1))}:".encode("ascii"))
        values = pairs.detach().cpu().long().contiguous()
        if values.size(1) > 1_000_000:
            positions = torch.linspace(
                0, values.size(1) - 1, steps=1024, dtype=torch.float64
            ).long()
            values = values[:, positions].contiguous()
        digest.update(memoryview(values.numpy()).cast("B"))
        return digest.hexdigest()

    def _tuning_file(self):
        if not self.tuning_cache_path:
            return None
        ratio = f"{self.target_ratio:.10f}".rstrip("0").rstrip(".").replace(".", "p")
        return Path(self.tuning_cache_path) / self.dataset / f"ratio_{ratio}.json"

    def _load_tuning(self, *, num_nodes, edge_count, target_count, fingerprint):
        path = self._tuning_file()
        if path is None or not path.is_file():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        expected = {
            "schema_version": 1,
            "dataset": self.dataset,
            "num_nodes": int(num_nodes),
            "input_edges": int(edge_count),
            "target_edges": int(target_count),
            "target_ratio": self.target_ratio,
            "seed": self.seed,
            "topology_fingerprint": fingerprint,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            return None
        stretch = record.get("chosen_stretch")
        if not isinstance(stretch, int) or stretch < 3 or stretch % 2 == 0:
            return None
        return record

    def _save_tuning(self, record):
        path = self._tuning_file()
        if path is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
        temporary.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return path

    @staticmethod
    def _candidate_stretches(num_nodes, edge_count, target_count, configured):
        candidates = sorted({3, 5, 7, 9, 11, int(configured)})

        def estimated_edges(stretch):
            k = (stretch + 1) // 2
            estimate = num_nodes ** (1.0 + 1.0 / k) + k * num_nodes
            return min(float(edge_count), estimate)

        start = min(
            candidates,
            key=lambda value: abs(estimated_edges(value) - target_count),
        )
        return candidates, start

    def _networkx_spanner_ids(self, graph, key_to_id, num_nodes, stretch):
        candidate = nx.spanner(
            graph,
            stretch=stretch,
            weight=None,
            seed=self.seed + stretch * 1_000_003,
        )
        selected = []
        for u, v in candidate.edges():
            key = min(int(u), int(v)) * num_nodes + max(int(u), int(v))
            selected.append(key_to_id[key])
        return torch.tensor(selected, dtype=torch.long)

    def _force_target(self, selected_ids, edge_count, target_count, generator=None):
        selected_ids = selected_ids.long()
        selected_mask = torch.zeros(
            edge_count, dtype=torch.bool, device=selected_ids.device
        )
        selected_mask[selected_ids] = True
        spanner_edge_count = int(selected_ids.numel())
        if generator is None:
            generator = torch.Generator(device=selected_ids.device)
            generator.manual_seed(self.seed)
        if spanner_edge_count > target_count:
            order = torch.randperm(
                spanner_edge_count,
                device=selected_ids.device,
                generator=generator,
            )
            selected_ids = selected_ids[order[:target_count]]
            action = "random_trim"
        elif spanner_edge_count < target_count:
            remaining_ids = torch.nonzero(
                ~selected_mask, as_tuple=False
            ).view(-1)
            order = torch.randperm(
                int(remaining_ids.numel()),
                device=selected_ids.device,
                generator=generator,
            )
            selected_ids = torch.cat((
                selected_ids,
                remaining_ids[order[:target_count - spanner_edge_count]],
            ))
            action = "random_fill"
        else:
            action = "exact"
        if int(selected_ids.numel()) != int(target_count):
            raise RuntimeError(
                "Las Vegas spanner failed exact target control: "
                f"target={target_count} output={selected_ids.numel()}"
            )
        return selected_ids, action, spanner_edge_count

    def _build_pyg_output(self, data, pairs, selected_ids):
        selected_ids = selected_ids.detach().cpu()
        kept = pairs[:, selected_ids]
        edge_index = torch.cat((kept, kept.flip(0)), dim=1).contiguous()
        output = Data(
            x=data.x,
            edge_index=edge_index,
            y=data.y,
            num_nodes=data.num_nodes,
        )
        output.edge_index_is_symmetric_unique = True
        output.num_undirected_edges = int(selected_ids.numel())
        return output

    def _sparsify_adaptive_networkx(
        self, data, pairs, num_nodes, edge_count, target_count
    ):
        fingerprint = self._topology_fingerprint(pairs, num_nodes)
        record = self._load_tuning(
            num_nodes=num_nodes,
            edge_count=edge_count,
            target_count=target_count,
            fingerprint=fingerprint,
        )
        source_edges = pairs.t().tolist()
        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from(source_edges)
        key_to_id = {
            min(int(u), int(v)) * num_nodes + max(int(u), int(v)): index
            for index, (u, v) in enumerate(source_edges)
        }
        candidate_counts = {}
        if record is not None:
            chosen_stretch = int(record["chosen_stretch"])
            selected_ids = self._networkx_spanner_ids(
                graph, key_to_id, num_nodes, chosen_stretch
            )
            candidate_counts[str(chosen_stretch)] = int(selected_ids.numel())
            tuning_status = "cache_hit"
        else:
            candidates, start = self._candidate_stretches(
                num_nodes, edge_count, target_count, self.stretch
            )
            evaluated = {}

            def evaluate(stretch):
                if stretch not in evaluated:
                    evaluated[stretch] = self._networkx_spanner_ids(
                        graph, key_to_id, num_nodes, stretch
                    )
                return int(evaluated[stretch].numel())

            count = evaluate(start)
            position = candidates.index(start)
            if count > target_count:
                while position + 1 < len(candidates):
                    position += 1
                    if evaluate(candidates[position]) <= target_count:
                        break
            elif count < target_count:
                while position > 0:
                    position -= 1
                    if evaluate(candidates[position]) >= target_count:
                        break
            chosen_stretch = min(
                evaluated,
                key=lambda value: (
                    abs(int(evaluated[value].numel()) - target_count),
                    0 if int(evaluated[value].numel()) >= target_count else 1,
                    value,
                ),
            )
            selected_ids = evaluated[chosen_stretch]
            candidate_counts = {
                str(value): int(ids.numel())
                for value, ids in sorted(evaluated.items())
            }
            record = {
                "schema_version": 1,
                "dataset": self.dataset,
                "num_nodes": int(num_nodes),
                "input_edges": int(edge_count),
                "target_edges": int(target_count),
                "target_ratio": self.target_ratio,
                "seed": self.seed,
                "topology_fingerprint": fingerprint,
                "strategy": "adaptive_networkx_baswana_sen",
                "chosen_stretch": int(chosen_stretch),
                "candidate_edge_counts": candidate_counts,
            }
            self._save_tuning(record)
            tuning_status = "cache_saved"

        selected_ids, action, base_count = self._force_target(
            selected_ids, edge_count, target_count
        )
        output = self._build_pyg_output(data, pairs, selected_ids)
        print(
            "[LasVegasSpanner] strategy=adaptive_networkx_baswana_sen "
            f"chosen_t={chosen_stretch} candidates={candidate_counts} "
            f"input_edges={edge_count} spanner_edges={base_count} "
            f"target_edges={target_count} output_edges={selected_ids.numel()} "
            f"budget_action={action} tuning={tuning_status} "
            f"tuning_cache={self._tuning_file()} cpu_workers={self.parallel_workers}",
            flush=True,
        )
        return output

    def _compute_device(self):
        if self.device == "cpu":
            return torch.device("cpu")
        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested for Las Vegas spanner but unavailable")
            return torch.device(self.device)
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def sparsify(self, data_or_graph):
        if not isinstance(data_or_graph, Data):
            graph = self._ensure_nx(data_or_graph)
            if graph.is_directed():
                raise ValueError("Las Vegas spanner requires an undirected graph")
            spanner = nx.spanner(graph, stretch=3, weight=None, seed=self.seed)
            source_edges = list(graph.edges())
            if not source_edges:
                return spanner
            target_count = max(
                1, min(len(source_edges), round(len(source_edges) * self.target_ratio))
            )
            selected = list(spanner.edges())
            rng = random.Random(self.seed)
            if len(selected) > target_count:
                selected = rng.sample(selected, target_count)
            elif len(selected) < target_count:
                selected_keys = {tuple(sorted(edge)) for edge in selected}
                remaining = [
                    edge for edge in source_edges
                    if tuple(sorted(edge)) not in selected_keys
                ]
                selected.extend(rng.sample(remaining, target_count - len(selected)))
            output = nx.Graph()
            output.add_nodes_from(graph.nodes(data=True))
            output.add_edges_from(selected)
            return output

        pairs = self._pyg_undirected_pairs(data_or_graph)
        edge_count = int(pairs.size(1))
        num_nodes = int(data_or_graph.num_nodes)
        if edge_count == 0:
            output = data_or_graph.clone()
            output.edge_index = torch.empty((2, 0), dtype=torch.long)
            output.edge_index_is_symmetric_unique = True
            output.num_undirected_edges = 0
            return output

        target_count = max(
            1, min(edge_count, round(edge_count * self.target_ratio))
        )
        if edge_count <= self.networkx_max_edges:
            return self._sparsify_adaptive_networkx(
                data_or_graph,
                pairs,
                num_nodes,
                edge_count,
                target_count,
            )

        device = self._compute_device()
        src = pairs[0].to(device)
        dst = pairs[1].to(device)
        edge_ids = torch.arange(edge_count, device=device, dtype=torch.long)
        sample_probability = num_nodes ** -0.5
        size_limit = int(math.ceil(2.0 * num_nodes ** 1.5))
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed)

        accepted = None
        for attempt in range(1, 129):
            sampled = torch.rand(num_nodes, device=device, generator=generator) < sample_probability
            if not bool(sampled.any()):
                sampled[torch.randint(num_nodes, (1,), device=device, generator=generator)] = True

            candidate_forward = (~sampled[src]) & sampled[dst]
            candidate_reverse = (~sampled[dst]) & sampled[src]
            nodes = torch.cat((src[candidate_forward], dst[candidate_reverse]))
            candidates = torch.cat((edge_ids[candidate_forward], edge_ids[candidate_reverse]))
            sentinel = edge_count
            chosen = torch.full((num_nodes,), sentinel, dtype=torch.long, device=device)
            if candidates.numel():
                chosen.scatter_reduce_(0, nodes, candidates, reduce="amin", include_self=True)

            joined = (~sampled) & (chosen < sentinel)
            inactive = (~sampled) & (~joined)
            phase_mask = inactive[src] | inactive[dst]
            phase_mask[chosen[joined]] = True
            phase_edges = int(phase_mask.sum().item())
            if phase_edges <= size_limit:
                accepted = (sampled, joined, inactive, chosen, phase_mask, attempt)
                break
        if accepted is None:
            raise RuntimeError("Las Vegas spanner did not satisfy its phase size certificate")

        sampled, joined, inactive, chosen, selected_mask, attempts = accepted
        cluster = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        node_ids = torch.arange(num_nodes, device=device, dtype=torch.long)
        cluster[sampled] = node_ids[sampled]
        joined_edges = chosen[joined]
        joined_nodes = node_ids[joined]
        joined_src = src[joined_edges]
        joined_dst = dst[joined_edges]
        joined_centers = torch.where(joined_src == joined_nodes, joined_dst, joined_src)
        cluster[joined] = joined_centers

        residual = (
            (~inactive[src])
            & (~inactive[dst])
            & (cluster[src] != cluster[dst])
        )
        residual_ids = edge_ids[residual]
        residual_src = src[residual]
        residual_dst = dst[residual]
        keys = torch.cat((
            residual_src * num_nodes + cluster[residual_dst],
            residual_dst * num_nodes + cluster[residual_src],
        ))
        directed_edge_ids = torch.cat((residual_ids, residual_ids))
        if keys.numel():
            ordered_keys, order = torch.sort(keys)
            first = torch.ones(ordered_keys.numel(), dtype=torch.bool, device=device)
            first[1:] = ordered_keys[1:] != ordered_keys[:-1]
            selected_mask[directed_edge_ids[order[first]]] = True

        selected_ids = torch.nonzero(selected_mask, as_tuple=False).view(-1)
        spanner_edge_count = int(selected_ids.numel())
        selected_ids, budget_action, spanner_edge_count = self._force_target(
            selected_ids, edge_count, target_count, generator=generator
        )
        output = self._build_pyg_output(data_or_graph, pairs, selected_ids)
        fingerprint = self._topology_fingerprint(pairs, num_nodes)
        record = {
            "schema_version": 1,
            "dataset": self.dataset,
            "num_nodes": num_nodes,
            "input_edges": edge_count,
            "target_edges": target_count,
            "target_ratio": self.target_ratio,
            "seed": self.seed,
            "topology_fingerprint": fingerprint,
            "strategy": "scalable_cuda_t3",
            "chosen_stretch": 3,
            "candidate_edge_counts": {"3": spanner_edge_count},
        }
        tuning_file = self._save_tuning(record)
        print(
            f"[LasVegasSpanner] strategy=scalable_cuda_t3 chosen_t=3 "
            f"attempts={attempts} "
            f"input_edges={edge_count} spanner_edges={spanner_edge_count} "
            f"target_edges={target_count} output_edges={selected_ids.numel()} "
            f"budget_action={budget_action} "
            f"tuning_cache={tuning_file} "
            f"device={device} cpu_workers={self.parallel_workers}",
            flush=True,
        )
        return output
