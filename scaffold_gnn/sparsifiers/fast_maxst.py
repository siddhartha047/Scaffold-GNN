import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .base import BaseSparsifier
from .scaffold.spanning_tree import (
    FAST_TREE_DEFAULT_BUCKETS,
    build_fast_weighted_spanning_forest_mask,
    build_fast_weighted_spanning_forest_nx,
)


class FastMaxSTSparsifier(BaseSparsifier):
    """Benchmark-style parallel bucketed maximum spanning forest.

    PyG graphs use cosine feature similarity as edge weight. Feature scoring
    and priority bucketing use the configured CPU thread pool; deterministic
    Kruskal union-find then constructs one complete spanning forest.
    """

    def __init__(
        self,
        weight_method="cosine",
        bucket_count=FAST_TREE_DEFAULT_BUCKETS,
        parallel_workers=None,
        target_ratio=None,
    ):
        self.weight_method = str(weight_method).lower()
        if self.weight_method not in {"uniform", "cosine", "euclidean", "dot"}:
            raise ValueError(f"unsupported fast-maxst weight method: {weight_method}")
        self.bucket_count = int(bucket_count)
        self.parallel_workers = parallel_workers
        # Accepted for API compatibility only. A spanning forest is inherently
        # ratio-independent and is never truncated to this requested budget.
        self.target_ratio = target_ratio

    @torch.no_grad()
    def _feature_scores(self, data, pairs, batch_size=250_000):
        if self.weight_method == "uniform":
            return None
        features = getattr(data, "x", None)
        if features is None:
            print(
                "[Fast-MaxST] node features unavailable; using uniform priorities",
                flush=True,
            )
            return None
        x = torch.as_tensor(features).detach().cpu().float()
        if x.ndim == 1:
            x = x.unsqueeze(1)
        src, dst = pairs[0], pairs[1]
        parts = []
        for start in range(0, int(src.numel()), batch_size):
            stop = min(int(src.numel()), start + batch_size)
            left = x[src[start:stop]]
            right = x[dst[start:stop]]
            if self.weight_method == "cosine":
                score = (F.cosine_similarity(left, right, dim=-1, eps=1e-12) + 1.0) * 0.5
            elif self.weight_method == "euclidean":
                score = 1.0 / (1.0 + torch.linalg.vector_norm(left - right, ord=2, dim=-1))
            else:
                score = (left * right).sum(dim=-1)
            parts.append(score.float())
        scores = torch.cat(parts) if parts else torch.empty(0, dtype=torch.float)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        if self.weight_method == "dot" and scores.numel():
            low, high = scores.min(), scores.max()
            scores = (
                torch.ones_like(scores)
                if float(high - low) <= 1e-12
                else (scores - low) / (high - low)
            )
        return scores.clamp(0.0, 1.0).numpy().astype(np.float32, copy=False)

    def sparsify(self, data_or_graph):
        if isinstance(data_or_graph, Data):
            pairs = self._pyg_undirected_pairs(data_or_graph)
            src = pairs[0].numpy().astype(np.int64, copy=False)
            dst = pairs[1].numpy().astype(np.int64, copy=False)
            scores = self._feature_scores(data_or_graph, pairs)
            mask = build_fast_weighted_spanning_forest_mask(
                int(data_or_graph.num_nodes),
                src,
                dst,
                scores=scores,
                maximum=True,
                bucket_count=self.bucket_count,
                parallel_workers=self.parallel_workers,
            )
            kept = pairs[:, torch.from_numpy(mask)]
            edge_index = torch.cat((kept, kept.flip(0)), dim=1).contiguous()
            output = Data(
                x=data_or_graph.x,
                edge_index=edge_index,
                y=data_or_graph.y,
                num_nodes=data_or_graph.num_nodes,
            )
            output.edge_index_is_symmetric_unique = True
            output.num_undirected_edges = int(kept.size(1))
            return output

        graph = self._ensure_nx(data_or_graph)
        if graph.is_directed():
            raise ValueError("FastMaxSTSparsifier requires an undirected graph")
        weight_key = "weight" if nx.is_weighted(graph) else None
        return build_fast_weighted_spanning_forest_nx(
            graph,
            weight_key=weight_key,
            maximum=True,
            bucket_count=self.bucket_count,
            parallel_workers=self.parallel_workers,
        )
