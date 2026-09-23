import networkit as nk
import torch
from torch_geometric.data import Data

from .base import BaseSparsifier, deterministic_nk_rng


class RandomDropout(BaseSparsifier):
    def __init__(self, drop_prob=0.5, target_ratio=None, seed=None):
        self.drop_prob = float(drop_prob)
        self.target_ratio = target_ratio
        self.seed = seed
        if self.target_ratio is None and not 0.0 <= self.drop_prob <= 1.0:
            raise ValueError('drop_prob must be in [0, 1]')

    def sparsify(self, data_or_graph):
        if self.target_ratio is not None:
            ratio = max(1e-6, min(1.0, float(self.target_ratio)))
        else:
            ratio = 1.0 - self.drop_prob

        if isinstance(data_or_graph, Data):
            return self._sparsify_pyg(data_or_graph, ratio)

        # RandomEdgeScore is drawn in parallel from a shared RNG, so the seed
        # alone does not pin the support -- see deterministic_nk_rng.
        with deterministic_nk_rng(nk, self.seed):
            nk_sp = nk.sparsification.RandomEdgeSparsifier()
            return self._sparsify_via_nk_exact(data_or_graph, nk_sp, ratio)

    def _sparsify_pyg(self, data, ratio):
        edge_index = data.edge_index.detach().cpu()
        src = edge_index[0]
        dst = edge_index[1]
        non_loop = src != dst

        if bool(getattr(data, "edge_index_is_undirected_unique", False)):
            pairs = edge_index[:, non_loop]
        elif bool(getattr(data, "edge_index_is_symmetric_unique", False)):
            pairs = edge_index[:, non_loop & (src < dst)]
        else:
            lo = torch.minimum(src[non_loop], dst[non_loop])
            hi = torch.maximum(src[non_loop], dst[non_loop])
            pairs = torch.unique(torch.stack((lo, hi), dim=0), dim=1)

        edge_count = int(pairs.size(1))
        if edge_count == 0:
            out_edge_index = torch.empty((2, 0), dtype=torch.long)
            num_undirected_edges = 0
        else:
            keep_count = max(1, min(edge_count, round(edge_count * ratio)))
            if keep_count >= edge_count:
                kept_pairs = pairs
            else:
                generator = torch.Generator(device="cpu")
                if self.seed is not None:
                    generator.manual_seed(int(self.seed))
                perm = torch.randperm(edge_count, generator=generator)[:keep_count]
                kept_pairs = pairs[:, perm]
            out_edge_index = torch.cat((kept_pairs, kept_pairs.flip(0)), dim=1).contiguous()
            num_undirected_edges = int(kept_pairs.size(1))

        out = Data(
            x=data.x,
            edge_index=out_edge_index,
            y=data.y,
            num_nodes=data.num_nodes,
        )
        for key in ("train_mask", "val_mask", "test_mask"):
            if hasattr(data, key):
                setattr(out, key, getattr(data, key))
        out.edge_index_is_symmetric_unique = True
        out.num_undirected_edges = num_undirected_edges
        return out
