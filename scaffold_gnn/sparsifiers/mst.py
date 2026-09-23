import networkx as nx
import torch
from torch_geometric.data import Data

from .base import BaseSparsifier
from .scaffold.spanning_tree import (
    build_spanning_forest_mask,
    build_spanning_forest_nx,
)


class MSTSparsifier(BaseSparsifier):
    """Minimum spanning tree via the fast native Kruskal union-find.

    Uses the same builder as ``MaxSTSparsifier`` (light edges first). For
    unweighted graphs the result is any spanning tree/forest.
    """

    def __init__(self, algorithm='kruskal', weight=None, target_ratio=None):
        # Kept for backward-compat with existing configs; the fast path always
        # uses Kruskal-with-union-find.
        self.algorithm = str(algorithm)
        self.weight = weight
        self.target_ratio = target_ratio

    def sparsify(self, data_or_graph):
        if isinstance(data_or_graph, Data) and self.weight is None:
            pairs = self._pyg_undirected_pairs(data_or_graph)
            mask = build_spanning_forest_mask(
                int(data_or_graph.num_nodes),
                pairs[0].numpy(),
                pairs[1].numpy(),
            )
            kept = pairs[:, torch.from_numpy(mask)]
            edge_index = torch.cat((kept, kept.flip(0)), dim=1).contiguous()
            data = Data(
                x=data_or_graph.x,
                edge_index=edge_index,
                y=data_or_graph.y,
                num_nodes=data_or_graph.num_nodes,
            )
            data.edge_index_is_symmetric_unique = True
            data.num_undirected_edges = int(kept.size(1))
            return data

        G = self._ensure_nx(data_or_graph)
        if G.is_directed():
            raise ValueError('MSTSparsifier requires an undirected graph')
        weight_key = self.weight if (self.weight and nx.is_weighted(G)) else None
        H = build_spanning_forest_nx(
            G,
            weight_key=weight_key,
            maximum=False,
        )
        if isinstance(data_or_graph, Data):
            return self._ensure_pyg(H, original_data=data_or_graph)
        return H
