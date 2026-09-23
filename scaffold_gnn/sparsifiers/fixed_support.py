"""Load a fixed, precomputed unweighted support for controlled experiments."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import torch
from torch_geometric.data import Data

from .base import BaseSparsifier


class FixedSupportSparsifier(BaseSparsifier):
    """Return exactly the support stored in ``support_path``.

    The artifact must be a ``torch.save`` dictionary containing either
    ``undirected_pairs`` or ``edge_index``.  The former is preferred because it
    makes the one-edge-per-undirected-pair contract explicit.
    """

    def __init__(self, support_path, support_sha256=None, target_ratio=None):
        self.support_path = str(Path(support_path).expanduser().resolve())
        # Included in the sparse-cache identity.  Validation is performed by
        # the experiment runner before training starts.
        self.support_sha256 = support_sha256
        self.target_ratio = target_ratio

    def _load_pairs(self, num_nodes: int) -> torch.Tensor:
        path = Path(self.support_path)
        if not path.is_file():
            raise FileNotFoundError(f"fixed support does not exist: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch before weights_only support.
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: expected a dictionary artifact")

        if "undirected_pairs" in payload:
            pairs = torch.as_tensor(payload["undirected_pairs"], dtype=torch.long)
        elif "edge_index" in payload:
            edges = torch.as_tensor(payload["edge_index"], dtype=torch.long)
            if edges.ndim != 2 or edges.size(0) != 2:
                raise ValueError(f"{path}: edge_index must have shape [2, m]")
            keep = edges[0] != edges[1]
            lo = torch.minimum(edges[0, keep], edges[1, keep])
            hi = torch.maximum(edges[0, keep], edges[1, keep])
            pairs = torch.unique(torch.stack((lo, hi), dim=0), dim=1)
        else:
            raise ValueError(
                f"{path}: missing 'undirected_pairs' or 'edge_index'"
            )

        if pairs.ndim != 2 or pairs.size(0) != 2:
            raise ValueError(f"{path}: undirected_pairs must have shape [2, m]")
        pairs = pairs.contiguous()
        if pairs.numel():
            if int(pairs.min()) < 0 or int(pairs.max()) >= int(num_nodes):
                raise ValueError(
                    f"{path}: node id lies outside [0, {int(num_nodes) - 1}]"
                )
            if bool((pairs[0] >= pairs[1]).any()):
                raise ValueError(
                    f"{path}: undirected_pairs must be canonical with u < v"
                )
            if torch.unique(pairs, dim=1).size(1) != pairs.size(1):
                raise ValueError(f"{path}: duplicate undirected edges")
        expected_nodes = payload.get("num_nodes")
        if expected_nodes is not None and int(expected_nodes) != int(num_nodes):
            raise ValueError(
                f"{path}: support has {int(expected_nodes)} nodes; "
                f"dataset has {int(num_nodes)}"
            )
        return pairs

    def sparsify(self, data_or_graph):
        if isinstance(data_or_graph, Data):
            pairs = self._load_pairs(int(data_or_graph.num_nodes))
            edge_index = torch.cat((pairs, pairs.flip(0)), dim=1).contiguous()
            out = Data(
                x=data_or_graph.x,
                edge_index=edge_index,
                y=data_or_graph.y,
                num_nodes=data_or_graph.num_nodes,
            )
            for name in ("train_mask", "val_mask", "test_mask"):
                if hasattr(data_or_graph, name):
                    setattr(out, name, getattr(data_or_graph, name))
            out.edge_index_is_symmetric_unique = True
            out.num_undirected_edges = int(pairs.size(1))
            return out

        graph = self._ensure_nx(data_or_graph)
        pairs = self._load_pairs(graph.number_of_nodes())
        out = nx.Graph()
        out.add_nodes_from(graph.nodes(data=True))
        nodes = list(graph.nodes())
        out.add_edges_from(
            (nodes[int(u)], nodes[int(v)]) for u, v in pairs.t().tolist()
        )
        return out
