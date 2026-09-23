"""PyG model definitions from tunedGNN-org ``large_graph/product.py``."""

import torch

from .spmm_adj import to_spmm_adj
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv

# Shared with the default profile so both paths build the same GIN aggregator
# rather than drifting into two architectures under one name.
from .model import _gin_mlp


class GNNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout, ln, gnn="gcn"):
        super().__init__()
        self.norm = LayerNorm(in_channels, elementwise_affine=True)
        if gnn == "gcn":
            self.conv = GCNConv(in_channels, out_channels)
        elif gnn == "sage":
            self.conv = SAGEConv(in_channels, out_channels)
        elif gnn == "gat":
            self.conv = GATConv(in_channels, out_channels)
        elif gnn == "gin":
            # train_eps matches the default profile. GINConv implements
            # message_and_aggregate, so it takes the fused SpMM adjacency
            # ProductsGNN.forward builds, like GCN/SAGE and unlike nothing here.
            self.conv = GINConv(_gin_mlp(in_channels, out_channels),
                                train_eps=True)
        else:
            raise ValueError(f"Unsupported products GNN: {gnn}")
        self.dropout = dropout
        self.ln = ln
        self.gnn = gnn

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, *, eval_chunks=1):
        x = self.norm(x).relu() if self.ln else x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.gnn == "gcn":
            return self.conv(x, edge_index, edge_weight=edge_weight)
        if self.gnn == "gat" and not self.training and eval_chunks > 1:
            # Batch destinations, retaining ALL source neighbors for each one.
            # Induced node partitions would silently discard crossing edges.
            from torch_sparse import SparseTensor

            adj = edge_index
            if isinstance(adj, torch.Tensor):
                adj = SparseTensor(row=adj[1], col=adj[0],
                                   sparse_sizes=(x.size(0), x.size(0)))
            add_loops = self.conv.add_self_loops
            if add_loops:
                adj = adj.set_diag()
            # Local destination indices no longer equal global source indices.
            # Insert self-loops BEFORE slicing, never inside the bipartite conv.
            self.conv.add_self_loops = False
            size = max(1, (x.size(0) + int(eval_chunks) - 1) // int(eval_chunks))
            try:
                return torch.cat([
                    self.conv((x, x[start:min(start + size, x.size(0))]),
                              adj[start:min(start + size, x.size(0))])
                    for start in range(0, x.size(0), size)
                ], dim=0)
            finally:
                self.conv.add_self_loops = add_loops
        return self.conv(x, edge_index)


class ProductsGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, ln, gnn, jk, res):
        super().__init__()
        self.dropout = dropout
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.norm = LayerNorm(hidden_channels, elementwise_affine=True)
        self.ln = ln
        self.jk = jk
        self.res = res
        self.gat_eval_chunks = 1
        self.convs = torch.nn.ModuleList(
            [GNNConv(hidden_channels, hidden_channels, dropout, ln, gnn) for _ in range(num_layers)]
        )

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.norm.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        # Fused SpMM instead of COO message passing; see models/spmm_adj.py.
        edge_index, edge_weight = to_spmm_adj(edge_index, edge_weight, x.size(0))
        x_final = 0
        x = self.lin1(x)
        x_final += x
        for conv in self.convs:
            convolved = conv(x, edge_index, edge_weight=edge_weight,
                             eval_chunks=self.gat_eval_chunks)
            x = convolved + x if self.res else convolved
            x_final += x
        x = self.norm(x).relu() if self.ln else x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.jk:
            x = x_final
        return self.lin2(x)
