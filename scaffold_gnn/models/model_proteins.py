"""GCN/GraphSAGE adapter for tunedGNN-org's OGBN-Proteins model."""

from __future__ import annotations

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv, SAGEConv


class ProteinGNN(nn.Module):
    """Preserve tunedGNN's DGL layer construction with a PyG call boundary."""

    def __init__(self, node_feats, n_classes, *, n_layers, n_heads, n_hidden,
                 dropout, input_drop, mpnn, jumping_knowledge=False):
        super().__init__()
        if mpnn not in {"gcn", "sage"}:
            raise ValueError("The proteins adapter supports gcn and sage")
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.n_hidden = int(n_hidden)
        self.mpnn = mpnn
        self.jk = bool(jumping_knowledge)
        self.node_encoder = nn.Linear(node_feats, n_hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.edge_encoder = nn.ModuleList()
        for layer in range(self.n_layers):
            in_hidden = n_heads * n_hidden if layer > 0 else n_hidden
            out_hidden = n_heads * n_hidden
            # Retained although unused by GCN/SAGE to preserve initialization.
            self.edge_encoder.append(nn.Linear(8, 16))
            if mpnn == "sage":
                conv = SAGEConv(in_hidden, out_hidden, aggregator_type="mean")
            else:
                # Sparse samples may isolate nodes. This only disables DGL's
                # zero-in-degree exception; non-isolated message passing is
                # unchanged from the tunedGNN GraphConv.
                conv = GraphConv(
                    in_hidden, out_hidden, allow_zero_in_degree=True
                )
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(out_hidden))
        self.pred_linear = nn.Linear(n_heads * n_hidden, n_classes)
        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        for encoder in self.edge_encoder:
            encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        self.pred_linear.reset_parameters()

    def forward_blocks(self, blocks, x):
        """Run the exact layerwise DGL-block computation from tunedGNN."""

        hidden = self.input_drop(F.relu(self.node_encoder(x), inplace=True))
        local_outputs = []
        previous = None
        for block, conv, norm in zip(blocks, self.convs, self.norms):
            if self.mpnn == "gcn":
                edge_weight = (
                    block.edata["edge_weight"]
                    if "edge_weight" in block.edata else None
                )
                hidden = conv(
                    block, hidden, edge_weight=edge_weight
                ).flatten(1, -1)
            else:
                hidden = conv(block, hidden).flatten(1, -1)
            if previous is not None:
                hidden = hidden + previous[: hidden.shape[0], :]
            previous = hidden
            hidden = self.dropout(F.relu(norm(hidden), inplace=True))
            local_outputs.append(hidden)
        if self.jk:
            hidden = torch.stack([value[: hidden.shape[0], :] for value in local_outputs]).sum(dim=0)
        return self.pred_linear(hidden)

    def forward(self, x, edge_index, edge_weight=None):
        graph = dgl.graph(
            (edge_index[0], edge_index[1]),
            num_nodes=x.size(0),
            device=x.device,
        )
        if edge_weight is not None:
            graph.edata["edge_weight"] = edge_weight
        return self.forward_blocks([graph] * self.n_layers, x)
