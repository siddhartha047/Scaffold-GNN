import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeMLP(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        local_layers=3,
        dropout=0.5,
        in_dropout=0.0,
        pre_linear=False,
        res=False,
        ln=False,
        bn=False,
    ):
        super().__init__()
        self.dropout = dropout
        self.in_dropout = in_dropout
        self.pre_linear = pre_linear
        self.res = res
        self.ln = ln
        self.bn = bn

        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.bns = nn.ModuleList()

        input_dim = hidden_channels if pre_linear else in_channels
        for _ in range(local_layers):
            self.layers.append(nn.Linear(input_dim, hidden_channels))
            self.lns.append(nn.LayerNorm(hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            input_dim = hidden_channels

        self.pred = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.pred.reset_parameters()

    def forward(self, x, edge_index=None, edge_weight=None):
        # Keep the graph-free model compatible with the common node-model
        # evaluation contract.  Both graph arguments are intentionally ignored.
        x = F.dropout(x, p=self.in_dropout, training=self.training)
        if self.pre_linear:
            x = self.input_proj(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        for idx, layer in enumerate(self.layers):
            x_next = layer(x)
            if self.res and x_next.shape == x.shape:
                x = x_next + x
            else:
                x = x_next

            if self.ln:
                x = self.lns[idx](x)
            elif self.bn:
                x = self.bns[idx](x)

            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.pred(x)
