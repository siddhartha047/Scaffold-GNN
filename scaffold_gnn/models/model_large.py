import torch

from .spmm_adj import to_spmm_adj
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class LargeMPNNs(torch.nn.Module):
    """tunedGNN large_graph/lg_model.py architecture."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        local_layers=3,
        in_dropout=0.15,
        dropout=0.5,
        heads=1,
        pre_ln=False,
        bn=True,
        local_attn=False,
        res=True,
        ln=False,
        jk=False,
        sage=False,
    ):
        super().__init__()

        self.in_drop = in_dropout
        self.dropout = dropout
        self.pre_ln = pre_ln
        self.bn = bn
        self.res = res
        self.jk = jk
        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()
        if self.bn:
            self.bns = torch.nn.ModuleList()

        if local_attn:
            self.local_convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        elif sage:
            self.local_convs.append(SAGEConv(in_channels, hidden_channels))
        else:
            self.local_convs.append(GCNConv(in_channels, hidden_channels, cached=False, normalize=True))

        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.lns.append(torch.nn.LayerNorm(hidden_channels))
        if self.pre_ln:
            self.pre_lns.append(torch.nn.LayerNorm(in_channels))
        if self.bn:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        for _ in range(local_layers - 1):
            self.h_lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if local_attn:
                self.local_convs.append(GATConv(hidden_channels, hidden_channels, heads=heads))
            elif sage:
                self.local_convs.append(SAGEConv(hidden_channels, hidden_channels))
            else:
                self.local_convs.append(GCNConv(hidden_channels, hidden_channels, cached=False, normalize=True))

            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(hidden_channels))
            if self.bn:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.lin_in = torch.nn.Linear(in_channels, hidden_channels)
        self.ln = torch.nn.LayerNorm(hidden_channels)
        self.pred_local = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.pre_ln:
            for pre_ln in self.pre_lns:
                pre_ln.reset_parameters()
        if self.bn:
            for bn in self.bns:
                bn.reset_parameters()
        self.lin_in.reset_parameters()
        self.ln.reset_parameters()
        self.pred_local.reset_parameters()

    def _spmm(self, x, edge_index, edge_weight):
        """Swap the COO edge list for a fused SpMM adjacency; see models/spmm_adj.py."""
        adj, weight = to_spmm_adj(edge_index, edge_weight, x.size(0))
        return x, adj, weight

    def forward(self, x, edge_index, edge_weight=None):
        x, edge_index, edge_weight = self._spmm(x, edge_index, edge_weight)
        x = F.dropout(x, p=self.in_drop, training=self.training)

        x_final = 0
        for i, local_conv in enumerate(self.local_convs):
            if self.pre_ln:
                x = self.pre_lns[i](x)
            if isinstance(local_conv, GCNConv):
                convolved = local_conv(x, edge_index, edge_weight=edge_weight)
            else:
                convolved = local_conv(x, edge_index)
            if self.res:
                x = convolved + self.lins[i](x)
            else:
                x = convolved
            if self.bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.jk:
                x_final = x_final + x
            else:
                x_final = x

        return self.pred_local(x_final)
