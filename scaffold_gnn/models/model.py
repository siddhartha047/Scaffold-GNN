import torch

from .spmm_adj import to_spmm_adj
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv
import torch.nn as nn

def _gin_mlp(in_channels, out_channels):
    """GIN's learnable aggregator, as in PyG's examples/jit/gin.py."""
    return nn.Sequential(
        nn.Linear(in_channels, out_channels), nn.ReLU(),
        nn.Linear(out_channels, out_channels),
    )


class MPNNs(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, local_layers=3, dropout=0.5, heads=1, pre_ln=False, pre_linear=False, res=False, ln=False, bn=False, jk=False, gnn='gcn'):
        super(MPNNs, self).__init__()
        self.dropout = dropout
        self.pre_ln = pre_ln
        self.pre_linear = pre_linear
        self.res = res
        self.ln = ln
        self.bn = bn
        self.jk = jk
        self.gnn = gnn
        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()
        self.lin_in = torch.nn.Linear(in_channels, hidden_channels)
        if not self.pre_linear:
            if gnn == 'gat':
                self.local_convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True, add_self_loops=False, bias=False))
            elif gnn == 'gin':
                self.local_convs.append(GINConv(_gin_mlp(in_channels, hidden_channels), train_eps=True))
            elif gnn == 'sage':
                self.local_convs.append(SAGEConv(in_channels, hidden_channels))
            else:
                self.local_convs.append(GCNConv(in_channels, hidden_channels, cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(in_channels))
            local_layers = local_layers - 1
        for _ in range(local_layers):
            if gnn == 'gat':
                self.local_convs.append(GATConv(hidden_channels, hidden_channels, heads=heads, concat=True, add_self_loops=False, bias=False))
            elif gnn == 'gin':
                self.local_convs.append(GINConv(_gin_mlp(hidden_channels, hidden_channels), train_eps=True))
            elif gnn == 'sage':
                self.local_convs.append(SAGEConv(hidden_channels, hidden_channels))
            else:
                self.local_convs.append(GCNConv(hidden_channels, hidden_channels, cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(hidden_channels))
        self.pred_local = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        self.lin_in.reset_parameters()
        self.pred_local.reset_parameters()

    def _spmm(self, x, edge_index, edge_weight):
        """Swap the COO edge list for a fused SpMM adjacency; see models/spmm_adj.py."""
        adj, weight = to_spmm_adj(edge_index, edge_weight, x.size(0),
                                  drop_values=self.gnn != 'gcn')
        return x, adj, weight

    def forward(self, x, edge_index, edge_weight=None):
        x, edge_index, edge_weight = self._spmm(x, edge_index, edge_weight)
        if self.pre_linear:
            x = self.lin_in(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_final = 0
        for i, local_conv in enumerate(self.local_convs):
            if isinstance(local_conv, GCNConv):
                convolved = local_conv(x, edge_index, edge_weight=edge_weight)
            else:
                convolved = local_conv(x, edge_index)
            if self.res:
                x = convolved + self.lins[i](x)
            else:
                x = convolved
            if self.ln:
                x = self.lns[i](x)
            elif self.bn:
                x = self.bns[i](x)
            else:
                pass
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.jk:
                x_final = x_final + x
            else:
                x_final = x
        x = self.pred_local(x_final)
        return x
