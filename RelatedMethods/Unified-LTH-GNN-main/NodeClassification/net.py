import torch
import torch.nn as nn
import pdb
import copy
import utils
from torch_geometric.nn import GCNConv

class net_gcn(nn.Module):

    def __init__(
        self,
        embedding_dim,
        adj,
        dropout=0.5,
        input_dropout=0.0,
        pre_linear=False,
        residual=False,
        layer_norm=False,
        batch_norm=False,
        jumping_knowledge=False,
        tuned_backbone=False,
    ):
        super().__init__()

        self.tuned_backbone = bool(tuned_backbone)
        self.layer_num = (
            len(embedding_dim) - 2
            if self.tuned_backbone
            else len(embedding_dim) - 1
        )
        self.input_dropout = float(input_dropout)
        self.pre_linear = bool(pre_linear)
        self.residual = bool(residual)
        self.layer_norm = bool(layer_norm)
        self.batch_norm = bool(batch_norm)
        self.jumping_knowledge = bool(jumping_knowledge)
        if self.tuned_backbone and self.pre_linear:
            message_dims = [embedding_dim[1]] * (self.layer_num + 1)
        else:
            message_dims = embedding_dim[: self.layer_num + 1]
        self.net_layer = nn.ModuleList(
            GCNConv(
                message_dims[ln],
                message_dims[ln + 1],
                cached=False,
                normalize=True,
                bias=self.tuned_backbone,
            )
            for ln in range(self.layer_num)
        )
        self.residual_lins = nn.ModuleList(
            nn.Linear(
                message_dims[ln],
                message_dims[ln + 1],
                bias=True,
            )
            for ln in range(
                self.layer_num
                if self.tuned_backbone
                else self.layer_num - 1
            )
        )
        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(message_dims[ln + 1])
            for ln in range(
                self.layer_num
                if self.tuned_backbone
                else self.layer_num - 1
            )
        )
        self.batch_norms = nn.ModuleList(
            nn.BatchNorm1d(message_dims[ln + 1])
            for ln in range(
                self.layer_num
                if self.tuned_backbone
                else self.layer_num - 1
            )
        )
        self.lin_in = (
            nn.Linear(embedding_dim[0], embedding_dim[1])
            if self.tuned_backbone and self.pre_linear
            else None
        )
        self.pred_local = (
            nn.Linear(embedding_dim[-2], embedding_dim[-1])
            if self.tuned_backbone
            else None
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=float(dropout))
        self.adj_nonzero = torch.nonzero(adj, as_tuple=False).shape[0]
        self.adj_mask1_train = nn.Parameter(self.generate_adj_mask(adj))
        self.adj_mask2_fixed = nn.Parameter(self.generate_adj_mask(adj), requires_grad=False)
    
    def forward(self, x, adj, val_test=False, full_graph=False):
        
        if full_graph:
            masked_adj = adj
        else:
            masked_adj = torch.mul(adj, self.adj_mask1_train)
            masked_adj = torch.mul(masked_adj, self.adj_mask2_fixed)
        edge_index = torch.nonzero(adj, as_tuple=False).t().contiguous()
        edge_weight = masked_adj[edge_index[0], edge_index[1]]
        if self.input_dropout > 0 and not val_test:
            x = nn.functional.dropout(
                x, p=self.input_dropout, training=True
            )
        if self.tuned_backbone and self.pre_linear:
            x = self.lin_in(x)
            x = nn.functional.dropout(
                x,
                p=self.dropout.p,
                training=self.training and not val_test,
            )
        x_final = 0
        for ln in range(self.layer_num):
            previous = x
            x = self.net_layer[ln](
                x,
                edge_index,
                edge_weight=edge_weight,
            )
            if not self.tuned_backbone and ln == self.layer_num - 1:
                break
            if self.residual:
                x = x + self.residual_lins[ln](previous)
            if self.layer_norm:
                x = self.layer_norms[ln](x)
            elif self.batch_norm:
                x = self.batch_norms[ln](x)
            x = self.relu(x)
            x = nn.functional.dropout(
                x,
                p=self.dropout.p,
                training=self.training and not val_test,
            )
            if self.tuned_backbone:
                x_final = x_final + x if self.jumping_knowledge else x
        if self.tuned_backbone:
            return self.pred_local(x_final)
        return x

    def generate_adj_mask(self, input_adj):
        
        sparse_adj = input_adj
        zeros = torch.zeros_like(sparse_adj)
        ones = torch.ones_like(sparse_adj)
        mask = torch.where(sparse_adj != 0, ones, zeros)
        return mask


class net_gcn_admm(nn.Module):

    def __init__(self, embedding_dim, adj):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.adj_nonzero = torch.nonzero(adj, as_tuple=False).shape[0]
        self.adj_layer1 = nn.Parameter(copy.deepcopy(adj), requires_grad=True)
        self.adj_layer2 = nn.Parameter(copy.deepcopy(adj), requires_grad=True)
        
    def forward(self, x, adj, val_test=False):

        for ln in range(self.layer_num):
            if ln == 0:
                x = torch.mm(self.adj_layer1, x)
            elif ln == 1:
                x = torch.mm(self.adj_layer2, x)
            else:
                assert False
            x = self.net_layer[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)
        return x

    # def forward(self, x, adj, val_test=False):

    #     for ln in range(self.layer_num):
    #         x = torch.mm(self.adj_list[ln], x)
    #         x = self.net_layer[ln](x)
    #         if ln == self.layer_num - 1:
    #             break
    #         x = self.relu(x)
    #         if val_test:
    #             continue
    #         x = self.dropout(x)
    #     return x

    def generate_adj_mask(self, input_adj):
        
        sparse_adj = input_adj
        zeros = torch.zeros_like(sparse_adj)
        ones = torch.ones_like(sparse_adj)
        mask = torch.where(sparse_adj != 0, ones, zeros)
        return mask

class net_gcn_baseline(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, adj, val_test=False):

        for ln in range(self.layer_num):
            x = torch.mm(adj, x)
            # x = torch.spmm(adj, x)
            x = self.net_layer[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)
        return x


class net_gcn_multitask(nn.Module):

    def __init__(self, embedding_dim, ss_dim):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
        self.ss_classifier = nn.Linear(embedding_dim[-2], ss_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        

    def forward(self, x, adj, val_test=False):

        x_ss = x

        for ln in range(self.layer_num):
            x = torch.spmm(adj, x)
            x = self.net_layer[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)

        if not val_test:
            for ln in range(self.layer_num):
                x_ss = torch.spmm(adj, x_ss)
                if ln == self.layer_num - 1:
                    break
                x_ss = self.net_layer[ln](x_ss)
                x_ss = self.relu(x_ss)
                x_ss = self.dropout(x_ss)
            x_ss = self.ss_classifier(x_ss)

        return x, x_ss
