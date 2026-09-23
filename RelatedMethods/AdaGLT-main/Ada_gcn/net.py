"""AdaGLT edge-mask learner with a tunedGNN-compatible dense GCN backbone."""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj

import utils


class net_gcn_dense(nn.Module):
    """Dense differentiable adjacency masks around the tunedGNN GCN.

    ``embedding_dim`` is ``[input, hidden, ..., hidden]`` and contains one
    hidden width per message-passing layer. Classification is performed by a
    separate predictor, exactly as in tunedGNN. Model weights are never masked;
    AdaGLT learns only graph-edge masks.
    """

    def __init__(
        self,
        embedding_dim,
        edge_index,
        device,
        spar_wei,
        spar_adj,
        num_nodes,
        use_res,
        use_bn,
        use_ln=False,
        dropout=0.5,
        input_dropout=0.0,
        pre_linear=False,
        jumping_knowledge=False,
        out_channels=None,
        coef=None,
        mode="prune",
    ):
        super().__init__()
        if spar_wei:
            raise ValueError(
                "AdaGLT is configured for edge-only sparsification; "
                "model-weight masks are disabled"
            )
        if out_channels is None:
            raise ValueError("out_channels is required")

        self.mode = mode
        self.num_nodes = int(num_nodes)
        self.adj_binary = to_dense_adj(
            edge_index, max_num_nodes=self.num_nodes
        )[0]
        self.layer_num = len(embedding_dim) - 1
        self.spar_wei = False
        self.spar_adj = bool(spar_adj)
        self.edge_mask_archive = []
        self.coef = coef
        self.use_bn = bool(use_bn)
        self.use_ln = bool(use_ln)
        self.use_res = bool(use_res)
        self.pre_linear = bool(pre_linear)
        self.jumping_knowledge = bool(jumping_knowledge)
        self.input_dropout = float(input_dropout)
        self.device = device

        hidden_channels = int(embedding_dim[-1])
        if self.pre_linear:
            message_dims = [hidden_channels] * (self.layer_num + 1)
            self.lin_in = nn.Linear(embedding_dim[0], hidden_channels)
        else:
            message_dims = list(embedding_dim)
            self.lin_in = None

        self.net_layer = nn.ModuleList(
            GCNConv(
                message_dims[layer],
                message_dims[layer + 1],
                cached=False,
                normalize=True,
                bias=True,
            )
            for layer in range(self.layer_num)
        )
        self.residual_lins = nn.ModuleList(
            nn.Linear(
                message_dims[layer],
                message_dims[layer + 1],
                bias=True,
            )
            for layer in range(self.layer_num)
        )
        self.norms = nn.ModuleList(
            (
                nn.BatchNorm1d(message_dims[layer + 1])
                if self.use_bn
                else nn.LayerNorm(message_dims[layer + 1])
            )
            for layer in range(self.layer_num)
        ) if (self.use_bn or self.use_ln) else nn.ModuleList()
        self.pred_local = nn.Linear(hidden_channels, int(out_channels))
        self.dropout = float(dropout)

        if self.spar_adj:
            self.adj_thresholds = nn.ParameterList(
                nn.Parameter(
                    torch.ones(size=(num_nodes,))
                    * utils.initalize_thres(coef)
                )
                for _ in range(self.layer_num)
            )
            self.edge_learner = nn.Sequential(
                nn.Linear(embedding_dim[0] * 2, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1),
            )

    def backbone_parameters(self):
        modules = [
            self.net_layer,
            self.residual_lins,
            self.norms,
            self.pred_local,
        ]
        if self.lin_in is not None:
            modules.append(self.lin_in)
        for module in modules:
            yield from module.parameters()

    def sparsifier_parameters(self):
        if not self.spar_adj:
            return
        yield from self.adj_thresholds.parameters()
        yield from self.edge_learner.parameters()

    def _prepare_input(self, x, val_test):
        if self.input_dropout > 0:
            x = F.dropout(
                x,
                p=self.input_dropout,
                training=self.training and not val_test,
            )
        if self.lin_in is not None:
            x = self.lin_in(x)
            x = F.dropout(
                x,
                p=self.dropout,
                training=self.training and not val_test,
            )
        return x

    def _hidden_step(self, x, previous, layer, val_test):
        if self.use_res:
            x = x + self.residual_lins[layer](previous)
        if self.use_bn or self.use_ln:
            x = self.norms[layer](x)
        x = F.relu(x)
        return F.dropout(
            x,
            p=self.dropout,
            training=self.training and not val_test,
        )

    def forward_retain(self, x, edge_index, val_test, edge_masks):
        x = self._prepare_input(x, val_test)
        x_final = 0
        source, target = edge_index
        for layer in range(self.layer_num):
            previous = x
            edge_weight = (
                edge_masks[layer][source, target]
                if edge_masks
                else None
            )
            x = self.net_layer[layer](
                x,
                edge_index,
                edge_weight=edge_weight,
            )
            x = self._hidden_step(x, previous, layer, val_test)
            x_final = x_final + x if self.jumping_knowledge else x
        return self.pred_local(x_final)

    def forward(self, x, edge_index, val_test=False, **kwargs):
        if self.mode == "retain":
            return self.forward_retain(
                x,
                edge_index,
                val_test,
                kwargs["edge_masks"],
            )

        pretrain = bool(kwargs.get("pretrain", False))
        edge_weight = None
        if self.spar_adj:
            edge_weight = self.learn_soft_edge(x, edge_index)
            adjacency_mask = self.adj_binary
            self.edge_mask_archive = []
        adjacency_original = to_dense_adj(
            edge_index,
            edge_attr=edge_weight,
            max_num_nodes=self.num_nodes,
        )[0]

        x = self._prepare_input(x, val_test)
        x_final = 0
        source, target = edge_index
        for layer in range(self.layer_num):
            previous = x
            adjacency = adjacency_original
            if self.spar_adj and not pretrain:
                adjacency_mask = self.adj_pruning2(
                    adjacency_original,
                    self.adj_thresholds[layer],
                    adjacency_mask,
                )
                self.edge_mask_archive.append(
                    copy.deepcopy(adjacency_mask.detach().cpu())
                )
                adjacency = adjacency_mask * adjacency_original
            layer_edge_weight = (
                adjacency[source, target]
                if not pretrain or edge_weight is not None
                else None
            )
            x = self.net_layer[layer](
                x,
                edge_index,
                edge_weight=layer_edge_weight,
            )
            x = self._hidden_step(x, previous, layer, val_test)
            x_final = x_final + x if self.jumping_knowledge else x
        return self.pred_local(x_final)

    def learn_soft_edge(self, x, edge_index):
        row, col = edge_index
        learner_input = torch.cat([x[row], x[col]], dim=1)
        edge_weight = self.edge_learner(learner_input).squeeze(-1)
        edge_weight = torch.nan_to_num(edge_weight)
        mean = edge_weight.mean()
        std = edge_weight.std(unbiased=False).detach().clamp_min(1e-3)
        edge_weight = ((edge_weight - mean) * (0.01 / std)) + 1
        return torch.nan_to_num(
            edge_weight, nan=1.0, posinf=1.0, neginf=1.0
        )

    def adj_pruning2(self, adjacency, threshold, previous_mask, tau=0.1):
        edge_weight = adjacency[adjacency.nonzero(as_tuple=True)]
        edge_index = adjacency.nonzero().t().contiguous()
        edge_weight = torch.nan_to_num(edge_weight)
        mean = edge_weight.mean()
        std = edge_weight.std(unbiased=False).detach().clamp_min(1e-2)
        edge_weight = (edge_weight - mean) * (0.1 / std)
        dense_scores = to_dense_adj(
            edge_index,
            edge_attr=edge_weight,
            max_num_nodes=self.num_nodes,
        )[0]
        transformed = self.coef * (
            torch.pow(threshold, 3) + 20 * threshold
        )
        soft = torch.sigmoid(
            (dense_scores - transformed.view(-1, 1)) / tau
        )
        soft = torch.nan_to_num(
            soft, nan=0.0, posinf=1.0, neginf=0.0
        )
        hard = (
            soft + torch.eye(self.num_nodes, device=self.device) > 0.5
        ).float()
        result = hard - soft.detach() + soft
        result = result * previous_mask
        return torch.nan_to_num(
            result, nan=0.0, posinf=1.0, neginf=0.0
        )

    def generate_wei_mask(self):
        return []
