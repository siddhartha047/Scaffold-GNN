"""Exact tunedGNN medium-graph GCN architecture for GraphSAINT batches.

The parameterized architecture comes directly from this repository's tunedGNN
implementation. This subclass only extends the call boundary so GraphSAINT's
cached edge correction can be supplied during sampled training and so large
graphs can be evaluated layer by layer.

The one thing that cannot be inherited is *where* the GCN normalization is
computed. ``GCNConv(normalize=True)`` re-derives ``D^-1/2 A D^-1/2`` from the
edge weights it is handed, so feeding it GraphSAINT's ``edge_norm`` would both
destroy the sampling-bias correction and normalize against subgraph degrees
instead of full-graph ones. Both training and layer-wise inference therefore
supply precomputed full-graph coefficients and leave ``normalize=False``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import degree


SUPPORT_GRAPH_ROOT = Path(
    os.environ.get("SUPPORT_GRAPH_ROOT", Path(__file__).resolve().parents[2])
).resolve()
TUNEDGNN_MODEL_PATH = (
    SUPPORT_GRAPH_ROOT
    / "RelatedMethods"
    / "tunedGNN-main"
    / "medium_graph"
    / "model.py"
)


def _load_tunedgnn_mpnn():
    # Load the upstream file directly. Importing Benchmark's package-level
    # re-export would also import optional DGL models that this method does not
    # need.
    spec = importlib.util.spec_from_file_location(
        "_tunedgnn_graphsaint_medium_model", TUNEDGNN_MODEL_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load tunedGNN model: {TUNEDGNN_MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MPNNs


MPNNs = _load_tunedgnn_mpnn()


class TunedGNNGCN(MPNNs):
    """tunedGNN ``MPNNs(..., gnn='gcn')`` with sampled-edge support.

    No trainable module is added or replaced. Only the edge coefficients handed
    to each ``GCNConv`` change, and they are always the full-graph GCN weights
    -- optionally multiplied by GraphSAINT's ``edge_norm`` during normalized
    sampled training.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.5,
        input_dropout: float = 0.0,
        pre_linear: bool = False,
        residual: bool = False,
        layer_norm: bool = False,
        batch_norm: bool = False,
        multilabel: bool = False,
    ):
        super().__init__(
            in_channels,
            hidden_channels,
            out_channels,
            local_layers=num_layers,
            dropout=dropout,
            heads=1,
            pre_ln=False,
            pre_linear=pre_linear,
            res=residual,
            ln=layer_norm,
            bn=batch_norm,
            jk=False,
            gnn="gcn",
        )
        # The propagation coefficients are precomputed on the full graph, so
        # each conv must consume them verbatim instead of re-normalizing the
        # subgraph it happens to see.
        for local_conv in self.local_convs:
            local_conv.normalize = False
            local_conv.add_self_loops = False
        # tunedGNN's medium model has no input-dropout stage; its large-graph
        # model opens with one (`in_dropout`, 0.15 on Reddit). Apply it here so
        # the resolved preset is honored. It is a no-op at the medium-graph
        # default of 0.0.
        self.input_dropout = float(input_dropout)
        self.multilabel = bool(multilabel)

    # ------------------------------------------------------------------
    # Full-graph GCN coefficients
    # ------------------------------------------------------------------
    @staticmethod
    def prepare_data(data) -> None:
        """Attach full-graph ``D^-1/2 A~ D^-1/2`` coefficients to ``data``.

        ``gcn_norm`` is edge-level and ``gcn_self_norm`` node-level, so the
        GraphSAINT and NeighborLoader collates index them alongside the graph
        they already slice. Degrees include the implicit GCN self-loop, but the
        self-loops themselves stay out of ``data.edge_index``: keeping the edge
        set untouched is what lets this method reuse a ``node_norm`` /
        ``edge_norm`` cache computed for the plain GraphSAINT baseline.
        """
        row, col = data.edge_index
        # The loader is fed an undirected, self-loop-free, coalesced graph, so
        # in-degree and out-degree agree and `deg + 1` is the self-looped
        # degree GCN normalizes against.
        deg = degree(col, data.num_nodes, dtype=torch.float) + 1.0
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.0)
        data.gcn_norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        data.gcn_self_norm = 1.0 / deg

    def build_edge_inputs(self, batch, use_saint_norm: bool, device):
        """Full-graph GCN weights on the sampled edges, plus self-loops.

        A self-loop is present exactly when its node is, so GraphSAINT's
        ``alpha`` for it is 1 and only the real edges take the ``edge_norm``
        correction.
        """
        if not hasattr(batch, "gcn_norm"):
            raise AttributeError(
                "TunedGNNGCN requires TunedGNNGCN.prepare_data(data) to run "
                "before the loaders are built"
            )
        edge_index = batch.edge_index.to(device)
        edge_weight = batch.gcn_norm.to(device)
        if use_saint_norm:
            edge_weight = edge_weight * batch.edge_norm.to(device)
        return self._append_self_loops(
            edge_index, edge_weight, batch.gcn_self_norm.to(device)
        )

    @staticmethod
    def _append_self_loops(edge_index, edge_weight, self_norm):
        loop = torch.arange(self_norm.size(0), device=edge_index.device)
        edge_index = torch.cat([edge_index, loop.unsqueeze(0).repeat(2, 1)], dim=1)
        edge_weight = torch.cat([edge_weight, self_norm], dim=0)
        return edge_index, edge_weight

    def set_aggr(self, _aggr: str) -> None:
        """Keep GCNConv's native normalized-add aggregation unchanged."""

    def _format_output(self, logits: torch.Tensor) -> torch.Tensor:
        return logits if self.multilabel else logits.log_softmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.input_dropout > 0:
            x = F.dropout(x, p=self.input_dropout, training=self.training)

        if self.pre_linear:
            x = self.lin_in(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_final: torch.Tensor | int = 0
        for layer, local_conv in enumerate(self.local_convs):
            if self.res:
                x = local_conv(x, edge_index, edge_weight) + self.lins[layer](x)
            else:
                x = local_conv(x, edge_index, edge_weight)
            if self.ln:
                x = self.lns[layer](x)
            elif self.bn:
                x = self.bns[layer](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_final = x_final + x if self.jk else x

        return self._format_output(self.pred_local(x_final))

    @torch.no_grad()
    def inference(
        self,
        x_all: torch.Tensor,
        subgraph_loader: NeighborLoader,
        device: torch.device,
    ) -> torch.Tensor:
        """Run the unchanged tunedGNN layers with bounded evaluation memory.

        Each 1-hop batch carries the full-graph coefficients for its sampled
        edges, so seed-node outputs match full-graph evaluation even though the
        sampled neighbors' own degrees are truncated.
        """

        self.eval()
        x_all = x_all.cpu()
        if self.pre_linear:
            x_all = self._linear_on_cpu(self.lin_in, x_all, device)

        x_final = None
        for layer, local_conv in enumerate(self.local_convs):
            output = torch.empty(
                (x_all.size(0), local_conv.out_channels), dtype=x_all.dtype
            )
            for batch in subgraph_loader:
                node_ids = batch.n_id.cpu()
                batch_size = int(batch.batch_size)
                previous = x_all[node_ids].to(device)
                edge_index, edge_weight = self.build_edge_inputs(
                    batch, use_saint_norm=False, device=device
                )
                hidden = local_conv(previous, edge_index, edge_weight)
                hidden = hidden[:batch_size]
                if self.res:
                    hidden = hidden + self.lins[layer](previous[:batch_size])
                if self.ln:
                    hidden = self.lns[layer](hidden)
                elif self.bn:
                    hidden = self.bns[layer](hidden)
                output[node_ids[:batch_size]] = F.relu(hidden).cpu()
            x_all = output
            x_final = x_all if not self.jk else (
                x_all if x_final is None else x_final + x_all
            )

        if x_final is None:
            raise RuntimeError("tunedGNN must contain at least one GCN layer")
        logits = self._linear_on_cpu(self.pred_local, x_final, device)
        return self._format_output(logits)

    @staticmethod
    def _linear_on_cpu(
        linear: torch.nn.Linear,
        values: torch.Tensor,
        device: torch.device,
        block_size: int = 200_000,
    ) -> torch.Tensor:
        blocks = []
        for start in range(0, values.size(0), block_size):
            block = values[start : start + block_size].to(device)
            blocks.append(linear(block).cpu())
        return torch.cat(blocks, dim=0)
