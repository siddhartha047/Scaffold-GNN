import os
import subprocess
import tempfile
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from .base import BaseSparsifier


class ERSparsifier(BaseSparsifier):
    """Effective-resistance sparsifier using the gSparse C++ binary.

    Output size is controlled by epsilon (smaller epsilon → denser output).
    ER values are expensive to compute; pass er_cache_path to save/reuse them
    across calls (mirrors the paper's stage3.npz caching strategy).
    """

    def __init__(self, epsilon=0.5, er_cache_path=None):
        if epsilon <= 0:
            raise ValueError('epsilon must be > 0')
        self.epsilon = float(epsilon)
        self.er_cache_path = er_cache_path

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._binary = os.path.join(repo_root, 'cpp', 'gsparse_sparsifier')
        self._build_script = os.path.join(repo_root, 'cpp', 'build_gsparse_sparsifier.sh')
        self._repo_root = repo_root

    def _ensure_binary(self):
        if not os.path.exists(self._binary):
            subprocess.run(['bash', self._build_script], check=True, cwd=self._repo_root)

    def sparsify(self, data_or_graph):
        self._ensure_binary()

        G = self._ensure_nx(data_or_graph)
        if G.is_directed():
            raise ValueError('ERSparsifier requires an undirected graph')
        nodelist = sorted(G.nodes(), key=lambda x: repr(x))

        if not nodelist:
            H = nx.Graph()
            if isinstance(data_or_graph, Data):
                return self._ensure_pyg(H, original_data=data_or_graph)
            return H

        mapping = {node: idx for idx, node in enumerate(nodelist)}

        def _run(td):
            in_path = os.path.join(td, 'graph.el')
            out_edge_path = os.path.join(td, 'graph_sparse.el')
            out_weight_path = os.path.join(td, 'graph_sparse.wel')
            er_cache = self.er_cache_path if self.er_cache_path is not None \
                       else os.path.join(td, 'graph.er')

            with open(in_path, 'w') as f:
                for u, v in G.edges():
                    if u == v:
                        continue
                    uu, vv = mapping[u], mapping[v]
                    a, b = (uu, vv) if uu <= vv else (vv, uu)
                    f.write(f'{a} {b}\n')

            subprocess.run(
                [self._binary, str(self.epsilon),
                 in_path, out_edge_path, out_weight_path, er_cache],
                check=True,
                cwd=self._repo_root,
            )

            H = nx.Graph()
            H.add_nodes_from(nodelist)
            if os.path.exists(out_edge_path) and os.path.getsize(out_edge_path) > 0:
                edges = np.loadtxt(out_edge_path, dtype=int)
                for row in np.atleast_2d(edges):
                    H.add_edge(nodelist[int(row[0])], nodelist[int(row[1])])
            return H

        with tempfile.TemporaryDirectory() as td:
            H = _run(td)

        if isinstance(data_or_graph, Data):
            return self._ensure_pyg(H, original_data=data_or_graph)
        return H
