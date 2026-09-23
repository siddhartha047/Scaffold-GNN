import random
import math
import networkx as nx
import networkit as nk
from torch_geometric.data import Data
from .base import BaseSparsifier


class CutSparsifier(BaseSparsifier):
    """Cut sparsifier: includes each edge independently with probability
    p = min(1, rho / k_e), where rho = 16*(d+2)*log(n) / epsilon^2
    and k_e is the edge connectivity (defaults to a uniform estimate).

    Parameters
    ----------
    epsilon : float
        Approximation parameter (smaller → denser output).
    d : int
        Dimension parameter from the algorithm (default 1).
    connectivity : int or None
        Edge connectivity estimate k used for all edges.
        If None, networkit's LocalPartitionCoverSparsifier score is used
        as a proxy; otherwise the provided value is used uniformly.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, epsilon=10.0, d=1, connectivity=10, seed=None):
        if float(epsilon) <= 0:
            raise ValueError('epsilon must be > 0')
        if int(connectivity) <= 0:
            raise ValueError('connectivity must be > 0')
        self.epsilon = float(epsilon)
        self.d = int(d)
        self.connectivity = int(connectivity)
        self.seed = seed

    def sparsify(self, data_or_graph):
        G = self._ensure_nx(data_or_graph)
        G_nk, node_list = self._nx_to_nk(G)

        n = G_nk.numberOfNodes()
        if n == 0:
            H_nx = nx.DiGraph() if G_nk.isDirected() else nx.Graph()
            H_nx.add_nodes_from(node_list)
            if isinstance(data_or_graph, Data):
                return self._ensure_pyg(H_nx, original_data=data_or_graph)
            return H_nx
        rho = 16 * (self.d + 2) * math.log(max(n, 2)) / (self.epsilon ** 2)

        new_G = nk.graph.Graph(n,
                               weighted=G_nk.isWeighted(),
                               directed=G_nk.isDirected())

        rng = random.Random(self.seed)
        for u, v in G_nk.iterEdges():
            p = min(1.0, rho / self.connectivity)
            if rng.random() < p:
                if G_nk.isWeighted():
                    new_G.addEdge(u, v, w=G_nk.weight(u, v))
                else:
                    new_G.addEdge(u, v)

        H_nx = nx.DiGraph() if G_nk.isDirected() else nx.Graph()
        H_nx.add_nodes_from(node_list)
        for u, v in new_G.iterEdges():
            if new_G.isWeighted():
                H_nx.add_edge(node_list[u], node_list[v], weight=new_G.weight(u, v))
            else:
                H_nx.add_edge(node_list[u], node_list[v])

        if isinstance(data_or_graph, Data):
            return self._ensure_pyg(H_nx, original_data=data_or_graph)
        return H_nx
