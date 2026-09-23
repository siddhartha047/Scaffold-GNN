import random
import networkx as nx
import networkit as nk
from torch_geometric.data import Data
from .base import BaseSparsifier


class KRandomNeighbor(BaseSparsifier):
    def __init__(self, k=5, seed=None):
        self.k = int(k)
        self.seed = seed
        if self.k < 1:
            raise ValueError('k must be >= 1')

    def sparsify(self, data_or_graph):
        G_nk, node_list = self._nx_to_nk(self._ensure_nx(data_or_graph))

        new_G = nk.graph.Graph(G_nk.numberOfNodes(),
                               weighted=G_nk.isWeighted(),
                               directed=G_nk.isDirected())

        rng = random.Random(self.seed)
        for node in range(G_nk.numberOfNodes()):
            neighbors = list(G_nk.iterNeighbors(node))
            if len(neighbors) > self.k:
                neighbors = rng.sample(neighbors, self.k)
            for neighbor in neighbors:
                if not new_G.hasEdge(node, neighbor):
                    if G_nk.isWeighted():
                        new_G.addEdge(node, neighbor,
                                      w=G_nk.weight(node, neighbor))
                    else:
                        new_G.addEdge(node, neighbor)

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
