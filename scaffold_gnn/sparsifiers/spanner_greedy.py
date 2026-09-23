import copy
import networkx as nx
import networkit as nk
from torch_geometric.data import Data
from .base import BaseSparsifier


class GreedySpanner(BaseSparsifier):
    def __init__(self, stretch=3, seed=None):
        self.stretch = int(stretch)
        self.seed = seed

    def sparsify(self, data_or_graph):
        G = self._ensure_nx(data_or_graph)

        G_nk, node_list = self._nx_to_nk(G)
        new_G = nk.graph.Graph(G_nk.numberOfNodes(),
                               weighted=G_nk.isWeighted(),
                               directed=G_nk.isDirected())

        if self.seed is not None:
            nk.setSeed(self.seed, False)
        G_copy = copy.deepcopy(G_nk)
        while G_copy.numberOfEdges():
            edge = nk.graphtools.randomEdge(G_copy)
            G_copy.removeEdge(*edge)
            dist = (nk.distance.BidirectionalDijkstra(new_G, edge[0], edge[1])
                    .run().getDistance())
            # For weighted graphs compare against stretch * edge_weight; for
            # unweighted, edge weight is 1 so the condition reduces to dist > stretch.
            threshold = self.stretch * (G_nk.weight(*edge) if G_nk.isWeighted() else 1)
            if dist > threshold:
                if G_nk.isWeighted():
                    new_G.addEdge(edge[0], edge[1], w=G_nk.weight(*edge))
                else:
                    new_G.addEdge(*edge)

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
