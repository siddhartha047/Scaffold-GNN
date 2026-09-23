from torch_geometric.data import Data
from .base import BaseSparsifier

class FullGraph(BaseSparsifier):

    def __init__(self):
        pass

    def sparsify(self, data_or_graph):
        if isinstance(data_or_graph, Data):
            return data_or_graph.clone()
        else:
            import networkx as nx
            return data_or_graph.copy()