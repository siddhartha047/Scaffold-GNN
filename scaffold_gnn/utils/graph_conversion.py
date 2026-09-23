import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx

def pyg_to_nx(data, to_undirected_graph=True):
    if to_undirected_graph:
        return to_networkx(data, to_undirected=True)
    else:
        return to_networkx(data, to_undirected=False)

def nx_to_pyg(G, node_features=None, node_labels=None, num_nodes=None):
    data = from_networkx(G)
    if num_nodes is not None:
        data.num_nodes = num_nodes
    if node_features is not None:
        data.x = node_features
    if node_labels is not None:
        data.y = node_labels
    return data