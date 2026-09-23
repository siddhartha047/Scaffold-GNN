import torch
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected


def normalize_undirected(edge_index, num_nodes, return_info=False):
    directed_in = int(edge_index.size(1))
    self_loops_removed = int((edge_index[0] == edge_index[1]).sum().item())

    edge_index_no_self, _ = remove_self_loops(edge_index)
    edge_index_undir = to_undirected(edge_index_no_self, num_nodes=num_nodes)
    directed_before_coalesce = int(edge_index_undir.size(1))

    edge_index_coalesced = coalesce(edge_index_undir, num_nodes=num_nodes)
    directed_after_coalesce = int(edge_index_coalesced.size(1))

    info = {
        'directed_in': directed_in,
        'self_loops_removed': self_loops_removed,
        'directed_before_coalesce': directed_before_coalesce,
        'directed_after_coalesce': directed_after_coalesce,
        'coalesce_removed_directed': max(0, directed_before_coalesce - directed_after_coalesce),
    }

    if return_info:
        return edge_index_coalesced, info
    return edge_index_coalesced


def edge_stats(edge_index):
    directed = int(edge_index.size(1))
    if directed == 0:
        return 0, 0, 0

    u = edge_index[0]
    v = edge_index[1]
    self_loops = int((u == v).sum().item())
    a = torch.minimum(u, v)
    b = torch.maximum(u, v)
    undirected_pairs = torch.stack([a, b], dim=0)
    undirected = int(torch.unique(undirected_pairs, dim=1).size(1))
    return directed, undirected, self_loops
