import pdb
import time
import random
random.seed(42)
import numpy as np
import torch
from torch_geometric.utils import degree, to_undirected


_MULTINOMIAL_CATEGORY_LIMIT = 1 << 24
_LARGE_SAMPLE_CHUNK_SIZE = 1 << 24


def _chunked_weighted_edge_sample(
    probabilities,
    budget,
    generator,
    *,
    chunk_size=_LARGE_SAMPLE_CHUNK_SIZE,
):
    """Weighted sampling without replacement without a category-count limit.

    Exponential-race keys produce the same Plackett-Luce weighted
    without-replacement distribution as sequential multinomial sampling.
    Keeping only the best ``budget`` keys after each chunk bounds temporary
    GPU memory independently of the total number of edges.
    """

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    selected_scores = torch.empty(
        0,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    selected_indices = torch.empty(
        0,
        dtype=torch.long,
        device=probabilities.device,
    )
    edge_count = probabilities.numel()

    for start in range(0, edge_count, chunk_size):
        end = min(start + chunk_size, edge_count)
        weights = probabilities[start:end]
        scores = torch.empty_like(weights)
        scores.exponential_(1.0, generator=generator)
        scores.div_(weights)

        local_count = min(budget, scores.numel())
        if local_count < scores.numel():
            local_scores, local_offsets = torch.topk(
                scores,
                local_count,
                largest=False,
                sorted=False,
            )
            local_indices = local_offsets.add(start)
        else:
            local_scores = scores
            local_indices = torch.arange(
                start,
                end,
                dtype=torch.long,
                device=probabilities.device,
            )

        if selected_scores.numel() == 0:
            selected_scores = local_scores
            selected_indices = local_indices
            continue

        candidate_scores = torch.cat((selected_scores, local_scores))
        candidate_indices = torch.cat((selected_indices, local_indices))
        keep_count = min(budget, candidate_scores.numel())
        if keep_count < candidate_scores.numel():
            selected_scores, keep_offsets = torch.topk(
                candidate_scores,
                keep_count,
                largest=False,
                sorted=False,
            )
            selected_indices = candidate_indices[keep_offsets]
        else:
            selected_scores = candidate_scores
            selected_indices = candidate_indices

    return selected_indices


def _exact_weighted_edge_sample(probabilities, budget, seed):
    """Sample an exact number of unique edge indices without replacement."""

    if probabilities.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if not probabilities.is_floating_point():
        raise TypeError("probabilities must use a floating-point dtype")

    edge_count = probabilities.numel()
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if budget == 0:
        indices = torch.empty(
            0,
            dtype=torch.long,
            device=probabilities.device,
        )
    elif budget >= edge_count:
        indices = torch.arange(edge_count, device=probabilities.device)
    else:
        if not bool(torch.isfinite(probabilities).all()):
            raise ValueError("probabilities must be finite")
        if bool((probabilities < 0).any()):
            raise ValueError("probabilities must be non-negative")
        positive_count = int(torch.count_nonzero(probabilities).item())
        if positive_count < budget:
            raise ValueError(
                "budget exceeds the number of edges with positive probability"
            )

        generator_device = (
            probabilities.device if probabilities.is_cuda else "cpu"
        )
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(seed)
        if edge_count > _MULTINOMIAL_CATEGORY_LIMIT:
            print(
                "weighted edge count exceeds safe multinomial limit; "
                "using chunked exponential-race sampling"
            )
            indices = _chunked_weighted_edge_sample(
                probabilities,
                budget,
                generator,
            )
        else:
            indices = torch.multinomial(
                probabilities,
                budget,
                replacement=False,
                generator=generator,
            )
    counts = torch.ones(
        indices.numel(),
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    return indices, counts


def _exact_weighted_undirected_edge_sample(
    edge_index,
    probabilities,
    budget,
    seed,
    num_nodes,
):
    """Sample an exact directed-edge budget while preserving edge pairs.

    Medium tunedGNN datasets are represented with both directions of every
    non-loop edge.  Sampling those directions independently creates a directed
    graph and changes the GCN operator.  We instead sample canonical
    undirected pairs, expand both directions, and use at most one singleton
    direction when an odd requested budget cannot be perfectly symmetric.
    """

    if budget == 0:
        empty_edges = edge_index[:, :0]
        empty_values = probabilities.new_empty(0, dtype=torch.float)
        return empty_edges, empty_values

    src, dst = edge_index
    if bool((src == dst).any()):
        raise ValueError(
            "undirected DSpar sampling expects self-loops to be added by the GNN"
        )
    low = torch.minimum(src, dst)
    high = torch.maximum(src, dst)
    pair_keys = low * int(num_nodes) + high
    unique_keys, inverse = torch.unique(
        pair_keys,
        sorted=True,
        return_inverse=True,
    )
    pair_probabilities = torch.zeros(
        unique_keys.numel(),
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    pair_probabilities.scatter_add_(0, inverse, probabilities)
    pair_probabilities /= pair_probabilities.sum()

    full_pair_count, singleton_count = divmod(int(budget), 2)
    sampled_pair_count = full_pair_count + singleton_count
    sampled_pairs, _counts = _exact_weighted_edge_sample(
        pair_probabilities,
        sampled_pair_count,
        seed,
    )
    sampled_keys = unique_keys[sampled_pairs]
    sampled_low = torch.div(
        sampled_keys,
        int(num_nodes),
        rounding_mode='floor',
    )
    sampled_high = sampled_keys.remainder(int(num_nodes))
    # This is the same inverse-probability weight as directed sampling when
    # each undirected pair contains two equiprobable directions.
    if sampled_pair_count >= pair_probabilities.numel():
        sampled_weights = torch.ones_like(
            pair_probabilities[sampled_pairs],
            dtype=torch.float,
        )
    else:
        approximate_inclusion = (
            sampled_pair_count * pair_probabilities[sampled_pairs]
        ).clamp(max=1.0)
        sampled_weights = approximate_inclusion.reciprocal().float()

    paired_low = sampled_low[:full_pair_count]
    paired_high = sampled_high[:full_pair_count]
    paired_weights = sampled_weights[:full_pair_count]
    new_src = torch.cat((paired_low, paired_high))
    new_dst = torch.cat((paired_high, paired_low))
    edge_attr = torch.cat((paired_weights, paired_weights))
    if singleton_count:
        new_src = torch.cat((new_src, sampled_low[-1:]))
        new_dst = torch.cat((new_dst, sampled_high[-1:]))
        edge_attr = torch.cat((edge_attr, sampled_weights[-1:]))

    return torch.stack((new_src, new_dst), dim=0), edge_attr


def maybe_sparsfication(data, dataset, follow_by_subgraph_sampling, random=False, is_undirected=True, reweighted=True, target_ratio=None):
    N, E = data.num_nodes, data.num_edges
    if target_ratio is not None and not 0.0 < float(target_ratio) <= 1.0:
        raise ValueError("target_ratio must be in (0, 1]")
    src, dst = data.edge_index
    epsilon = 0.25
    if dataset == 'ogbn-arxiv':
        epsilon = 0.25 if not random else 0.35
    elif dataset == 'reddit2':
        epsilon = 0.3 if not random else 0.32
    elif dataset == 'ogbn-products':
        epsilon = 0.4 if not random else 0.45
    elif dataset == 'yelp':
        epsilon = 0.5 if not random else 0.6
    elif dataset == 'ogbn-proteins':
        epsilon = 0.25

    if follow_by_subgraph_sampling and dataset == 'ogbn-products':
        epsilon = 0.15 if not random else 0.2

    print(f'epsilon: {epsilon}')
    if target_ratio is None:
        Q = int(0.16 * N * np.log(N) / epsilon ** 2)
    else:
        Q = max(1, int(float(target_ratio) * E))
    print(f"Q: {Q}")
    print(f'E/Q ratio: {E/Q}')
    print(f'E/nlogn ratio: {E/N/np.log(N)}')
    print('sparsify the input graph')
    data = data.clone()
    s = time.time()
    if random:
        pe = torch.ones(size=(E,), dtype=torch.double) / E
    else:
        print('sparsify the graph by degrees')
        node_degree = degree(dst, data.num_nodes)
        di, dj = torch.nan_to_num(1. / node_degree[src]), torch.nan_to_num(1. / node_degree[dst])
        pe = (di + dj).double()
        pe = pe / torch.sum(pe)
    print(f'cal edge distribution used {time.time() - s} sec')
    # For reproducibility, we manually set the seed of graph sparsification to 42. We note that this seed is only effective for the graph sparsification, 
    # it does not impact any following process.
    seed_val = 42
    s = time.time()
    if target_ratio is None:
        # The optional extension is only needed by the original with-replacement
        # sampler. Explicit retained-edge budgets use the PyTorch path below.
        import dspar.cpp_extension.sampler as sampler
        p_cumsum = torch.cumsum(pe, 0)
        sampled = sampler.edge_sample(p_cumsum, Q, seed_val)
        e_indices, e_cnt = torch.unique(sampled, return_counts=True)
    elif is_undirected:
        edge_index, edge_attr = _exact_weighted_undirected_edge_sample(
            data.edge_index,
            pe,
            Q,
            seed_val,
            data.num_nodes,
        )
        e_indices = e_cnt = None
    else:
        # The benchmark API defines target_ratio as a retained-edge budget.
        # Sampling with replacement can undershoot it after duplicate removal,
        # so explicit targets use weighted sampling without replacement.
        e_indices, e_cnt = _exact_weighted_edge_sample(pe, Q, seed_val)
    print(f'sample edge used {time.time() - s} sec')
    if e_indices is not None:
        if Q >= E:
            new_graph = torch.ones_like(pe[e_indices])
        else:
            # Explicit budgets sample without replacement.  The old
            # with-replacement count/(Q*p) formula can assign weights below
            # one and does not reduce to the original graph at ratio=1.
            approximate_inclusion = (Q * pe[e_indices]).clamp(max=1.0)
            new_graph = approximate_inclusion.reciprocal()
        new_src, new_dst = src[e_indices], dst[e_indices]
        edge_index = torch.cat([new_src.view(1, -1), new_dst.view(1, -1)], dim=0)
        edge_attr = new_graph.float()
    if is_undirected and target_ratio is None:
        data.edge_index, data.edge_attr = to_undirected(edge_index, edge_attr)
    else:
        # Do not expand explicitly budgeted samples with reverse edges: that
        # would make the graph used for training exceed the requested ratio.
        data.edge_index, data.edge_attr = edge_index, edge_attr
    if not reweighted:
        print('not reweight')
        data.edge_attr = None
    actual_edges = int(data.num_edges)
    actual_ratio = actual_edges / E
    data.dspar_original_num_edges = int(E)
    data.dspar_target_num_edges = int(Q)
    data.dspar_actual_num_edges = actual_edges
    data.dspar_actual_kept_ratio = actual_ratio
    print(
        f'before sparsification, num_edges: {E}, '
        f'after sparsification, num_edges: {actual_edges}, '
        f'ratio: {actual_ratio}'
    )
    return data
