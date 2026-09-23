import pytest
import torch
from torch_geometric.data import Data

import dspar.sparsification as sparsification
from dspar.sparsification import (
    _chunked_weighted_edge_sample,
    _exact_weighted_edge_sample,
    _exact_weighted_undirected_edge_sample,
    maybe_sparsfication,
)


def _chunked_sample(probabilities, budget, seed, chunk_size):
    generator = torch.Generator(device=probabilities.device)
    generator.manual_seed(seed)
    return _chunked_weighted_edge_sample(
        probabilities,
        budget,
        generator,
        chunk_size=chunk_size,
    )


def test_chunked_sampler_returns_exact_unique_budget_and_is_reproducible():
    probabilities = torch.linspace(1.0, 4.0, 100_003, dtype=torch.double)

    first = _chunked_sample(probabilities, 5_000, seed=42, chunk_size=8_192)
    second = _chunked_sample(probabilities, 5_000, seed=42, chunk_size=8_192)

    assert first.numel() == 5_000
    assert torch.unique(first).numel() == 5_000
    assert torch.equal(first, second)
    assert int(first.min()) >= 0
    assert int(first.max()) < probabilities.numel()


def test_chunked_sampler_never_selects_zero_weight_edges():
    probabilities = torch.tensor(
        [0.0, 0.0, 1.0, 2.0, 3.0, 0.0],
        dtype=torch.double,
    )

    indices = _chunked_sample(probabilities, 3, seed=7, chunk_size=2)

    assert set(indices.tolist()) == {2, 3, 4}


def test_exact_sampler_rejects_budget_larger_than_positive_support():
    probabilities = torch.tensor([0.0, 1.0, 0.0], dtype=torch.double)

    with pytest.raises(ValueError, match="positive probability"):
        _exact_weighted_edge_sample(probabilities, 2, seed=42)


def test_exact_sampler_preserves_small_input_multinomial_behavior():
    probabilities = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.double)

    indices, counts = _exact_weighted_edge_sample(
        probabilities,
        budget=3,
        seed=42,
    )

    assert indices.numel() == 3
    assert torch.unique(indices).numel() == 3
    assert torch.equal(counts, torch.ones(3, dtype=torch.double))


def test_exact_sampler_dispatches_large_inputs_to_chunked_path(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(sparsification, "_MULTINOMIAL_CATEGORY_LIMIT", 3)
    probabilities = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.double)

    indices, _counts = _exact_weighted_edge_sample(
        probabilities,
        budget=2,
        seed=42,
    )

    assert indices.numel() == 2
    assert torch.unique(indices).numel() == 2
    assert "chunked exponential-race sampling" in capsys.readouterr().out


@pytest.mark.parametrize("budget", (2, 3, 6))
def test_undirected_sampler_preserves_pairs_and_exact_budget(budget):
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 0],
            [1, 0, 2, 1, 3, 2, 0, 3],
        ]
    )
    probabilities = torch.ones(8, dtype=torch.double) / 8

    sampled_edges, sampled_weights = _exact_weighted_undirected_edge_sample(
        edge_index,
        probabilities,
        budget=budget,
        seed=42,
        num_nodes=4,
    )

    assert sampled_edges.size(1) == budget
    assert sampled_weights.numel() == budget
    pairs = list(zip(*sampled_edges.tolist()))
    paired_prefix = 2 * (budget // 2)
    for src, dst in pairs[:paired_prefix]:
        assert (dst, src) in pairs[:paired_prefix]


def test_ratio_one_preserves_the_unweighted_graph():
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 0],
            [1, 0, 2, 1, 3, 2, 0, 3],
        ]
    )
    data = Data(
        x=torch.randn(4, 3),
        edge_index=edge_index,
        num_nodes=4,
    )

    sparse = maybe_sparsfication(
        data,
        dataset='cora',
        follow_by_subgraph_sampling=False,
        random=False,
        is_undirected=True,
        target_ratio=1.0,
    )

    assert sparse.num_edges == data.num_edges
    assert torch.equal(
        torch.sort(
            sparse.edge_index[0] * data.num_nodes + sparse.edge_index[1]
        ).values,
        torch.sort(
            data.edge_index[0] * data.num_nodes + data.edge_index[1]
        ).values,
    )
    assert torch.equal(sparse.edge_attr, torch.ones(data.num_edges))
