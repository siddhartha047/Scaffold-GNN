from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from benchmark_adapter import (
    align_effective_resistance,
    attach_directed_effective_resistance,
    directed_edge_fingerprint,
    effective_resistance_artifact,
    load_directed_effective_resistance,
)
from export_benchmark_graph import _bytes, _canonical_undirected_edges, _directed_to_undirected


def test_reciprocal_pyg_edges_are_one_conductance() -> None:
    edge_index = np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]])
    src, dst, weight = _canonical_undirected_edges(edge_index, 3)
    np.testing.assert_array_equal(src, [0, 1])
    np.testing.assert_array_equal(dst, [1, 2])
    np.testing.assert_allclose(weight, [1.0, 1.0])


def test_self_loops_are_removed() -> None:
    edge_index = np.asarray([[0, 0, 1], [0, 1, 0]])
    src, dst, weight = _canonical_undirected_edges(edge_index, 2)
    np.testing.assert_array_equal(src, [0])
    np.testing.assert_array_equal(dst, [1])
    np.testing.assert_allclose(weight, [1.0])


def test_directed_mapping_preserves_duplicates_and_loops() -> None:
    edge_index = np.asarray([[0, 1, 0, 1, 2], [1, 0, 1, 2, 2]])
    src, dst, _ = _canonical_undirected_edges(edge_index, 3)
    mapping = _directed_to_undirected(edge_index, 3, src, dst)
    np.testing.assert_array_equal(mapping, [0, 0, 0, 1, -1])


def test_align_resistance_to_directed_edges(tmp_path: Path) -> None:
    artifact = tmp_path / "er.npz"
    np.savez(
        artifact,
        src=np.asarray([0, 1]),
        dst=np.asarray([1, 2]),
        conductance=np.ones(2),
        resistance=np.asarray([0.25, 0.75]),
        leverage_score=np.asarray([0.25, 0.75]),
    )
    directed = np.asarray([[0, 1, 2, 2], [1, 0, 1, 2]])
    np.testing.assert_allclose(
        align_effective_resistance(directed, artifact).numpy(),
        [0.25, 0.25, 0.75, 0.0],
    )

    with pytest.raises(ValueError, match="absent"):
        align_effective_resistance(np.asarray([[0], [2]]), artifact)


def test_directed_loader_rejects_same_length_wrong_edge_order(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    artifact = tmp_path / "effective_resistance_directed.npz"
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    np.savez(
        artifact,
        resistance=np.asarray([0.25, 0.25, 0.75, 0.75], dtype=np.float32),
        directed_fingerprint=_bytes(directed_edge_fingerprint(edge_index, 3)),
    )

    loaded = load_directed_effective_resistance(
        artifact, edge_index=edge_index, num_nodes=3
    )
    np.testing.assert_allclose(loaded, [0.25, 0.25, 0.75, 0.75])

    reordered = edge_index[:, [2, 1, 0, 3]]
    with pytest.raises(ValueError, match="does not match"):
        load_directed_effective_resistance(
            artifact, edge_index=reordered, num_nodes=3
        )


def test_attach_directed_resistance_checks_and_attaches(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    torch_geometric = pytest.importorskip("torch_geometric")
    artifact = tmp_path / "effective_resistance_directed.npz"
    edge_index = torch.tensor([[0, 1], [1, 0]])
    np.savez(
        artifact,
        resistance=np.asarray([0.5, 0.5], dtype=np.float32),
        directed_fingerprint=_bytes(directed_edge_fingerprint(edge_index, 2)),
    )
    data = torch_geometric.data.Data(edge_index=edge_index, num_nodes=2)
    output = attach_directed_effective_resistance(data, artifact)
    assert output is data
    assert torch.equal(data.edge_effective_resistance, torch.tensor([0.5, 0.5]))


def test_artifact_path_resolves_aliases(tmp_path: Path) -> None:
    assert effective_resistance_artifact(tmp_path, "ogbn_products") == (
        tmp_path
        / "ogb/ogbn_products/spectral_sparsification/"
        "effective_resistance_directed.npz"
    )
    assert effective_resistance_artifact(tmp_path, "Cora", directed=False) == (
        tmp_path
        / "Planetoid/Cora/spectral_sparsification/effective_resistance.npz"
    )
