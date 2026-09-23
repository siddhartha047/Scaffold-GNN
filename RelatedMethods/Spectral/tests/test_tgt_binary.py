from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from finalize_tgt import finalize_tgt
from tgt_binary import UINT32_MAX, convert_to_tgt_binary


ORIENTED = struct.Struct("<8sIIQQQ64sddiIIIQQQQ")


def _bytes(value: str) -> np.ndarray:
    return np.frombuffer(value.encode(), dtype=np.uint8).copy()


def _input(path: Path, *, weight=(1.0, 1.0)) -> None:
    graph_fingerprint = "a" * 64
    directed_fingerprint = "b" * 64
    np.savez(
        path,
        num_nodes=np.asarray([3], dtype=np.int64),
        src=np.asarray([0, 1], dtype=np.int64),
        dst=np.asarray([1, 2], dtype=np.int64),
        weight=np.asarray(weight, dtype=np.float64),
        graph_fingerprint=_bytes(graph_fingerprint),
        directed_fingerprint=_bytes(directed_fingerprint),
        directed_to_undirected=np.asarray([0, 0, 1, -1], dtype=np.int32),
        num_directed_edges=np.asarray([4], dtype=np.int64),
        dataset=_bytes("toy"),
        schema_version=np.asarray([2], dtype=np.int64),
    )


def test_streaming_binary_and_final_alignment(tmp_path: Path):
    source = tmp_path / "input.npz"
    target = tmp_path / "input.tgtbin"
    _input(source)
    binary = convert_to_tgt_binary(source, target)
    assert not binary.cached
    assert convert_to_tgt_binary(source, target).cached
    assert np.memmap(target, "<u4", "r", offset=binary.src_offset, shape=(2,)).tolist() == [0, 1]
    assert np.memmap(target, "<u4", "r", offset=binary.dst_offset, shape=(2,)).tolist() == [1, 2]
    mapping = np.memmap(target, "<u4", "r", offset=binary.mapping_offset, shape=(4,))
    assert mapping.tolist() == [0, 0, 1, UINT32_MAX]

    oriented = tmp_path / "oriented.bin"
    low_offset = 4096
    high_offset = 8192
    file_bytes = high_offset + 8
    with oriented.open("w+b") as output:
        output.truncate(file_bytes)
        output.write(
            ORIENTED.pack(
                b"SSORI001",
                2,
                3,
                3,
                2,
                3,
                b"a" * 64,
                0.05,
                0.01,
                10,
                8,
                1,
                3,
                42,
                low_offset,
                high_offset,
                file_bytes,
            )
        )
    low = np.memmap(oriented, "<f4", "r+", offset=low_offset, shape=(2,))
    high = np.memmap(oriented, "<f4", "r+", offset=high_offset, shape=(2,))
    low[:] = [0.2, 0.4]
    high[:] = [0.3, 0.2]
    low.flush()
    high.flush()
    del low, high

    artifacts = tmp_path / "artifacts"
    finalized = finalize_tgt(binary, oriented, artifacts, compute_seconds=1.25)
    with np.load(finalized["canonical_artifact"], allow_pickle=False) as values:
        assert values["resistance"].tolist() == pytest.approx([0.5, 0.6])
    with np.load(finalized["artifact"], allow_pickle=False) as values:
        assert values["resistance"].tolist() == pytest.approx([0.5, 0.5, 0.6, 0.0])


def test_tgt_rejects_weighted_graph(tmp_path: Path):
    source = tmp_path / "weighted.npz"
    _input(source, weight=(1.0, 2.0))
    with pytest.raises(ValueError, match="unit-conductance"):
        convert_to_tgt_binary(source, tmp_path / "weighted.tgtbin")
