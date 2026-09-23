"""Streaming interchange format for the memory-bounded TGT backend."""

from __future__ import annotations

import io
import os
import struct
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np


MAGIC = b"SSTGT001"
VERSION = 1
HEADER_BYTES = 4096
UINT32_MAX = np.iinfo(np.uint32).max
_HEADER = struct.Struct("<8sIIQQQ64s64s64sQQQQ")


@dataclass(frozen=True)
class InputGraphInfo:
    path: Path
    dataset: str
    num_nodes: int
    num_undirected_edges: int
    num_directed_edges: int
    graph_fingerprint: str
    directed_fingerprint: str


@dataclass(frozen=True)
class TGTBinaryInfo:
    path: Path
    graph: InputGraphInfo
    src_offset: int
    dst_offset: int
    mapping_offset: int
    file_bytes: int
    cached: bool


def _member(name: str) -> str:
    return name if name.endswith(".npy") else f"{name}.npy"


def _array_header(stream: BinaryIO) -> tuple[tuple[int, ...], bool, np.dtype]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    elif version == (3, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"unsupported NPY format version {version}")
    return tuple(int(value) for value in shape), bool(fortran), np.dtype(dtype)


def _read_small(zip_file: zipfile.ZipFile, name: str) -> np.ndarray:
    with zip_file.open(_member(name), "r") as source:
        payload = source.read()
    return np.load(io.BytesIO(payload), allow_pickle=False)


def _shape(zip_file: zipfile.ZipFile, name: str) -> tuple[int, ...]:
    with zip_file.open(_member(name), "r") as source:
        shape, _, _ = _array_header(source)
    return shape


def _decode(array: np.ndarray) -> str:
    return bytes(np.asarray(array, dtype=np.uint8).tolist()).decode("utf-8")


def inspect_input_graph(path: Path) -> InputGraphInfo:
    """Read only scalar/header data from an Benchmark graph NPZ."""

    resolved = path.expanduser().resolve()
    with zipfile.ZipFile(resolved, "r") as values:
        names = set(values.namelist())
        required = {
            _member("num_nodes"),
            _member("src"),
            _member("dst"),
            _member("weight"),
            _member("num_directed_edges"),
            _member("directed_to_undirected"),
            _member("graph_fingerprint"),
            _member("directed_fingerprint"),
            _member("dataset"),
        }
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"{resolved} is missing NPZ members: {', '.join(missing)}")
        num_nodes = int(np.asarray(_read_small(values, "num_nodes")).reshape(-1)[0])
        num_directed = int(
            np.asarray(_read_small(values, "num_directed_edges")).reshape(-1)[0]
        )
        src_shape = _shape(values, "src")
        dst_shape = _shape(values, "dst")
        weight_shape = _shape(values, "weight")
        mapping_shape = _shape(values, "directed_to_undirected")
        if len(src_shape) != 1 or dst_shape != src_shape or weight_shape != src_shape:
            raise ValueError(f"{resolved} has inconsistent canonical edge-array shapes")
        if mapping_shape != (num_directed,):
            raise ValueError(
                f"{resolved} mapping shape {mapping_shape} does not match "
                f"num_directed_edges={num_directed}"
            )
        graph_fingerprint = _decode(_read_small(values, "graph_fingerprint"))
        directed_fingerprint = _decode(_read_small(values, "directed_fingerprint"))
        dataset = _decode(_read_small(values, "dataset"))
    if num_nodes <= 0 or src_shape[0] <= 0 or num_directed < 0:
        raise ValueError(f"{resolved} contains invalid graph dimensions")
    if len(graph_fingerprint) != 64 or len(directed_fingerprint) != 64:
        raise ValueError(f"{resolved} has malformed SHA-256 fingerprints")
    return InputGraphInfo(
        path=resolved,
        dataset=dataset,
        num_nodes=num_nodes,
        num_undirected_edges=src_shape[0],
        num_directed_edges=num_directed,
        graph_fingerprint=graph_fingerprint,
        directed_fingerprint=directed_fingerprint,
    )


def _chunks(
    zip_file: zipfile.ZipFile,
    name: str,
    *,
    chunk_elements: int = 1_048_576,
) -> Iterator[np.ndarray]:
    with zip_file.open(_member(name), "r") as source:
        shape, fortran, dtype = _array_header(source)
        # C/Fortran order is equivalent for a one-dimensional array.  Julia's
        # NPZ writer commonly marks its vectors as Fortran contiguous.
        if len(shape) != 1:
            raise ValueError(f"{name} must be a one-dimensional array")
        remaining = shape[0]
        while remaining:
            count = min(remaining, chunk_elements)
            expected = count * dtype.itemsize
            raw = source.read(expected)
            if len(raw) != expected:
                raise EOFError(
                    f"short read in {name}: expected {expected} bytes, got {len(raw)}"
                )
            yield np.frombuffer(raw, dtype=dtype, count=count)
            remaining -= count


def _align(value: int, alignment: int = 4096) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _padded(value: str, size: int = 64) -> bytes:
    encoded = value.encode("utf-8")
    if len(encoded) > size:
        raise ValueError(f"value is longer than {size} bytes: {value!r}")
    return encoded.ljust(size, b"\0")


def _read_binary_header(path: Path) -> tuple[object, ...]:
    with path.open("rb") as source:
        payload = source.read(_HEADER.size)
    if len(payload) != _HEADER.size:
        raise ValueError("truncated TGT binary header")
    return _HEADER.unpack(payload)


def _cached_binary(path: Path, graph: InputGraphInfo) -> TGTBinaryInfo | None:
    if not path.is_file():
        return None
    try:
        (
            magic,
            version,
            flags,
            num_nodes,
            num_edges,
            num_directed,
            graph_fp,
            directed_fp,
            _dataset,
            src_offset,
            dst_offset,
            mapping_offset,
            file_bytes,
        ) = _read_binary_header(path)
        matches = (
            magic == MAGIC
            and version == VERSION
            and flags == 3
            and num_nodes == graph.num_nodes
            and num_edges == graph.num_undirected_edges
            and num_directed == graph.num_directed_edges
            and graph_fp.rstrip(b"\0").decode("ascii") == graph.graph_fingerprint
            and directed_fp.rstrip(b"\0").decode("ascii") == graph.directed_fingerprint
            and file_bytes == path.stat().st_size
        )
        if not matches:
            return None
        return TGTBinaryInfo(
            path=path,
            graph=graph,
            src_offset=src_offset,
            dst_offset=dst_offset,
            mapping_offset=mapping_offset,
            file_bytes=file_bytes,
            cached=True,
        )
    except (OSError, UnicodeDecodeError, ValueError, struct.error):
        return None


def convert_to_tgt_binary(
    input_npz: Path,
    output_path: Path | None = None,
    *,
    force: bool = False,
) -> TGTBinaryInfo:
    """Convert without loading a full endpoint or mapping array into RAM."""

    graph = inspect_input_graph(input_npz)
    path = (
        output_path.expanduser().resolve()
        if output_path is not None
        else graph.path.with_suffix(".tgtbin")
    )
    if not force:
        cached = _cached_binary(path, graph)
        if cached is not None:
            return cached

    if graph.num_nodes > UINT32_MAX:
        raise ValueError("TGT binary supports at most 2^32-1 nodes")
    if graph.num_undirected_edges > UINT32_MAX:
        raise ValueError("TGT binary supports at most 2^32-1 canonical edges")

    src_offset = HEADER_BYTES
    dst_offset = _align(src_offset + 4 * graph.num_undirected_edges)
    mapping_offset = _align(dst_offset + 4 * graph.num_undirected_edges)
    file_bytes = mapping_offset + 4 * graph.num_directed_edges
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b", buffering=8 * 1024 * 1024) as output:
            output.truncate(file_bytes)
            with zipfile.ZipFile(graph.path, "r") as values:
                for name, offset in (("src", src_offset), ("dst", dst_offset)):
                    output.seek(offset)
                    seen = 0
                    for chunk in _chunks(values, name):
                        integer = np.asarray(chunk, dtype=np.int64)
                        if (integer < 0).any() or (integer >= graph.num_nodes).any():
                            raise ValueError(f"{name} contains an invalid node id")
                        output.write(integer.astype("<u4", copy=False).tobytes())
                        seen += integer.size
                    if seen != graph.num_undirected_edges:
                        raise ValueError(f"{name} length changed during conversion")

                # TGT+ computes unweighted spanning centrality.  Refuse an
                # accidental weighted graph instead of changing its meaning.
                weight_count = 0
                for chunk in _chunks(values, "weight"):
                    weight = np.asarray(chunk, dtype=np.float64)
                    if not np.isfinite(weight).all() or not np.equal(weight, 1.0).all():
                        raise ValueError(
                            "TGT+ supports only unit-conductance graphs; use the "
                            "Laplacians.jl backend for weighted input"
                        )
                    weight_count += weight.size
                if weight_count != graph.num_undirected_edges:
                    raise ValueError("weight length changed during conversion")

                output.seek(mapping_offset)
                seen = 0
                for chunk in _chunks(values, "directed_to_undirected"):
                    mapping = np.asarray(chunk, dtype=np.int64)
                    if (mapping < -1).any() or (
                        mapping[mapping >= 0] >= graph.num_undirected_edges
                    ).any():
                        raise ValueError("directed edge mapping contains an invalid index")
                    encoded = mapping.astype("<u4", copy=True)
                    encoded[mapping < 0] = UINT32_MAX
                    output.write(encoded.tobytes())
                    seen += mapping.size
                if seen != graph.num_directed_edges:
                    raise ValueError("directed mapping length changed during conversion")

            header = _HEADER.pack(
                MAGIC,
                VERSION,
                3,  # mapping present + unit conductance
                graph.num_nodes,
                graph.num_undirected_edges,
                graph.num_directed_edges,
                _padded(graph.graph_fingerprint),
                _padded(graph.directed_fingerprint),
                _padded(graph.dataset),
                src_offset,
                dst_offset,
                mapping_offset,
                file_bytes,
            )
            output.seek(0)
            output.write(header)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

    return TGTBinaryInfo(
        path=path,
        graph=graph,
        src_offset=src_offset,
        dst_offset=dst_offset,
        mapping_offset=mapping_offset,
        file_bytes=file_bytes,
        cached=False,
    )


__all__ = [
    "InputGraphInfo",
    "TGTBinaryInfo",
    "convert_to_tgt_binary",
    "inspect_input_graph",
]
