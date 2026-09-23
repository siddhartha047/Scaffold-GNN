"""Deterministic systematic sampling over a shared cumulative probability array."""

from __future__ import annotations

import numpy as np

from .parallel_utils import auto_worker_count, parallel_threads

try:
    from numba import njit, prange
except Exception:  # pragma: no cover - optional acceleration
    njit = None
    prange = range

# Fixed tick blocks keep both arithmetic and selected edges independent of p.
# Small draws use NumPy to avoid JIT and thread dispatch overhead.
TICKS_PER_BLOCK = 32768


def _scan_ticks(cumulative, offset, count):
    out = np.empty(count, dtype=np.int64)
    blocks = (count + TICKS_PER_BLOCK - 1) // TICKS_PER_BLOCK
    size = cumulative.size
    for block in prange(blocks):
        start = block * TICKS_PER_BLOCK
        end = min(count, start + TICKS_PER_BLOCK)
        tick = offset + float(start)
        lo, hi = 0, size
        while lo < hi:
            mid = (lo + hi) // 2
            if cumulative[mid] < tick:
                lo = mid + 1
            else:
                hi = mid
        pos = min(lo, size - 1)
        for i in range(start, end):
            tick = offset + float(i)
            while pos < size - 1 and cumulative[pos] < tick:
                pos += 1
            out[i] = pos
    return out


_scan_kernel = (
    njit(cache=True, nogil=True, parallel=True)(_scan_ticks)
    if njit is not None else None
)


def systematic_positions(cumulative, offset, count, workers):
    """Return clipped left-search positions and the kernel's worker budget.

    The cumulative sum and RNG draw stay unchanged. Independent tick blocks
    search their first position and then scan forward, releasing the GIL in
    the compiled kernel. No parallel floating-point reduction is introduced.
    Duplicate correction remains the caller's responsibility.
    """
    count = int(count)
    if count <= 0:
        return np.empty(0, dtype=np.int64), 1
    cumulative = np.ascontiguousarray(cumulative, dtype=np.float64)
    if cumulative.size == 0:
        raise ValueError("cannot draw positive ticks from an empty pool")
    if _scan_kernel is None or count < TICKS_PER_BLOCK:
        ticks = float(offset) + np.arange(count, dtype=np.float64)
        return np.minimum(
            np.searchsorted(cumulative, ticks, side="left"), cumulative.size - 1
        ), 1

    blocks = (count + TICKS_PER_BLOCK - 1) // TICKS_PER_BLOCK
    requested = min(auto_worker_count(workers), blocks)
    with parallel_threads(requested) as used:
        out = _scan_kernel(cumulative, float(offset), count)
    return out, used

