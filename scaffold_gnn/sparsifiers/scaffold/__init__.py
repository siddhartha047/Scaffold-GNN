"""SCAFFOLD sparsifiers.

This package hosts the five variants of the proposed SCAFFOLD method:

* :class:`ScaffoldGreedySparsifier` -- exact per-round scoring, for very small
  graphs such as Karate.
* :class:`ScaffoldHeapSparsifier` -- cluster-local lazy heap, suitable for
  medium-sized graphs such as Cora.
* :class:`ScaffoldBatchSparsifier` -- sampled per-cluster top-r growth with
  batch-scoped dilation/congestion scoring.
* :class:`ScaffoldFastSparsifier` -- one LCA tree-prefix pass and global top-k,
  with an optional tensor backend for large-scale graphs.
* :class:`ScaffoldSampleSparsifier` -- one-time precomputed edge weights plus a
  per-epoch systematic pi-ps draw over a guaranteed spanning-forest backbone.

The wrapper :class:`ParallelClusteredJointDilationCongestionSparsifier` is
preserved for backward compatibility with older sparsifier names.
"""

from importlib import import_module

__all__ = [
    "ParallelClusteredJointDilationCongestionSparsifier",
    "ScaffoldGreedySparsifier",
    "ScaffoldBatchSparsifier",
    "ScaffoldFastSparsifier",
    "ScaffoldHeapSparsifier",
    "ScaffoldSampleSparsifier",
]

_EXPORTS = {
    "ParallelClusteredJointDilationCongestionSparsifier": (
        "scaffold_gnn.sparsifiers.scaffold.sparsifier",
        "ParallelClusteredJointDilationCongestionSparsifier",
    ),
    "ScaffoldGreedySparsifier": (
        "scaffold_gnn.sparsifiers.scaffold.scaffold_greedy",
        "ScaffoldGreedySparsifier",
    ),
    "ScaffoldFastSparsifier": (
        "scaffold_gnn.sparsifiers.scaffold.scaffold_fast",
        "ScaffoldFastSparsifier",
    ),
    "ScaffoldBatchSparsifier": (
        "scaffold_gnn.sparsifiers.scaffold.scaffold_batch",
        "ScaffoldBatchSparsifier",
    ),
    "ScaffoldHeapSparsifier": (
        "scaffold_gnn.sparsifiers.scaffold.scaffold_heap",
        "ScaffoldHeapSparsifier",
    ),
    "ScaffoldSampleSparsifier": (
        "scaffold_gnn.sparsifiers.scaffold.scaffold_sample",
        "ScaffoldSampleSparsifier",
    ),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
