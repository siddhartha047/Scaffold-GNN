from .scaffold_fast import ScaffoldFastSparsifier
from .scaffold_heap import ScaffoldHeapSparsifier


class ParallelClusteredJointDilationCongestionSparsifier:
    """Backward-compatible wrapper for the old parallel clustered sparsifier name."""

    def __init__(self, *args, cluster_strategy="sample", **kwargs):
        strategy = str(cluster_strategy).lower()
        if strategy == "heap":
            self._delegate = ScaffoldHeapSparsifier(*args, **kwargs)
        elif strategy == "sample":
            self._delegate = ScaffoldFastSparsifier(*args, **kwargs)
        else:
            raise ValueError(f"Unknown cluster_strategy: {cluster_strategy!r}")

    def sparsify(self, data_or_graph):
        return self._delegate.sparsify(data_or_graph)

    def __getattr__(self, name):
        return getattr(self._delegate, name)
