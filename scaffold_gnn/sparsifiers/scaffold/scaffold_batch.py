"""Original sampled-batch SCAFFOLD growth under its canonical Batch name."""

from .scaffold_fast import ScaffoldFastSparsifier


class ScaffoldBatchSparsifier(ScaffoldFastSparsifier):
    """Sample candidate batches, score them, then add each batch's top edges.

    The NetworkX backend recomputes paths against the current support each
    round.  The tensor backend keeps the same sampled loop and ranks candidates
    with the exact LCA dilation/congestion objective (``tree_exact_loop``).
    """

    algorithm_name = "scaffold_batch"
    display_name = "SCAFFOLD-Batch"

    def __init__(
        self,
        *args,
        backend="networkx",
        fast_score=None,
        fast_mode="quality",
        **kwargs,
    ):
        if fast_score in (None, "auto"):
            fast_score = "tree_exact_loop" if backend == "tensor" else "exact"
        if backend == "tensor" and fast_score == "tree_exact":
            raise ValueError(
                "scaffold_batch requires a sampled-loop tensor score; use "
                "fast_score='tree_exact_loop'"
            )
        super().__init__(
            *args,
            backend=backend,
            fast_score=fast_score,
            fast_mode=fast_mode,
            **kwargs,
        )
        if (
            self.sampling_mode != "full"
            and self.sample_size > 0
            and self.cluster_add_per_round >= self.sample_size
        ):
            raise ValueError(
                "cluster_add_per_round must be smaller than sample_size for "
                "scaffold_batch; otherwise every sampled edge is committed"
            )

    def _grow(self, G, base_support, delta):
        return self._grow_batch(G, base_support, delta)


__all__ = ["ScaffoldBatchSparsifier"]
