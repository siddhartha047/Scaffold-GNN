"""Train on a pre-built, pre-measured sequence of supports.

``fixed_support`` freezes one support for a whole run, which is what every grid
experiment uses.  This sparsifier instead holds ``T`` supports in memory and
hands out support ``t`` when the training loop asks for a refresh, so a run can
see many sparse graphs while the *sequence itself* is an input to the
experiment rather than a side effect of it.

Why a stored sequence instead of calling a selector per epoch: the certificate
(D, C_E, C_V) of every support is measured once, offline, by the builder.  Every
arm that consumes the same artifact therefore trains on the byte-identical
support sequence with byte-identical certificates, which is the only way a
paired comparison between "one support" and "many supports" is about the number
of supports rather than about which supports happened to be drawn.

Index contract.  ``main.py`` resets its refresh counter at the top of every run
and calls ``resparsify(seed=base + k)`` for ``k = 1, 2, ...``, where ``base`` is
the ``seed`` entry of the sparsifier params.  Index ``k`` modulo ``T`` therefore
identifies the support, is independent of the GNN run, and repeats exactly
across runs and across arms.  ``sparsify()`` -- the initial graph -- is index 0.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .fixed_support import FixedSupportSparsifier


class SupportSequenceSparsifier(FixedSupportSparsifier):
    """Serve support ``k mod T`` from a stored sequence artifact.

    The artifact is a ``torch.save`` dictionary with

    ``supports``   list of ``[2, q_t]`` long tensors, canonical ``u < v``,
    ``num_nodes``  node count the supports are indexed against,
    ``seeds``      generator seed behind each support (recorded, not used here).
    """

    def __init__(
        self,
        sequence_path,
        seed=0,
        sequence_sha256=None,
        target_ratio=None,
        log_every=100,
    ):
        path = Path(sequence_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"support sequence does not exist: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch before weights_only support.
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or "supports" not in payload:
            raise ValueError(f"{path}: expected a dict with a 'supports' entry")

        self.sequence_path = str(path)
        # Reuse the parent's identity fields so the sparse-graph cache key and
        # the artifact-mismatch errors read the same as for fixed_support.
        self.support_path = str(path)
        self.support_sha256 = sequence_sha256
        self.target_ratio = target_ratio
        self.seed = int(seed)
        self.log_every = int(log_every)

        self._num_nodes = int(payload["num_nodes"])
        self._seeds = [int(s) for s in payload.get("seeds", [])]
        self._supports = [
            self._checked(torch.as_tensor(pairs, dtype=torch.long), position)
            for position, pairs in enumerate(payload["supports"])
        ]
        if not self._supports:
            raise ValueError(f"{path}: sequence is empty")
        self._index = 0
        self._served = 0
        print(
            f"[SupportSequence] {path.name}: T={len(self._supports)} supports, "
            f"n={self._num_nodes}, q={self._supports[0].size(1)}, "
            f"base_seed={self.seed}",
            flush=True,
        )

    def _checked(self, pairs: torch.Tensor, position: int) -> torch.Tensor:
        """Apply the fixed_support contract to one member of the sequence."""
        where = f"{Path(self.sequence_path).name}[{position}]"
        if pairs.ndim != 2 or pairs.size(0) != 2:
            raise ValueError(f"{where}: pairs must have shape [2, q]")
        if not pairs.numel():
            raise ValueError(f"{where}: support has no edges")
        if int(pairs.min()) < 0 or int(pairs.max()) >= self._num_nodes:
            raise ValueError(
                f"{where}: node id lies outside [0, {self._num_nodes - 1}]"
            )
        if bool((pairs[0] >= pairs[1]).any()):
            raise ValueError(f"{where}: pairs must be canonical with u < v")
        if torch.unique(pairs, dim=1).size(1) != pairs.size(1):
            raise ValueError(f"{where}: duplicate undirected edges")
        return pairs.contiguous()

    @property
    def length(self) -> int:
        return len(self._supports)

    def _load_pairs(self, num_nodes: int) -> torch.Tensor:
        """Return the currently selected support instead of reading a file."""
        if int(num_nodes) != self._num_nodes:
            raise ValueError(
                f"{self.sequence_path}: sequence has {self._num_nodes} nodes; "
                f"dataset has {int(num_nodes)}"
            )
        return self._supports[self._index]

    def cache_source(self, data_or_graph):
        """Remember the pre-sparsification input, as the scaffold family does."""
        self._resparsify_source = data_or_graph

    def resparsify(self, seed=None):
        """Advance to the support the refresh seed names and return it.

        ``seed`` is ``base + k`` with ``k`` restarting at 1 each run, so the
        index depends only on how far into the run the loop is.
        """
        source = getattr(self, "_resparsify_source", None)
        if source is None:
            raise RuntimeError(
                "resparsify() requires cache_source(data) to be called first."
            )
        step = 0 if seed is None else int(seed) - self.seed
        self._index = step % len(self._supports)
        self._served += 1
        if self._served <= 3 or self._served % max(1, self.log_every) == 0:
            print(
                f"[SupportSequence] refresh {self._served}: seed={seed} "
                f"-> index {self._index} "
                f"(q={self._supports[self._index].size(1)})",
                flush=True,
            )
        return self.sparsify(source)
