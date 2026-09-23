import networkit as nk
from .base import BaseSparsifier


class LocalDegreeSparsifier(BaseSparsifier):
    def __init__(self, alpha=0.5, target_ratio=None):
        self.alpha = float(alpha)
        self.target_ratio = target_ratio
        if self.target_ratio is None and not 0.0 < self.alpha <= 1.0:
            raise ValueError('alpha must be in (0, 1]')

    def sparsify(self, data_or_graph):
        if self.target_ratio is not None:
            ratio = max(1e-6, min(1.0, float(self.target_ratio)))
        else:
            ratio = self.alpha

        nk_sp = nk.sparsification.LocalDegreeSparsifier()
        return self._sparsify_via_nk_exact(data_or_graph, nk_sp, ratio)
