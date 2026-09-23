import networkit as nk
from .base import BaseSparsifier


class SCANSparsifier(BaseSparsifier):
    def __init__(self, ratio=0.5, target_ratio=None):
        self.ratio = float(ratio)
        self.target_ratio = target_ratio
        if not 0.0 < self.ratio <= 1.0:
            raise ValueError('ratio must be in (0, 1]')

    def sparsify(self, data_or_graph):
        if self.target_ratio is not None:
            ratio = max(1e-6, min(1.0, float(self.target_ratio)))
        else:
            ratio = self.ratio

        nk_sp = nk.sparsification.SCANSparsifier()
        return self._sparsify_via_nk_exact(data_or_graph, nk_sp, ratio)
