import networkit as nk
from .base import BaseSparsifier


class GSPAR(BaseSparsifier):
    def __init__(self, target_ratio=1.0):
        self.target_ratio = float(target_ratio)
        if not 0.0 < self.target_ratio <= 1.0:
            raise ValueError('target_ratio must be in (0, 1]')

    def sparsify(self, data_or_graph):
        nk_sp = nk.sparsification.JaccardSimilaritySparsifier()
        return self._sparsify_via_nk_exact(data_or_graph, nk_sp, self.target_ratio)


def g_spar(G, target_ratio):
    return GSPAR(target_ratio=target_ratio).sparsify(G)
