import networkit as nk
from .base import BaseSparsifier


class LSIM(BaseSparsifier):
    def __init__(self, k=None, target_ratio=None):
        self.k = int(k) if k is not None else None
        self.target_ratio = target_ratio
        if self.target_ratio is None and self.k is not None and self.k < 1:
            raise ValueError('k must be >= 1')

    def sparsify(self, data_or_graph):
        if self.target_ratio is not None:
            ratio = max(1e-6, min(1.0, float(self.target_ratio)))
        elif self.k is not None:
            G = self._ensure_nx(data_or_graph)
            m = G.number_of_edges()
            ratio = max(1e-6, min(1.0, self.k / m)) if m > 0 else 1.0
        else:
            ratio = 0.5

        nk_sp = nk.sparsification.LocalSimilaritySparsifier()
        return self._sparsify_via_nk_exact(data_or_graph, nk_sp, ratio)


def local_similarity_sparsifier(G, k=5):
    return LSIM(k=k).sparsify(G)
