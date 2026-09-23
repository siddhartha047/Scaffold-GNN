"""Opt-in epoch refresh for the existing uniform Random baseline."""
from torch_geometric.data import Data

from .random_dropout import RandomDropout


class RandomRefresh(RandomDropout):
    """Draw every support from the original graph, using a local seeded RNG.

    Keeping this separate from RandomDropout preserves the fixed Random-1
    baseline, even though main.py defaults its refresh interval to one.
    """

    def cache_source(self, data):
        if not isinstance(data, Data):
            raise TypeError('RandomRefresh requires a PyG Data source')
        self._source = Data(
            x=data.x, y=data.y, num_nodes=data.num_nodes,
            edge_index=self._pyg_undirected_pairs(data).clone(),
        )
        self._source.edge_index_is_undirected_unique = True
        for key in ('train_mask', 'val_mask', 'test_mask'):
            if hasattr(data, key):
                setattr(self._source, key, getattr(data, key))

    def sparsify(self, data):
        self.cache_source(data)
        return super().sparsify(self._source)

    def resparsify(self, seed=None):
        if not hasattr(self, '_source'):
            raise RuntimeError('Cache the original graph before refreshing')
        previous = self.seed
        try:
            if seed is not None:
                self.seed = int(seed)
            return super().sparsify(self._source)
        finally:
            self.seed = previous
