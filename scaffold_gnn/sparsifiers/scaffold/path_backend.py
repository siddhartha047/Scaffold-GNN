from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - optional acceleration
    njit = None


def _bfs_parents(rowptr, neighbors, source, targets):
    """Read-only adjacency; each search owns its queue and parent array."""
    n = rowptr.size - 1
    parent = np.full(n, -1, dtype=np.int64)
    wanted = np.zeros(n, dtype=np.bool_)
    wanted[targets] = True
    wanted[source] = False
    remaining = int(wanted.sum())
    parent[source] = source
    queue = np.empty(n, dtype=np.int64)
    queue[0] = source
    head, tail = 0, 1
    while head < tail and remaining:
        node = queue[head]
        head += 1
        for j in range(rowptr[node], rowptr[node + 1]):
            neighbor = neighbors[j]
            if parent[neighbor] >= 0:
                continue
            parent[neighbor] = node
            queue[tail] = neighbor
            tail += 1
            if wanted[neighbor]:
                remaining -= 1
                if remaining == 0:
                    break
    return parent


_bfs_kernel = njit(cache=True, nogil=True)(_bfs_parents) if njit is not None else None


class UnweightedPathBackend:
    """BFS path helper for small unweighted graphs with nodes 0..n-1."""

    def __init__(self, adj):
        self.adj = adj
        self._csr = None
        self._csr_lock = threading.Lock()

    @property
    def parallel_available(self):
        return _bfs_kernel is not None

    def _snapshot(self):
        # Builds happen before reads; add_edge invalidates between rounds.
        # The lock also protects the first access by Heap's cluster workers.
        with self._csr_lock:
            if self._csr is None:
                rows = [sorted(neighbors) for neighbors in self.adj]
                rowptr = np.zeros(len(rows) + 1, dtype=np.int64)
                np.cumsum([len(row) for row in rows], out=rowptr[1:])
                neighbors = np.fromiter(
                    (v for row in rows for v in row), dtype=np.int64,
                    count=int(rowptr[-1]),
                )
                self._csr = rowptr, neighbors
            return self._csr

    def paths_for_sources(self, targets_by_source, workers=1):
        """Independent searches on one support snapshot, in source order."""
        items = list(targets_by_source.items())
        if not items:
            return {}
        if not self.parallel_available or workers <= 1 or len(items) == 1:
            return {source: self.multi_target_paths(source, targets)
                    for source, targets in items}
        self._snapshot()
        with ThreadPoolExecutor(max_workers=min(int(workers), len(items))) as pool:
            paths = pool.map(lambda item: self.multi_target_paths(*item), items)
            return {source: found for (source, _), found in zip(items, paths)}

    @classmethod
    def maybe_build(cls, H):
        nodes = list(H.nodes())
        n = len(nodes)
        if any(not isinstance(node, int) for node in nodes):
            return None
        if set(nodes) != set(range(n)):
            return None

        adj = [set() for _ in range(n)]
        for u, v in H.edges():
            if not isinstance(u, int) or not isinstance(v, int):
                return None
            if u < 0 or v < 0 or u >= n or v >= n:
                return None
            adj[u].add(v)
            adj[v].add(u)
        return cls(adj)

    def add_edge(self, u, v):
        if not self._has_node(u) or not self._has_node(v):
            return
        self.adj[u].add(v)
        self.adj[v].add(u)
        self._csr = None

    def shortest_path(self, source, target):
        return self.multi_target_paths(source, {target}).get(target)

    def multi_target_paths(self, source, targets):
        targets = {target for target in targets if self._has_node(target)}
        if not self._has_node(source) or not targets:
            return {}

        if _bfs_kernel is not None:
            parent = _bfs_kernel(
                *self._snapshot(), source, np.asarray(sorted(targets), dtype=np.int64)
            )
            paths = {}
            for target in targets:
                if parent[target] < 0:
                    continue
                path = [target]
                node = target
                while node != source:
                    node = int(parent[node])
                    path.append(node)
                paths[target] = path[::-1]
            return paths

        parent = {source: None}
        queue = deque([source])
        remaining = set(targets)
        remaining.discard(source)

        while queue and remaining:
            node = queue.popleft()
            for nbr in sorted(self.adj[node]):
                if nbr in parent:
                    continue
                parent[nbr] = node
                remaining.discard(nbr)
                queue.append(nbr)
                if not remaining:
                    break

        paths = {}
        for target in targets:
            if target not in parent:
                continue
            path = []
            node = target
            while node is not None:
                path.append(node)
                node = parent[node]
            paths[target] = list(reversed(path))
        return paths

    def radius_nodes(self, source, cutoff):
        if not self._has_node(source):
            return set()
        if cutoff < 0:
            return set()

        seen = {source}
        queue = deque([(source, 0)])
        while queue:
            node, dist = queue.popleft()
            if dist >= cutoff:
                continue
            for nbr in sorted(self.adj[node]):
                if nbr in seen:
                    continue
                seen.add(nbr)
                queue.append((nbr, dist + 1))
        return seen

    def _has_node(self, node):
        return isinstance(node, int) and 0 <= node < len(self.adj)
