"""Small parallel helpers for clustered sparsifier evaluation.

Two kinds of parallelism live here:

* ``evaluate_groups`` -- a thread pool over independent *cluster* groups, used
  by SCAFFOLD-Heap and SCAFFOLD-Batch;
* ``parallel_threads`` -- numba's thread count, used by the tree scorer's
  candidate-level ``prange`` kernels in :mod:`.tree_score`.

Both resolve their worker count through :func:`auto_worker_count`, so one
environment variable caps a whole batch of jobs.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
import os
import threading

try:  # pragma: no cover - depends on the install
    import numba as _numba
except Exception:  # pragma: no cover
    _numba = None

DEFAULT_WORKER_CAP = 8
_ENV_VARS = ("SCAFFOLD_NUM_WORKERS", "OMP_NUM_THREADS")

_THREAD_STATE = threading.local()


def available_cpu_count():
    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        affinity = None
    if affinity:
        return max(1, len(affinity))
    return max(1, os.cpu_count() or 1)


def _workers_from_env():
    for name in _ENV_VARS:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 0:
            return value
    return None


def auto_worker_count(requested_workers=0, cap=DEFAULT_WORKER_CAP):
    """Resolve a worker count: explicit > env > ``min(cap, cpus)``.

    The environment step is what makes concurrent runs safe on a shared node.
    These boxes have 32 cores, and several jobs each helping themselves to all
    of them slows every one of them down by roughly an order of magnitude, so
    the default is capped and ``OMP_NUM_THREADS`` (or ``SCAFFOLD_NUM_WORKERS``)
    lowers it for a whole batch at once.
    """
    requested_workers = int(requested_workers)
    available_workers = available_cpu_count()
    if requested_workers > 0:
        return max(1, min(available_workers, requested_workers))
    if requested_workers < 0:
        return available_workers
    from_env = _workers_from_env()
    if from_env is not None:
        return max(1, min(available_workers, from_env))
    return max(1, min(available_workers, int(cap)))


def split_workers(outer_tasks, workers):
    """Divide ``workers`` between an outer task pool and inner numba threads."""
    workers = max(1, int(workers))
    outer = max(1, min(int(outer_tasks), workers))
    return outer, max(1, workers // outer)


@contextlib.contextmanager
def parallel_threads(workers):
    """Set the calling thread's numba mask and restore it on exit.

    Numba's mask is thread-local: each pool task must set its own limit.
    Nested scopes may lower that limit, but cannot exceed their parent's budget.
    """
    workers = max(1, int(workers))
    if _numba is None:
        yield workers
        return

    ceiling = int(_numba.config.NUMBA_NUM_THREADS)
    target = max(1, min(workers, ceiling))

    parent = getattr(_THREAD_STATE, "limit", None)
    if parent is not None:
        target = min(target, parent)
    previous = int(_numba.get_num_threads())
    _numba.set_num_threads(target)
    _THREAD_STATE.limit = target
    try:
        yield target
    finally:
        _numba.set_num_threads(previous)
        _THREAD_STATE.limit = parent


def evaluate_groups(grouped_edges, evaluator, max_workers, executor=None):
    """Evaluate independent cluster edge groups.

    ``evaluator`` receives ``(cluster_id, edges)`` and returns the metrics dict
    for that group. A thread pool is used deliberately: NetworkX graphs are
    large to pickle, so process pools usually cost more than they save here.
    """
    items = [
        (cluster_id, list(edges))
        for cluster_id, edges in grouped_edges.items()
        if edges
    ]
    if not items:
        return {}

    workers = min(max(1, int(max_workers)), len(items))
    if workers <= 1:
        return {cluster_id: evaluator(cluster_id, edges) for cluster_id, edges in items}

    results = {}

    if executor is not None:
        futures = {
            executor.submit(evaluator, cluster_id, edges): cluster_id
            for cluster_id, edges in items
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
        return {cluster_id: results[cluster_id] for cluster_id, _ in items}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(evaluator, cluster_id, edges): cluster_id
            for cluster_id, edges in items
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return {cluster_id: results[cluster_id] for cluster_id, _ in items}
