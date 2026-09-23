import os
from pathlib import Path


DEFAULT_SUPPORT_ROOT = os.environ.get(
    'SUPPORT_GRAPH_STORAGE_ROOT',
    './results',
)
DEFAULT_DATA_DIR = (
    os.environ.get('SUPPORT_GRAPH_DATA_DIR')
    or os.environ.get('SCAFFOLD_DATA_ROOT')
    or os.environ.get('DATA_ROOT')
    or './data'
)
DEFAULT_CACHE_DIR = (
    os.environ.get('SUPPORT_GRAPH_CACHE_DIR')
    or os.environ.get('CACHE_ROOT')
    or str(Path(DEFAULT_SUPPORT_ROOT) / 'cache')
)
DEFAULT_RESULTS_DIR = (
    os.environ.get('SUPPORT_GRAPH_RESULTS_DIR')
    or str(Path(DEFAULT_SUPPORT_ROOT) / 'results')
)
DEFAULT_LOG_DIR = (
    os.environ.get('SUPPORT_GRAPH_LOG_DIR')
    or str(Path(DEFAULT_SUPPORT_ROOT) / 'logs')
)
