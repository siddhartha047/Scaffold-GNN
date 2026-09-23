"""Allocation-aware resource and portable path resolution."""
from __future__ import annotations

import json
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
# Editable checkouts keep data beside run.py. A wheel must not try to create
# datasets/results inside site-packages, which can be read-only.
WORKSPACE = ROOT if (ROOT / 'run.py').is_file() else Path.cwd()


def available_cpus():
    limits = [os.cpu_count() or 1]
    if hasattr(os, 'sched_getaffinity'):
        limits.append(len(os.sched_getaffinity(0)))
    value = os.environ.get('SLURM_CPUS_PER_TASK')
    if value and value.isdigit():
        limits.append(int(value))
    return max(1, min(limits))


def resolve_workers(value='auto'):
    count = available_cpus() if str(value) == 'auto' else int(value)
    if count < 1:
        raise ValueError('workers must be positive or auto')
    return min(count, available_cpus())


def resolve_device(value='auto', *, dry_run=False):
    if value == 'cpu':
        return 'cpu'
    if dry_run:
        return 'cuda:0' if value == 'auto' else value
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No visible CUDA GPU. For a CPU run, explicitly pass --device cpu.')
    if value == 'auto':
        # Only devices exposed to this process/allocation are considered.
        index = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])
    else:
        index = int(str(value).removeprefix('cuda:'))
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f'CUDA device {index} is not visible to this process')
    return f'cuda:{index}'


def read_environment(name='public'):
    path = ROOT / 'configs' / 'public.yaml' if name == 'public' else Path(name)
    # Retain the private checkout alias without distributing site settings.
    if name == 'deception':
        path = ROOT / '.local' / 'deception.yaml'
    settings = yaml.safe_load(path.read_text())
    if settings.get('profile') == 'public' and os.environ.get('SCAFFOLD_DATA_ROOT'):
        settings['data_dir'] = os.environ['SCAFFOLD_DATA_ROOT']
    if settings.get('profile') == 'deception':
        local = ROOT / '.local' / 'deception.json'
        local_data = json.loads(local.read_text()).get('data_dir') if local.exists() else None
        settings['data_dir'] = os.environ.get('SCAFFOLD_DATA_ROOT') or local_data
        if not settings['data_dir']:
            raise ValueError('Set SCAFFOLD_DATA_ROOT for the deception profile. Public users can use --config public.')
    for key in ('data_dir', 'results_dir', 'cache_dir'):
        path = Path(os.path.expandvars(settings[key])).expanduser()
        settings[key] = str((WORKSPACE / path).resolve() if not path.is_absolute() else path.resolve())
    return settings
