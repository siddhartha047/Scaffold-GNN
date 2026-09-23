#!/usr/bin/env python3
"""Check the active environment without downloading data or allocating jobs."""
import importlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scaffold_gnn.runtime import available_cpus

required=['torch','torch_geometric','torch_sparse','torch_scatter','numpy','scipy','networkit','numba','ogb','sklearn','yaml']
record={'python':sys.version.split()[0], 'allocated_cpus':available_cpus(),'packages':{},'optional':{}}
failed=False
for name in required:
    try:
        module=importlib.import_module(name)
        record['packages'][name]=getattr(module,'__version__','import OK')
    except Exception as error:
        record['packages'][name]=str(error)
        failed=True
try:
    import torch
    record['cuda_available']=torch.cuda.is_available()
    record['visible_gpus']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
except ImportError:
    pass
try:
    dgl=importlib.import_module('dgl')
    record['optional']['dgl']=getattr(dgl,'__version__','import OK')
except Exception as error:
    record['optional']['dgl']='Protein profile: '+str(error)
record['optional']['julia']='available' if shutil.which('julia') else 'not on PATH (Spectral only)'
record['optional']['cmake']='available' if shutil.which('cmake') else 'not on PATH (large Spectral only)'
print(json.dumps(record,indent=2))
raise SystemExit(int(failed))
