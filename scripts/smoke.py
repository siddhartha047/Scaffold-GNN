#!/usr/bin/env python3
"""Run short method checks in the current CPU/GPU allocation; no scheduler calls."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scaffold_gnn.cli import registry
from scaffold_gnn.runtime import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods',nargs='+',default=['full','scaffold-fast','scaffold-batch','scaffold-sample'])
    p.add_argument('--dataset',default='karate')
    p.add_argument('--config',default='public')
    p.add_argument('--device',default='auto')
    p.add_argument('--workers',default='2')
    a=p.parse_args()
    outcomes={}
    for method in a.methods:
        if method not in registry():
            p.error(f'Unknown method: {method}')
        code=subprocess.call([sys.executable,str(ROOT/'run.py'),'--config',a.config,'--method',method,
            '--dataset',a.dataset,'--device',a.device,'--workers',a.workers,'--smoke'],cwd=ROOT)
        outcomes[method]=code
    print(json.dumps(outcomes,indent=2))
    return int(any(outcomes.values()))


if __name__=='__main__':
    raise SystemExit(main())
