#!/usr/bin/env python3
"""Compute the paper's ER artifacts before running the Spectral sparsifier."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scaffold_gnn.runtime import ROOT, read_environment, resolve_workers


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='public')
    p.add_argument('--dataset',required=True,nargs='+')
    p.add_argument('--workers',default='auto')
    p.add_argument('--backend',choices=['auto','laplacians','jlpcg','tgt'],default='auto')
    p.add_argument('--julia',default='julia')
    a=p.parse_args()
    config=read_environment(a.config)
    env=os.environ.copy()
    env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    env['SCAFFOLD_SPECTRAL_CACHE']=str(Path(config['cache_dir'])/'spectral')
    report=Path(config['results_dir'])/'spectral-precompute.csv'
    report.parent.mkdir(parents=True,exist_ok=True)
    return subprocess.call([sys.executable,str(ROOT/'RelatedMethods/Spectral/precompute_er.py'),
        '--dataset',*a.dataset,'--data-root',config['data_dir'],'--threads',str(resolve_workers(a.workers)),
        '--backend',a.backend,'--julia',a.julia,'--status-file',str(report)],cwd=ROOT,env=env)

if __name__=='__main__':
    raise SystemExit(main())
