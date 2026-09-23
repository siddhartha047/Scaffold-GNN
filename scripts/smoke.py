#!/usr/bin/env python3
"""Run short method/backbone checks in the current allocation; no scheduler calls."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scaffold_gnn.cli import registry
from scaffold_gnn.runtime import ROOT, WORKSPACE

BACKBONES = ['fast-randsf','randsf','maxsf','minsf','fast-maxsf','fast-minsf',
             'randspt','slsf','glsf','llsf']


def run_case(name, command, directory, timeout):
    started = time.perf_counter()
    print(f'Running {name}; log: {directory / (name + ".log")}', flush=True)
    with (directory / (name + '.log')).open('w') as log:
        process = subprocess.Popen(command, cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if isinstance(exc, KeyboardInterrupt):
                raise
            code = 124
    result = dict(case=name, exit_code=code, wall_seconds=time.perf_counter()-started, command=command)
    (directory / (name + '.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(f'{name}: {"PASS" if code == 0 else "FAIL"} (exit {code})', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    methods = parser.add_mutually_exclusive_group()
    methods.add_argument('--methods', nargs='*', help='method names; use an empty list for backbone-only checks')
    methods.add_argument('--all-methods', action='store_true')
    parser.add_argument('--backbones', nargs='+', choices=BACKBONES)
    parser.add_argument('--all-backbones', action='store_true')
    parser.add_argument('--dataset', default='karate')
    parser.add_argument('--config', default='public')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--workers', default='2')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--timeout', type=float, default=1800, help='seconds per check')
    parser.add_argument('--output', type=Path, help='directory for check logs and summary.json')
    args = parser.parse_args()
    if args.epochs < 1 or args.timeout <= 0:
        parser.error('epochs and timeout must be positive')
    available = registry()
    selected = list(available) if args.all_methods else args.methods
    if selected is None:
        selected = ['full','scaffold-fast','scaffold-batch','scaffold-sample']
    for method in selected:
        if method not in available:
            parser.error(f'Unknown method: {method}')
    backbones = BACKBONES if args.all_backbones else args.backbones or []
    if not selected and not backbones:
        parser.error('select at least one method or backbone')
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    directory = (args.output or WORKSPACE/'results'/'smoke'/name).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    common = [sys.executable,'-m','scaffold_gnn.cli','--config',args.config,'--dataset',args.dataset,
              '--device',args.device,'--workers',args.workers,'--smoke','--epochs',str(args.epochs),'--runs','1']
    cases = []
    for method in selected:
        if method == 'spectral':
            cases.append(('spectral-precompute', [sys.executable,str(ROOT/'scripts/precompute_spectral.py'),
                          '--config',args.config,'--dataset',args.dataset,'--workers',args.workers]))
        cases.append((method, common+['--method',method]))
    for backbone in backbones:
        command = common+['--method','scaffold-fast','--mode','single','--clusters','1','--backbone',backbone]
        if backbone == 'llsf':
            # Exercise the same local-search code with bounded work on Cora.
            command += ['--llsf-init','randsf','--llsf-passes','1','--llsf-candidates','16',
                        '--llsf-eval-edges','128','--llsf-cycle-edges','4']
        cases.append(('backbone-'+backbone, command))
    outcomes = []
    for name, command in cases:
        outcomes.append(run_case(name, command, directory, args.timeout))
        (directory/'summary.json').write_text(json.dumps(outcomes,indent=2)+'\n')
    failed = [r['case'] for r in outcomes if r['exit_code']]
    print(f'Completed: {len(outcomes)-len(failed)}/{len(outcomes)} passed. Results: {directory}')
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
