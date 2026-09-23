"""One configuration interface for Scaffold and the comparison methods."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid
import yaml

from configs.tunedgnn_presets import get_tunedgnn_preset, TunedGNNPreset
from .runtime import ROOT, read_environment, resolve_device, resolve_workers


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='public', help='public defaults or a custom YAML path')
    p.add_argument('--method', default='scaffold-sample')
    p.add_argument('--dataset', default='citeseer')
    p.add_argument('--recipe', choices=['scaffold-1','scaffold-k','scaffold-full'],
                   help='load per-dataset settings from configs/reported_accuracy.yaml')
    p.add_argument('--ratio', type=float, help='retained fraction of undirected edges')
    p.add_argument('--device', help='auto, cpu, or cuda:N (visible-device index)')
    p.add_argument('--workers', help='auto or number of CPU workers')
    p.add_argument('--epochs', type=int)
    p.add_argument('--runs', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--backbone', help='e.g. fast-randsf, randsf, fast-maxsf')
    p.add_argument('--backend', choices=['auto','tensor','networkx'],
                   help='Fast/Batch construction backend; auto supports every backbone')
    p.add_argument('--llsf-passes', type=int, help='LLSF local-search passes')
    p.add_argument('--llsf-init', help='initial forest for LLSF, e.g. randsf')
    p.add_argument('--llsf-candidates', type=int, help='LLSF candidates per pass; 0 evaluates all')
    p.add_argument('--llsf-eval-edges', type=int, help='edges used to estimate LLSF stretch; 0 uses all')
    p.add_argument('--llsf-cycle-edges', type=int, help='LLSF removable edges per cycle; 0 uses all')
    p.add_argument('--edge-weights', choices=['uniform','cosine','euclidean','dot'])
    p.add_argument('--weighted-paths', action='store_true', help='use weighted supporting-path lengths')
    p.add_argument('--mode', choices=['single', 'multi', 'full'],
                   help='single support, union inference, or full-graph inference (default for Fast/Batch/Sample)')
    p.add_argument('--include-full-eval', action='store_true',
                   help='also report Scaffold-Full: sparse training, original full-graph evaluation (Fast/Batch/Sample)')
    p.add_argument('--views', type=int, choices=[1, 3, 5, 10])
    p.add_argument('--refresh-every', type=int)
    p.add_argument('--synchronous', action=argparse.BooleanOptionalAction, default=None,
                   help='wait for each scheduled refresh; default runs refresh asynchronously')
    p.add_argument('--sample-forests', type=int)
    p.add_argument('--batch-size', type=int, help='Scaffold-Batch candidates per cluster round')
    p.add_argument('--add-per-round', type=int)
    p.add_argument('--clusters')
    p.add_argument('--spectral-artifact', type=Path)
    p.add_argument('--smoke', action='store_true', help='one short plumbing run; not an accuracy experiment')
    p.add_argument('--dry-run', action='store_true', help='print configuration/command without loading data or requiring CUDA')
    p.add_argument('--list', action='store_true', help='list supported methods and datasets')
    return p


def registry():
    return {p.stem: yaml.safe_load(p.read_text()) for p in sorted((ROOT / 'configs/methods').glob('*.yaml'))}


def resolve(args):
    methods = registry()
    datasets = yaml.safe_load((ROOT / 'configs/datasets.yaml').read_text())
    if args.method not in methods:
        raise ValueError(f'Unknown method {args.method!r}; use --list')
    if args.dataset not in datasets and args.dataset != 'karate':
        raise ValueError(f'Unknown dataset {args.dataset!r}; use --list')
    config = read_environment(args.config)
    method = methods[args.method]
    recorded = {}
    if args.recipe:
        from .accuracy import load_recipe
        if args.mode is not None or args.views is not None or args.include_full_eval:
            raise ValueError('--recipe fixes the inference protocol; omit --mode, --views, and --include-full-eval')
        recorded, provenance = load_recipe(args.dataset, args.recipe, args.method)
        config['recipe'] = dict(name=args.recipe, **provenance)
        config['recipe_parameters'] = recorded
    override_map = dict(backbone='joint_init_support', edge_weights='scaffold_support_weight_method',
                        sample_forests='scaffold_sample_tree_count', batch_size='joint_sample_size',
                        add_per_round='joint_cluster_add_per_round', clusters='joint_cluster_count',
                        refresh_every='scaffold_resparsify_every', backend='scaffold_backend',
                        llsf_passes='llst_max_passes', llsf_init='llst_init_support',
                        llsf_candidates='llst_candidate_sample_size', llsf_eval_edges='llst_eval_sample_size',
                        llsf_cycle_edges='llst_cycle_sample_size')
    config['recipe_overrides'] = {core:getattr(args,key) for key,core in override_map.items()
                                  if getattr(args,key) is not None} if args.recipe else {}
    if args.recipe and args.synchronous is not None:
        config['recipe_overrides']['scaffold_async_resparsify'] = not args.synchronous
    if args.method in {'scaffold-fast','scaffold-batch'}:
        backbone = args.backbone or recorded.get('joint_init_support',method['parameters']['joint_init_support'])
        requested = args.backend or (recorded.get('scaffold_backend','auto') if args.backbone is None else 'auto')
        config['backend'] = construction_backend(backbone,requested)
        if args.recipe:
            config['recipe_overrides']['scaffold_backend'] = config['backend']
    defaults = dict(edge_weights='uniform', sample_forests=5, batch_size=512,
                    add_per_round=64, clusters='auto', refresh_every=1)
    for key, value in defaults.items():
        if getattr(args,key) is None:
            setattr(args,key,recorded.get(override_map[key],value))
    if args.synchronous is None:
        args.synchronous = not recorded.get('scaffold_async_resparsify',True)
    if args.views is None:
        args.views = recorded.get('scaffold_eval_ensemble_size',5)
    if args.mode is None:
        args.mode = ({'scaffold-1':'single','scaffold-k':'multi','scaffold-full':'full'}[args.recipe]
                     if args.recipe else ('full' if args.method in {'scaffold-fast','scaffold-batch','scaffold-sample'} else 'single'))
    config.update(method=args.method, dataset=args.dataset, kind=method['kind'])
    implementation=method.get('implementation',{})
    config['implementation']=implementation.get('overrides',{}).get(
        args.dataset,implementation.get('default',method.get('description','')))
    config['workers'] = resolve_workers(args.workers or config['workers'])
    config['device'] = resolve_device(args.device or config['device'], dry_run=args.dry_run)
    ratio = args.ratio if args.ratio is not None else recorded.get('target_ratio', datasets.get(args.dataset, {}).get('target_ratio', 0.7))
    if not 0 < ratio <= 1:
        raise ValueError('ratio must lie in (0, 1]')
    if args.method in {'full', 'no-graph', 'tuned-graphsage', 'tuned-graphsaint-rw'}:
        ratio = 1.0
    config['ratio'] = ratio
    model = 'sage' if args.method == 'tuned-graphsage' else 'gcn'
    preset = get_tunedgnn_preset(args.dataset, model, config['training_profile'])
    if args.dataset == 'karate':
        if not args.smoke:
            raise ValueError('Karate is a plumbing test only; pass --smoke')
        preset = TunedGNNPreset(0.01, 32, 2, 0.0005, 0.2, 1, 1, input_dropout=0.0)
    if preset is None:
        raise ValueError(f'No shared preset for {args.dataset}/{model}')
    if recorded:
        from .accuracy import preset_overrides
        preset = replace(preset, **preset_overrides(recorded))
    preset = replace(preset, epochs=args.epochs if args.epochs is not None else (1 if args.smoke else preset.epochs),
                     runs=args.runs if args.runs is not None else (1 if args.smoke else preset.runs),
                     seed=args.seed if args.seed is not None else recorded.get('seed',config['seed']))
    if preset.epochs < 1 or preset.runs < 1:
        raise ValueError('epochs and runs must be positive')
    config['training'] = asdict(preset)
    config['mode'] = args.mode
    config['include_full_eval'] = args.include_full_eval
    config['smoke'] = args.smoke
    if args.mode == 'multi' and method['kind'] != 'scaffold':
        raise ValueError('--mode multi applies only to Scaffold')
    if args.mode == 'multi' and args.method not in {'scaffold-fast','scaffold-batch','scaffold-sample'}:
        raise ValueError('The final union-inference runner supports Fast, Batch, and Sample.')
    if args.mode == 'full' and method['kind'] != 'scaffold':
        raise ValueError('--mode full applies only to Scaffold; --method full is the full-graph baseline')
    if args.mode == 'full' and args.include_full_eval:
        raise ValueError('--mode full already evaluates the full graph; omit --include-full-eval')
    if args.include_full_eval and args.method not in {'scaffold-fast','scaffold-batch','scaffold-sample'}:
        raise ValueError('--include-full-eval supports Scaffold Fast, Batch, and Sample.')
    if (args.refresh_every < 1 and args.mode != 'single') or args.sample_forests < 1:
        raise ValueError('refresh-every and sample-forests must be positive')
    if args.method == 'scaffold-batch' and (args.batch_size < 1 or not 1 <= args.add_per_round <= args.batch_size):
        raise ValueError('Batch requires 1 <= add-per-round <= batch-size')
    return config, method, preset


def construction_backend(backbone, requested='auto'):
    from .sparsifiers.scaffold.spanning_tree import canonical_support_name
    tensor_supported = canonical_support_name(backbone) in {
        'maxst','mst','fast_maxst','fast_mst','randst','fast_randst'}
    if requested == 'tensor' and not tensor_supported:
        raise ValueError(f'Backbone {backbone} requires --backend networkx (or auto)')
    return ('tensor' if tensor_supported else 'networkx') if requested == 'auto' else requested


def build_command(args, config, method, preset, output):
    """Commands use argv lists, never shell interpolation."""
    if args.recipe:
        from .accuracy import recipe_command
        return recipe_command(args,config,method,preset,output)
    c = [sys.executable, '-u', '-m', 'scaffold_gnn.main']
    flags = {}
    if method['kind'] == 'baseline':
        c = ['bash', str(ROOT / method['runner'])]
        flags = dict(dataset=args.dataset, **{'data-root': config['data_dir'], 'cache-root':config['cache_dir'],
            'results-root': str(output), 'kept-ratio':config['ratio'],
            'device':'0' if config['device']=='cpu' else config['device'].split(':')[-1],
            'python':sys.executable, 'epochs':preset.epochs, 'runs':preset.runs, 'seed':preset.seed,
            'hidden-channels':preset.hidden_channels, 'num-layers':preset.num_layers, 'heads':preset.heads,
            'learning-rate':preset.learning_rate, 'weight-decay':preset.weight_decay, 'dropout':preset.dropout,
            'input-dropout':preset.input_dropout, 'metric':preset.metric, 'preset-profile':config['training_profile']})
        for flag, attr in [('pre-linear','pre_linear'), ('residual-connections','residual_connections'),
                           ('layer-norm','layer_norm'), ('batch-norm','batch_norm'), ('jumping-knowledge','jumping_knowledge')]:
            c.append('--' + (flag if getattr(preset,attr) else 'no-' + flag))
        if args.smoke:
            c.append('--smoke-test')
    else:
        if method['kind'] == 'mlp':
            c[-1] = 'scaffold_gnn.main_mlp'
        flags = dict(dataset=args.dataset, data_dir=config['data_dir'], epochs=preset.epochs, runs=preset.runs,
                     seed=preset.seed, hidden_channels=preset.hidden_channels, local_layers=preset.num_layers,
                     lr=preset.learning_rate, weight_decay=preset.weight_decay, dropout=preset.dropout,
                     in_dropout=preset.input_dropout, metric=preset.metric, model_profile=preset.model_profile,
                     optimizer=preset.optimizer, lr_scheduler=preset.lr_scheduler,
                     lr_scheduler_factor=preset.lr_scheduler_factor, lr_scheduler_patience=preset.lr_scheduler_patience,
                     eval_step=1 if args.smoke else preset.eval_every,
                     eval_start_epoch=1 if args.smoke else preset.eval_start_epoch,
                     display_step=1 if args.smoke else preset.log_every, model_dir=str(output/'models'))
        for flag, val in [('res',preset.residual_connections),('ln',preset.layer_norm),('bn',preset.batch_norm)]:
            c.append('--' + (flag if val else 'no_' + flag))
        if preset.pre_linear:
            c.append('--pre_linear')
        if config['device'] == 'cpu':
            c.append('--cpu')
        else:
            flags['gpu'] = config['device'].split(':')[-1]
        if method['kind'] != 'mlp':
            flags.update(sparsifier=method['sparsifier'], target_ratio=config['ratio'], gnn='gcn', num_heads=preset.heads,
                         scaffold_config=str(ROOT/'configs/scaffold_base.json'), joint_parallel_workers=config['workers'],
                         joint_cluster_count=args.clusters, joint_cluster_cache_dir=str(Path(config['cache_dir'])/'partitions'),
                         sparsified_graph_cache_dir=str(Path(config['cache_dir'])/'supports'),
                         scaffold_eval_graph_bank_dir=str(output/'graph_bank'),
                         scaffold_resparsify_every=0 if args.mode=='single' else args.refresh_every)
            if preset.jumping_knowledge:
                c.append('--jk')
            if args.dataset != 'karate':
                c.append('--tunedgnn_strict')
            if preset.model_profile == 'products':
                flags['large_graph_pipeline'] = 'random_node_loader'
            flags.update(method.get('parameters', {}))
            if method['kind']=='scaffold':
                flags.update(joint_init_support=args.backbone or flags['joint_init_support'],
                             joint_node_beta=0.0, joint_sample_size=args.batch_size,
                             joint_cluster_add_per_round=args.add_per_round,
                             scaffold_support_weight_method=args.edge_weights)
                if args.weighted_paths:
                    c.append('--scaffold_weighted_paths')
                if 'backend' in config:
                    flags['scaffold_backend']=config['backend']
                for key,flag in dict(llsf_passes='llst_max_passes', llsf_init='llst_init_support',
                                     llsf_candidates='llst_candidate_sample_size', llsf_eval_edges='llst_eval_sample_size',
                                     llsf_cycle_edges='llst_cycle_sample_size').items():
                    if getattr(args,key) is not None:
                        flags[flag]=getattr(args,key)
                if args.method=='scaffold-sample':
                    flags['scaffold_sample_tree_count']=args.sample_forests
                if args.mode=='full':
                    flags.update(eval_graph='original', scaffold_eval_ensemble_size=1,
                                 scaffold_eval_ensemble_source='fresh', scaffold_fast_maxst_every=0)
                if args.mode!='single':
                    c.append('--scaffold_resparsify_include_final')
                    c.append('--no-scaffold_async_resparsify' if args.synchronous else '--scaffold_async_resparsify')
                if args.mode=='multi' or args.include_full_eval:
                    views = sorted({1,args.views}) if args.mode=='multi' else [1]
                    if args.include_full_eval:
                        views.append('all')
                        c.append('--scaffold_final_best_full_checkpoint')
                    flags.update(scaffold_final_eval_views=views,
                                 scaffold_final_graph_bank_size=args.views if args.mode=='multi' else 1,
                                 scaffold_final_eval_csv=str(output/'multiview.csv'),
                                 eval_graph='sparse', scaffold_eval_ensemble_source='best', scaffold_eval_ensemble_size=1)
                    c.append('--scaffold_final_report_best_checkpoint')
                    c.append('--report_final_epoch_only')
            if args.spectral_artifact:
                flags['effective_resistance_path']=str(args.spectral_artifact.resolve())
    for key,value in flags.items():
        c.append('--'+key)
        c.extend(map(str,value if isinstance(value,list) else [value]))
    return c


def main(argv=None):
    args = parser().parse_args(argv)
    if args.list:
        print('Methods: ' + ', '.join(registry()))
        print('Datasets: ' + ', '.join(yaml.safe_load((ROOT/'configs/datasets.yaml').read_text())))
        return 0
    try:
        config, method, preset = resolve(args)
        name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
        output=Path(config['results_dir'])/args.dataset/args.method/('dry-run' if args.dry_run else name)
        command=build_command(args,config,method,preset,output)
        config['command']=command
        config['output_dir']=str(output)
        if args.dry_run:
            print(json.dumps(config,indent=2))
            return 0
        output.mkdir(parents=True,exist_ok=False)
        (output/'resolved.json').write_text(json.dumps(config,indent=2)+'\n')
        env=os.environ.copy()
        env.update(PYTHONPATH=str(ROOT)+os.pathsep+env.get('PYTHONPATH',''), PYTHONUNBUFFERED='1',
                   SCAFFOLD_DATA_ROOT=config['data_dir'], SUPPORT_GRAPH_DATA_DIR=config['data_dir'],
                   SCAFFOLD_CACHE_ROOT=config['cache_dir'], SUPPORT_GRAPH_CACHE_DIR=config['cache_dir'],
                   SCAFFOLD_SPECTRAL_CACHE=str(Path(config['cache_dir'])/'spectral'),
                   SCAFFOLD_SCRATCH_ROOT=config['cache_dir'], RESULTS_ROOT=str(output),
                   SUPPORT_GRAPH_RESULTS_DIR=str(output), SUPPORT_GRAPH_LOG_DIR=str(output),
                   SUPPORT_GRAPH_STORAGE_ROOT=str(output), SCAFFOLD_METHOD_SCRATCH=str(output/'work'),
                   BASELINE_INDIVIDUAL_RUNS_CSV=str(output/'metrics.csv'),
                   SCAFFOLD_DATASET_SEED=str(preset.seed), SUPPORT_GRAPH_SPLIT_SEED=str(preset.seed),
                   SCAFFOLD_SPLIT_PROTOCOL='tunedgnn', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
                   MPLCONFIGDIR=str(Path(config['cache_dir'])/'matplotlib'),
                   NUMBA_CACHE_DIR=str(Path(config['cache_dir'])/'numba'))
        env.update({k:str(v) for k,v in method.get('environment',{}).items()})
        env.update(config.get('recipe',{}).get('environment',{}))
        if config['device']=='cpu':
            env['CUDA_VISIBLE_DEVICES']=''
        if args.smoke:
            env.update(ADAGLT_MASK_EPOCHS='1', BASELINE_TIME_BUDGET_SECONDS='180')
        print('Output: '+str(output),flush=True)
        print('Implementation: '+config['implementation'],flush=True)
        print(shlex.join(command),flush=True)
        started=time.perf_counter()
        with (output/'run.log').open('w') as log:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            try:
                for line in proc.stdout:
                    print(line,end='',flush=True)
                    log.write(line); log.flush()
                code=proc.wait()
            except KeyboardInterrupt:
                proc.terminate(); proc.wait()
                code=130
        (output/'status.json').write_text(json.dumps({'exit_code':code,'wall_seconds':time.perf_counter()-started},indent=2)+'\n')
        return code
    except (ValueError,RuntimeError,FileNotFoundError) as exc:
        print(f'Error: {exc}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
