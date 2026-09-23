"""Load portable, recorded experiment recipes without changing their protocol."""
from copy import deepcopy
from functools import lru_cache
import argparse
import yaml

from .runtime import ROOT


@lru_cache(maxsize=1)
def recipe_catalog():
    return yaml.safe_load((ROOT/'configs/reported_accuracy.yaml').read_text())


def load_recipe(dataset, protocol, method):
    document = recipe_catalog()
    variants = document['settings'].get(dataset, {}).get(protocol.replace('-', '_'), {})
    recipe = variants.get(method.replace('-', '_'))
    if recipe is None:
        choices = ', '.join(k.replace('_', '-') for k in variants) or 'none'
        raise ValueError(f'No {protocol} recipe for {dataset}/{method}; available methods: {choices}')
    parameters = {**document.get('training_defaults', {}),
                  **document.get('training', {}).get(dataset, {}),
                  **document.get('variant_defaults', {}).get(method.replace('-', '_'), {}),
                  **recipe['parameters']}
    return deepcopy(parameters), {k: deepcopy(v) for k, v in recipe.items() if k != 'parameters'}


@lru_cache(maxsize=1)
def core_parser():
    from .parse import parser_add_main_args
    return parser_add_main_args(argparse.ArgumentParser())


def core_arguments(parameters):
    """Serialize recorded flags, including explicit false values, via their schema."""
    parser = core_parser()
    actions = {}
    for action in parser._actions:
        actions.setdefault(action.dest, []).append(action)
    command = []
    for key, value in parameters.items():
        if value is None:
            continue
        options = actions.get(key, [])
        if not options:
            raise ValueError(f'Unknown recorded core setting: {key}')
        action = options[0]
        if isinstance(value, bool):
            boolean = next((a for a in options if isinstance(a, argparse.BooleanOptionalAction)), None)
            if boolean:
                flag = next(o for o in boolean.option_strings if o.startswith('--no-') == (not value))
                command.append(flag)
                continue
            desired = argparse._StoreTrueAction if value else argparse._StoreFalseAction
            match = next((a for a in options if isinstance(a, desired)), None)
            if match:
                command.append(match.option_strings[0])
                continue
            if any(isinstance(a, (argparse._StoreTrueAction, argparse._StoreFalseAction)) for a in options):
                if parser.get_default(key) == value:
                    continue
                raise ValueError(f'Cannot express {key}={value} on the core command line')
            value = str(value).lower()
        command.append(action.option_strings[0])
        command.extend(str(v) for v in (value if isinstance(value, (list, tuple)) else [value]))
    return command


def preset_overrides(parameters):
    from dataclasses import fields
    from configs.tunedgnn_presets import TunedGNNPreset
    names = {f.name for f in fields(TunedGNNPreset)}
    translations = dict(lr='learning_rate', local_layers='num_layers', num_heads='heads',
                        res='residual_connections', ln='layer_norm', bn='batch_norm',
                        jk='jumping_knowledge', in_dropout='input_dropout',
                        eval_step='eval_every', display_step='log_every')
    return {translations.get(k,k):v for k,v in parameters.items() if translations.get(k,k) in names}


def recipe_command(args, config, method, preset, output):
    import sys
    from pathlib import Path
    parameters = dict(config['recipe_parameters'])
    parameters.update(config['recipe_overrides'])
    parameters.update(dataset=args.dataset, sparsifier=method['sparsifier'], target_ratio=config['ratio'],
                      data_dir=config['data_dir'], epochs=preset.epochs, runs=preset.runs, seed=preset.seed,
                      scaffold_config=str(ROOT/'configs/scaffold_base.json'),
                      model_dir=str(output/'models'), joint_parallel_workers=config['workers'],
                      joint_cluster_cache_dir=str(Path(config['cache_dir'])/'partitions'),
                      sparsified_graph_cache_dir=str(Path(config['cache_dir'])/'supports'),
                      scaffold_eval_graph_bank_dir=str(output/'graph_bank'), tunedgnn_strict=True)
    if config['device']=='cpu':
        parameters['cpu']=True
    else:
        parameters['gpu']=int(config['device'].split(':')[-1])
    if parameters.get('scaffold_final_eval_views'):
        parameters['scaffold_final_eval_csv']=str(output/'multiview.csv')
    if args.weighted_paths:
        parameters['scaffold_weighted_paths']=True
    if args.smoke:
        parameters.update(eval_step=1, eval_start_epoch=1, display_step=1)
    return [sys.executable,'-u','-m','scaffold_gnn.main',*core_arguments(parameters)]
