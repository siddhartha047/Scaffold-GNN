from pathlib import Path
import argparse
import pytest
import yaml

from scaffold_gnn.cli import parser, registry, resolve, build_command
from scaffold_gnn.runtime import ROOT, resolve_workers


def test_all_paper_combinations_have_shared_presets_and_valid_core_arguments():
    from scaffold_gnn.parse import parser_add_main_args
    from scaffold_gnn.main_mlp import parser_add_main_args as mlp_args
    core=parser_add_main_args(argparse.ArgumentParser())
    mlp=mlp_args(argparse.ArgumentParser())
    datasets=yaml.safe_load((ROOT/'configs/datasets.yaml').read_text())
    for method in registry():
        for dataset in datasets:
            args=parser().parse_args(['--method',method,'--dataset',dataset,'--dry-run'])
            conf,spec,preset=resolve(args)
            command=build_command(args,conf,spec,preset,ROOT/'results'/'test-config')
            assert preset.hidden_channels>0 and preset.num_layers>0
            assert conf['ratio']>0
            if spec['kind']!='baseline':
                parsed=(mlp if spec['kind']=='mlp' else core).parse_args(command[4:])
                assert parsed.hidden_channels==preset.hidden_channels
                assert parsed.epochs==preset.epochs
            else:
                assert Path(command[1]).is_file()


def test_worker_count_respects_allocation(monkeypatch):
    monkeypatch.setattr('scaffold_gnn.runtime.available_cpus',lambda:4)
    assert resolve_workers('auto')==4
    assert resolve_workers('32')==4
    assert resolve_workers('2')==2
    with pytest.raises(ValueError):
        resolve_workers('0')


def test_shared_loader_reaches_core_without_changing_splits(tmp_path,monkeypatch):
    import torch
    from scaffold_gnn.data import load_dataset as shared_load
    from scaffold_gnn.utils.dataset import load_dataset as core_load
    monkeypatch.setenv('SCAFFOLD_DATASET_SEED','42')
    shared=shared_load(tmp_path,'karate',seed=42)
    core=core_load(tmp_path,'karate')
    assert torch.equal(shared.data.edge_index,core.graph['edge_index'])
    assert torch.equal(shared.data.x,core.graph['node_feat'])
    for a,b in zip(shared.splits,core.shared_splits):
        assert a.keys()==b.keys()
        assert all(torch.equal(a[k],b[k]) for k in a)


@pytest.mark.parametrize('forest,legacy',[('randsf','randst'),('maxsf','maxst'),('minsf','mst'),
    ('fast-randsf','fast-randst'),('fast-maxsf','fast-maxst'),('glsf','glst'),('llsf','llst')])
def test_forest_aliases(forest,legacy):
    from scaffold_gnn.sparsifiers.scaffold.spanning_tree import canonical_support_name
    assert canonical_support_name(forest)==canonical_support_name(legacy)


def test_public_paths_are_repository_relative():
    from scaffold_gnn.runtime import read_environment
    public=read_environment('public')
    assert Path(public['data_dir'])==ROOT/'data'
    assert Path(public['results_dir'])==ROOT/'results'


def test_multiview_command_satisfies_core_protocol():
    from scaffold_gnn.parse import parser_add_main_args
    from scaffold_gnn.utils.scaffold_multiview import validate_final_views
    args=parser().parse_args(['--method','scaffold-sample','--dataset','cora','--mode','multi','--views','3','--dry-run'])
    conf,spec,preset=resolve(args)
    command=build_command(args,conf,spec,preset,ROOT/'results'/'test-multi')
    parsed=parser_add_main_args(argparse.ArgumentParser()).parse_args(command[4:])
    assert validate_final_views(parsed)==('1','3')
    assert parsed.scaffold_resparsify_every==20


def test_anonymous_export_keeps_data_loader_not_local_data():
    from scripts.export_anonymous import source_files
    names={str(rel) for _,rel in source_files()}
    assert 'scaffold_gnn/data/datasets.py' in names
    assert 'scaffold_gnn/data/connector.py' in names
    assert not any(name.startswith(('.local/','results/','data/','.git/')) for name in names)


@pytest.mark.parametrize('profile',['ogbn-arxiv','ogbn-proteins'])
def test_large_graph_mog_sources_load_locally(profile):
    import torch
    from scripts.methods.scalable_sparse_node import NativeMoGEdgeScorer
    model=NativeMoGEdgeScorer(8,0.7,profile,hidden_channels=8)
    x=torch.randn(6,8,requires_grad=True)
    edges=torch.cartesian_prod(torch.arange(6),torch.arange(6)).t().contiguous()
    output=model(x,edges,torch.ones(edges.shape[1],8))
    assert output.numel()==edges.shape[1]
    assert torch.isfinite(output).all()
    (output.sum()+model.auxiliary_loss).backward()
