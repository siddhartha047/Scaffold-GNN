from importlib import import_module

__all__ = [
    'pyg_to_nx',
    'nx_to_pyg',
    'get_sparsifier',
    'set_seed',
    'set_seed_from_args',
    'eval_acc',
    'label_count_distribution',
    'eval_acc_per_class',
    'average_acc_per_class',
    'eval_f1',
    'eval_f1_macro',
    'eval_rocauc',
    'trace_support_graph',
    'initial_support_graph',
    'save_support_progression_gif',
    'support_graph_at_step',
]

_EXPORTS = {
    'pyg_to_nx': ('scaffold_gnn.utils.graph_conversion', 'pyg_to_nx'),
    'nx_to_pyg': ('scaffold_gnn.utils.graph_conversion', 'nx_to_pyg'),
    'get_sparsifier': ('scaffold_gnn.utils.factory', 'get_sparsifier'),
    'set_seed': ('scaffold_gnn.utils.seed', 'set_seed'),
    'set_seed_from_args': ('scaffold_gnn.utils.seed', 'set_seed_from_args'),
    'eval_acc': ('scaffold_gnn.utils.data_utils', 'eval_acc'),
    'label_count_distribution': ('scaffold_gnn.utils.data_utils', 'label_count_distribution'),
    'eval_acc_per_class': ('scaffold_gnn.utils.data_utils', 'eval_acc_per_class'),
    'average_acc_per_class': ('scaffold_gnn.utils.data_utils', 'average_acc_per_class'),
    'eval_f1': ('scaffold_gnn.utils.data_utils', 'eval_f1'),
    'eval_f1_macro': ('scaffold_gnn.utils.data_utils', 'eval_f1_macro'),
    'eval_rocauc': ('scaffold_gnn.utils.data_utils', 'eval_rocauc'),
    'trace_support_graph': ('scaffold_gnn.utils.support_graph_analysis', 'trace_support_graph'),
    'initial_support_graph': ('scaffold_gnn.utils.support_graph_analysis', 'initial_support_graph'),
    'save_support_progression_gif': ('scaffold_gnn.utils.support_graph_viz', 'save_support_progression_gif'),
    'support_graph_at_step': ('scaffold_gnn.utils.support_graph_viz', 'support_graph_at_step'),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
