def get_sparsifier(name, **kwargs):
    name = str(name).strip().lower().replace('-', '_')
    if name == 'scaffold_exact':
        name = 'scaffold_greedy'
    mapping = {
        'mst': ('scaffold_gnn.sparsifiers.mst', 'MSTSparsifier'),
        'maxst': ('scaffold_gnn.sparsifiers.max_spanning_tree', 'MaxSTSparsifier'),
        'fast_maxst': ('scaffold_gnn.sparsifiers.fast_maxst', 'FastMaxSTSparsifier'),
        'fast-maxst': ('scaffold_gnn.sparsifiers.fast_maxst', 'FastMaxSTSparsifier'),
        'effective_resistance': ('scaffold_gnn.sparsifiers.effective_resistance', 'EffectiveResistanceSparsifier'),
        'er_spectral': ('scaffold_gnn.sparsifiers.effective_resistance', 'EffectiveResistanceSparsifier'),
        'effective_resistance_fixed': ('scaffold_gnn.sparsifiers.effective_resistance', 'EffectiveResistanceExactSparsifier'),
        'er_fix': ('scaffold_gnn.sparsifiers.effective_resistance', 'EffectiveResistanceExactSparsifier'),
        'las_vegas_spanner': ('scaffold_gnn.sparsifiers.las_vegas_spanner', 'LasVegasSpannerSparsifier'),
        'tspanner_greedy': ('scaffold_gnn.sparsifiers.spanner_greedy', 'GreedySpanner'),
        'tspanner_halperin': ('scaffold_gnn.sparsifiers.spanner_halperin', 'HalperinSpanner'),
        'k_random_neighbor': ('scaffold_gnn.sparsifiers.k_rand_neighbors', 'KRandomNeighbor'),
        'random': ('scaffold_gnn.sparsifiers.random_dropout', 'RandomDropout'),
        'random_refresh': ('scaffold_gnn.sparsifiers.random_refresh', 'RandomRefresh'),
        'er': ('scaffold_gnn.sparsifiers.er', 'ERSparsifier'),
        'support': ('scaffold_gnn.sparsifiers.support_graph', 'SupportGraphSparsifier'),
        'support_dilation': ('scaffold_gnn.sparsifiers.support_graph_dilation', 'SupportGraphDilationSparsifier'),
        'glst': ('scaffold_gnn.sparsifiers.low_stretch_tree', 'GreedyLowStretchTreeSparsifier'),
        'low_stretch_tree': ('scaffold_gnn.sparsifiers.low_stretch_tree', 'GreedyLowStretchTreeSparsifier'),
        'slst': ('scaffold_gnn.sparsifiers.scalable_low_stretch_tree', 'ScalableLowStretchTreeSparsifier'),
        'scalable_low_stretch_tree': ('scaffold_gnn.sparsifiers.scalable_low_stretch_tree', 'ScalableLowStretchTreeSparsifier'),
        'fast_low_stretch_tree': ('scaffold_gnn.sparsifiers.scalable_low_stretch_tree', 'ScalableLowStretchTreeSparsifier'),
        'randspt': ('scaffold_gnn.sparsifiers.random_shortest_path_tree', 'RandomShortestPathTreeSparsifier'),
        'random_shortest_path_tree': ('scaffold_gnn.sparsifiers.random_shortest_path_tree', 'RandomShortestPathTreeSparsifier'),
        'llst': ('scaffold_gnn.sparsifiers.local_search_low_stretch_tree', 'LocalSearchLowStretchTreeSparsifier'),
        'local_search_low_stretch_tree': ('scaffold_gnn.sparsifiers.local_search_low_stretch_tree', 'LocalSearchLowStretchTreeSparsifier'),
        'joint_dilation_congestion': ('scaffold_gnn.sparsifiers.joint_dilation_congestion', 'JointDilationCongestionSparsifier'),
        'clustered_joint_dilation_congestion': ('scaffold_gnn.sparsifiers.clustered_joint_dilation_congestion', 'ClusteredLazyJointDilationCongestionSparsifier'),
        'parallel_clustered_joint_dilation_congestion': ('scaffold_gnn.sparsifiers.scaffold', 'ParallelClusteredJointDilationCongestionSparsifier'),
        'scaffold_heap': ('scaffold_gnn.sparsifiers.scaffold.scaffold_heap', 'ScaffoldHeapSparsifier'),
        'scaffold_fast': ('scaffold_gnn.sparsifiers.scaffold.scaffold_fast', 'ScaffoldFastSparsifier'),
        'scaffold_batch': ('scaffold_gnn.sparsifiers.scaffold.scaffold_batch', 'ScaffoldBatchSparsifier'),
        'scaffold_greedy': ('scaffold_gnn.sparsifiers.scaffold.scaffold_greedy', 'ScaffoldGreedySparsifier'),
        'scaffold_sample': ('scaffold_gnn.sparsifiers.scaffold.scaffold_sample', 'ScaffoldSampleSparsifier'),
        'support_cpp': ('scaffold_gnn.sparsifiers.support_graph_cpp', 'SupportGraphCppSparsifier'),
        'full': ('scaffold_gnn.sparsifiers.full_graph', 'FullGraph'),
        'fixed_support': ('scaffold_gnn.sparsifiers.fixed_support', 'FixedSupportSparsifier'),
        'support_sequence': ('scaffold_gnn.sparsifiers.support_sequence', 'SupportSequenceSparsifier'),
        'local_degree': ('scaffold_gnn.sparsifiers.local_degree', 'LocalDegreeSparsifier'),
        'rank_degree': ('scaffold_gnn.sparsifiers.rank_degree', 'RankDegreeSparsifier'),
        'gspar': ('scaffold_gnn.sparsifiers.g_spar', 'GSPAR'),
        'lspar': ('scaffold_gnn.sparsifiers.l_spar', 'LSPAR'),
        'lsim': ('scaffold_gnn.sparsifiers.l_sim', 'LSIM'),
        'forest_fire': ('scaffold_gnn.sparsifiers.forest_fire', 'ForestFireSparsifier'),
        'scan': ('scaffold_gnn.sparsifiers.scan', 'SCANSparsifier'),
        'cut': ('scaffold_gnn.sparsifiers.cut_sparsifier', 'CutSparsifier'),
        'spanning_forest': ('scaffold_gnn.sparsifiers.mst', 'MSTSparsifier'),
    }
    if name not in mapping:
        raise ValueError(f'Unknown sparsifier: {name}. Available: {list(mapping.keys())}')
    from importlib import import_module

    module_name, class_name = mapping[name]
    module = import_module(module_name)
    return getattr(module, class_name)(**kwargs)
