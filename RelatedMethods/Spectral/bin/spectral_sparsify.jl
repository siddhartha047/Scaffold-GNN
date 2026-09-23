#!/usr/bin/env julia

using ArgParse
using SpectralSparsification

function settings()
    options = ArgParseSettings(description="Compute effective resistance and a budgeted spectral sparsifier")
    @add_arg_table! options begin
        "--input"
            help = "Canonical input_graph.npz exported by export_benchmark_graph.py"
            required = true
        "--output-dir"
            help = "Directory for effective-resistance and sparsifier artifacts"
            required = true
        "--budget"
            help = "Number of independent edge draws (duplicates are aggregated)"
            arg_type = Int
            required = true
        "--method"
            help = "Effective-resistance method: approx or exact"
            default = "approx"
        "--jl-factor"
            help = "JL dimension multiplier; dimensions = ceil(factor * log(n))"
            arg_type = Float64
            default = 4.0
        "--solver-tolerance"
            help = "Approximate-Cholesky Laplacian solver tolerance"
            arg_type = Float64
            default = 1e-2
        "--er-seed"
            help = "Random seed for JL effective resistance"
            arg_type = Int
            default = 42
        "--sample-seed"
            help = "Random seed for sparsifier edge sampling"
            arg_type = Int
            default = 42
        "--exact-max-nodes"
            help = "Safety limit for dense exact mode"
            arg_type = Int
            default = 5000
        "--force-er"
            help = "Ignore and replace a compatible effective-resistance cache"
            action = :store_true
        "--force-sparsifier"
            help = "Ignore and replace a compatible sparsifier cache"
            action = :store_true
    end
    return options
end

arguments = parse_args(settings())
run_pipeline(
    arguments["input"],
    arguments["output-dir"];
    budget=arguments["budget"],
    method=arguments["method"],
    jl_factor=arguments["jl-factor"],
    solver_tolerance=arguments["solver-tolerance"],
    er_seed=arguments["er-seed"],
    sample_seed=arguments["sample-seed"],
    exact_max_nodes=arguments["exact-max-nodes"],
    force_er=arguments["force-er"],
    force_sparsifier=arguments["force-sparsifier"],
)
