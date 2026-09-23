#!/usr/bin/env julia

using ArgParse
using SpectralSparsification

function settings()
    options = ArgParseSettings(
        description="Precompute canonical and original-directed-order effective resistance",
    )
    @add_arg_table! options begin
        "--input"
            help = "Canonical input_graph.npz exported from Benchmark"
            required = true
        "--output-dir"
            help = "Dataset spectral_sparsification artifact directory"
            required = true
        "--method"
            help = "Effective-resistance method: approx or exact"
            default = "approx"
        "--jl-factor"
            arg_type = Float64
            default = 4.0
        "--solver-tolerance"
            arg_type = Float64
            default = 1e-2
        "--er-seed"
            arg_type = Int
            default = 42
        "--exact-max-nodes"
            arg_type = Int
            default = 5000
        "--progress-every"
            help = "Print progress after this many completed JL projections"
            arg_type = Int
            default = 1
        "--force-er"
            action = :store_true
    end
    return options
end

arguments = parse_args(settings())
precompute_effective_resistance(
    arguments["input"],
    arguments["output-dir"];
    method=arguments["method"],
    jl_factor=arguments["jl-factor"],
    solver_tolerance=arguments["solver-tolerance"],
    er_seed=arguments["er-seed"],
    exact_max_nodes=arguments["exact-max-nodes"],
    force_er=arguments["force-er"],
    progress_every=arguments["progress-every"],
)
