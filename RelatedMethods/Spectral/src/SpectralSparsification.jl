module SpectralSparsification

using Dates
using Laplacians
using LinearAlgebra
using NPZ
using Printf
using Random
using SparseArrays
using TOML

export GraphData,
       compute_effective_resistance,
       load_effective_resistance,
       load_graph,
       precompute_effective_resistance,
       run_pipeline,
       sample_sparsifier

const SCHEMA_VERSION = 5

"""A canonical, loop-free, undirected weighted graph with 1-based endpoints."""
struct GraphData
    num_nodes::Int
    src::Vector{Int}
    dst::Vector{Int}
    weight::Vector{Float64}
    fingerprint::String
    dataset::String
    directed_to_undirected::Vector{Int32}
    directed_fingerprint::String
end

GraphData(
    num_nodes::Int,
    src::Vector{Int},
    dst::Vector{Int},
    weight::Vector{Float64},
    fingerprint::String,
    dataset::String,
) = GraphData(
    num_nodes,
    src,
    dst,
    weight,
    fingerprint,
    dataset,
    Int32.(collect(eachindex(src))),
    fingerprint,
)

function _decode_bytes(value)
    return String(vec(reinterpret(UInt8, value)))
end

function load_graph(path::AbstractString)::GraphData
    values = npzread(path)
    required = (
        "num_nodes",
        "src",
        "dst",
        "weight",
        "graph_fingerprint",
        "directed_fingerprint",
        "directed_to_undirected",
        "dataset",
    )
    missing = filter(key -> !haskey(values, key), required)
    isempty(missing) || error("input graph is missing arrays: $(join(missing, ", "))")

    n = Int(first(values["num_nodes"]))
    src = Int.(vec(values["src"])) .+ 1
    dst = Int.(vec(values["dst"])) .+ 1
    weight = Float64.(vec(values["weight"]))
    fingerprint = _decode_bytes(values["graph_fingerprint"])
    directed_fingerprint = _decode_bytes(values["directed_fingerprint"])
    dataset = _decode_bytes(values["dataset"])
    raw_mapping = Int32.(vec(values["directed_to_undirected"]))
    directed_to_undirected = raw_mapping .+ Int32(1)

    length(src) == length(dst) == length(weight) || error("edge arrays have different lengths")
    n > 0 || error("num_nodes must be positive")
    isempty(src) && error("the graph has no usable undirected edges")
    all((1 .<= src) .& (src .< dst) .& (dst .<= n)) ||
        error("edges must be canonical 0-based pairs with src < dst before Julia conversion")
    all(isfinite, weight) || error("edge weights contain NaN or Inf")
    all(>(0.0), weight) || error("spectral conductances must be strictly positive")
    all(index -> 0 <= index <= length(src), directed_to_undirected) ||
        error("directed-to-undirected mapping contains an invalid edge index")

    return GraphData(
        n,
        src,
        dst,
        weight,
        fingerprint,
        dataset,
        directed_to_undirected,
        directed_fingerprint,
    )
end

function _adjacency(graph::GraphData)
    return sparse(
        vcat(graph.src, graph.dst),
        vcat(graph.dst, graph.src),
        vcat(graph.weight, graph.weight),
        graph.num_nodes,
        graph.num_nodes,
    )
end

function _component_count(graph::GraphData)
    parent = collect(1:graph.num_nodes)
    rank = zeros(UInt8, graph.num_nodes)

    function root(value::Int)
        while parent[value] != value
            parent[value] = parent[parent[value]]
            value = parent[value]
        end
        return value
    end

    for edge in eachindex(graph.src)
        left = root(graph.src[edge])
        right = root(graph.dst[edge])
        left == right && continue
        if rank[left] < rank[right]
            parent[left] = right
        elseif rank[left] > rank[right]
            parent[right] = left
        else
            parent[right] = left
            rank[left] += 1
        end
    end
    return length(Set(root(node) for node in 1:graph.num_nodes))
end

function _approximate_resistance(
    graph::GraphData;
    jl_factor::Float64,
    solver_tolerance::Float64,
    seed::Int,
    progress_every::Int,
)
    jl_factor > 0 || error("jl_factor must be positive")
    solver_tolerance > 0 || error("solver_tolerance must be positive")
    progress_every > 0 || error("progress_every must be positive")

    n = graph.num_nodes
    m = length(graph.src)
    dimensions = max(1, ceil(Int, jl_factor * log(max(n, 2))))
    adjacency = SparseMatrixCSC{Float64, Int}(_adjacency(graph))

    # Laplacians.jl's approximate elimination can use Julia's default RNG.
    # Seed it as well as the projection RNG so --force-er is reproducible.
    Random.seed!(seed)
    println("[$(now(UTC))] Building approximate-Cholesky Laplacian solver")
    flush(stdout)
    factor_seconds = @elapsed solver = approxchol_lap(adjacency, tol=solver_tolerance)
    @printf("[%s] Solver ready after %.3f seconds\n", string(now(UTC)), factor_seconds)
    flush(stdout)
    rng = MersenneTwister(seed)
    scale = inv(sqrt(dimensions))
    sqrt_weight = sqrt.(graph.weight)
    resistance = zeros(Float64, m)
    rhs = zeros(Float64, n)
    use_threaded_edges = Threads.nthreads() > 1 && m >= 5_000
    edge_threads = Threads.nthreads()
    rhs_parts = use_threaded_edges ? zeros(Float64, n, edge_threads) : zeros(Float64, 0, 0)
    projection_started = time()

    solve_seconds = @elapsed begin
        for projection in 1:dimensions
            fill!(rhs, 0.0)
            noise = randn(rng, m)
            if use_threaded_edges
                fill!(rhs_parts, 0.0)
                Threads.@threads :static for slot in 1:edge_threads
                    first_edge = div((slot - 1) * m, edge_threads) + 1
                    last_edge = div(slot * m, edge_threads)
                    @inbounds for edge in first_edge:last_edge
                        value = scale * sqrt_weight[edge] * noise[edge]
                        rhs_parts[graph.src[edge], slot] += value
                        rhs_parts[graph.dst[edge], slot] -= value
                    end
                end
                Threads.@threads for node in 1:n
                    value = 0.0
                    @inbounds for slot in 1:edge_threads
                        value += rhs_parts[node, slot]
                    end
                    rhs[node] = value
                end
            else
                @inbounds for edge in 1:m
                    value = scale * sqrt_weight[edge] * noise[edge]
                    rhs[graph.src[edge]] += value
                    rhs[graph.dst[edge]] -= value
                end
            end

            solution = solver(rhs)
            if use_threaded_edges
                Threads.@threads for edge in 1:m
                    difference = solution[graph.src[edge]] - solution[graph.dst[edge]]
                    resistance[edge] += difference * difference
                end
            else
                @inbounds for edge in 1:m
                    difference = solution[graph.src[edge]] - solution[graph.dst[edge]]
                    resistance[edge] += difference * difference
                end
            end

            if projection == 1 || projection == dimensions || projection % progress_every == 0
                @printf(
                    "[%s] JL projection %d/%d complete (elapsed %.1f seconds)\n",
                    string(now(UTC)),
                    projection,
                    dimensions,
                    time() - projection_started,
                )
                flush(stdout)
            end
        end
    end

    resistance .= max.(resistance, 0.0)
    return resistance, dimensions, factor_seconds, solve_seconds
end

function _exact_resistance(graph::GraphData; exact_max_nodes::Int)
    graph.num_nodes <= exact_max_nodes || error(
        "exact mode forms a dense Laplacian pseudoinverse; " *
        "$(graph.dataset) has $(graph.num_nodes) nodes, above --exact-max-nodes=$exact_max_nodes",
    )
    adjacency = _adjacency(graph)
    laplacian = Matrix(spdiagm(0 => vec(sum(adjacency, dims=2))) - adjacency)
    inverse_seconds = @elapsed laplacian_inverse = pinv(laplacian; rtol=sqrt(eps(Float64)))
    resistance = Vector{Float64}(undef, length(graph.src))
    edge_seconds = @elapsed begin
        @inbounds for edge in eachindex(graph.src)
            u = graph.src[edge]
            v = graph.dst[edge]
            resistance[edge] = max(
                0.0,
                laplacian_inverse[u, u] + laplacian_inverse[v, v] -
                2.0 * laplacian_inverse[u, v],
            )
        end
    end
    return resistance, 0, inverse_seconds, edge_seconds
end

"""
Compute edge-aligned effective resistance.

`method="approx"` implements the JL + approximate-Cholesky method used by the
original `compute_V_julia.jl`, but accumulates edge resistances directly and
does not materialize its `m × k` Gaussian matrix.
"""
function compute_effective_resistance(
    graph::GraphData;
    method::String="approx",
    jl_factor::Float64=4.0,
    solver_tolerance::Float64=1e-2,
    seed::Int=42,
    exact_max_nodes::Int=5_000,
    progress_every::Int=1,
)
    if method == "approx"
        return _approximate_resistance(
            graph;
            jl_factor=jl_factor,
            solver_tolerance=solver_tolerance,
            seed=seed,
            progress_every=progress_every,
        )
    elseif method == "exact"
        return _exact_resistance(graph; exact_max_nodes=exact_max_nodes)
    end
    error("method must be 'approx' or 'exact'")
end

function _alias_table(probabilities::Vector{Float64})
    count = length(probabilities)
    scaled = probabilities .* count
    threshold = zeros(Float64, count)
    alias = collect(1:count)
    small = Int[]
    large = Int[]
    sizehint!(small, count)
    sizehint!(large, count)

    for index in eachindex(scaled)
        push!(scaled[index] < 1.0 ? small : large, index)
    end
    while !isempty(small) && !isempty(large)
        low = pop!(small)
        high = pop!(large)
        threshold[low] = scaled[low]
        alias[low] = high
        scaled[high] -= 1.0 - scaled[low]
        push!(scaled[high] < 1.0 ? small : large, high)
    end
    for index in small
        threshold[index] = 1.0
    end
    for index in large
        threshold[index] = 1.0
    end
    return threshold, alias
end

"""Sample `budget` edges with replacement and aggregate duplicate draws."""
function sample_sparsifier(
    graph::GraphData,
    resistance::Vector{Float64};
    budget::Int,
    seed::Int=42,
)
    m = length(graph.src)
    length(resistance) == m || error("resistance array is not aligned with graph edges")
    budget > 0 || error("budget must be positive")

    leverage = max.(graph.weight .* resistance, 0.0)
    leverage_sum = sum(leverage)
    probabilities = if isfinite(leverage_sum) && leverage_sum > 0.0
        leverage ./ leverage_sum
    else
        @warn "all leverage scores were zero; using uniform edge probabilities"
        fill(inv(Float64(m)), m)
    end

    threshold, alias = _alias_table(probabilities)
    counts = zeros(Int64, m)
    rng = MersenneTwister(seed)
    @inbounds for _ in 1:budget
        candidate = rand(rng, 1:m)
        selected = rand(rng) <= threshold[candidate] ? candidate : alias[candidate]
        counts[selected] += 1
    end

    selected = findall(>(0), counts)
    new_weight = Vector{Float64}(undef, length(selected))
    @inbounds for (output_index, edge) in enumerate(selected)
        new_weight[output_index] =
            counts[edge] * graph.weight[edge] / (budget * probabilities[edge])
    end
    return (
        selected=selected,
        src=graph.src[selected],
        dst=graph.dst[selected],
        edge_weight=new_weight,
        sample_count=counts[selected],
        selection_probability=probabilities[selected],
        leverage=leverage,
        leverage_sum=leverage_sum,
    )
end

function _write_toml(path::AbstractString, values::Dict{String, Any})
    temporary = joinpath(
        dirname(path),
        ".$(basename(path)).$(randstring(10)).tmp",
    )
    open(temporary, "w") do output
        TOML.print(output, values; sorted=true)
        write(output, '\n')
    end
    mv(temporary, path; force=true)
end

function _write_npz(path::AbstractString, values::Dict{String, Any})
    temporary = joinpath(
        dirname(path),
        ".$(splitext(basename(path))[1]).$(randstring(10)).npz",
    )
    npzwrite(temporary, values)
    mv(temporary, path; force=true)
end

function _byte_array(value::AbstractString)
    return collect(codeunits(value))
end

function _cache_matches(path::AbstractString, expected::Dict{String, Any})
    isfile(path) || return false
    actual = try
        TOML.parsefile(path)
    catch
        return false
    end
    return all(get(actual, key, nothing) == value for (key, value) in expected)
end

function load_effective_resistance(path::AbstractString, graph::GraphData)
    values = npzread(path)
    fingerprint = _decode_bytes(values["graph_fingerprint"])
    fingerprint == graph.fingerprint || error("effective-resistance cache belongs to another graph")
    resistance = Float64.(vec(values["resistance"]))
    length(resistance) == length(graph.src) || error("effective-resistance cache has wrong edge count")
    return resistance
end

function _effective_cache_settings(
    graph::GraphData,
    method::String,
    jl_factor::Float64,
    solver_tolerance::Float64,
    seed::Int,
    exact_max_nodes::Int,
)
    return Dict{String, Any}(
        "schema_version" => SCHEMA_VERSION,
        "dataset" => graph.dataset,
        "graph_fingerprint" => graph.fingerprint,
        "method" => method,
        "jl_factor" => jl_factor,
        "solver_tolerance" => solver_tolerance,
        "seed" => seed,
        "exact_max_nodes" => exact_max_nodes,
        "julia_threads" => Threads.nthreads(),
    )
end

function _ensure_directed_resistance(
    graph::GraphData,
    resistance::Vector{Float64},
    output_dir::AbstractString,
    er_settings::Dict{String, Any};
    er_compute_seconds::Float64,
    force::Bool=false,
)
    path = joinpath(output_dir, "effective_resistance_directed.npz")
    meta_path = joinpath(output_dir, "effective_resistance_directed.toml")
    settings = merge(
        copy(er_settings),
        Dict{String, Any}(
            "directed_fingerprint" => graph.directed_fingerprint,
            "num_directed_edges" => length(graph.directed_to_undirected),
        ),
    )
    cached = !force && isfile(path) && _cache_matches(meta_path, settings)
    if cached
        println("Directed ER cache hit: $path")
        return path, true, Float64(get(TOML.parsefile(meta_path), "write_seconds", 0.0))
    end

    println(
        "[$(now(UTC))] Materializing $(length(graph.directed_to_undirected)) " *
        "ER values in original Benchmark edge order",
    )
    flush(stdout)
    directed = zeros(Float64, length(graph.directed_to_undirected))
    materialize_seconds = @elapsed begin
        if Threads.nthreads() > 1 && length(directed) >= 5_000
            Threads.@threads for edge in eachindex(directed)
                canonical = graph.directed_to_undirected[edge]
                directed[edge] = canonical == 0 ? 0.0 : resistance[Int(canonical)]
            end
        else
            @inbounds for edge in eachindex(directed)
                canonical = graph.directed_to_undirected[edge]
                directed[edge] = canonical == 0 ? 0.0 : resistance[Int(canonical)]
            end
        end
    end
    write_seconds = @elapsed _write_npz(
        path,
        Dict{String, Any}(
            "resistance" => directed,
            "num_directed_edges" => Int64[length(directed)],
            "graph_fingerprint" => _byte_array(graph.fingerprint),
            "directed_fingerprint" => _byte_array(graph.directed_fingerprint),
            "dataset" => _byte_array(graph.dataset),
        ),
    )
    metadata = merge(
        settings,
        Dict{String, Any}(
            "created_at_utc" => string(now(UTC)),
            "edge_order" => "Benchmark data.edge_index columns",
            "self_loop_resistance" => 0.0,
            "materialize_seconds" => materialize_seconds,
            "write_seconds" => write_seconds,
            "compute_seconds" => er_compute_seconds,
        ),
    )
    _write_toml(meta_path, metadata)
    @printf(
        "[%s] Directed ER artifact ready: %d values (materialize %.3fs, write %.3fs)\n",
        string(now(UTC)),
        length(directed),
        materialize_seconds,
        write_seconds,
    )
    flush(stdout)
    return path, false, write_seconds
end

function _budget_filename(budget::Int, seed::Int)
    return "sparsified_budget_$(budget)_seed_$(seed).npz"
end

"""Compute/cache effective resistance and optionally produce one sparsifier."""
function run_pipeline(
    input_path::AbstractString,
    output_dir::AbstractString;
    budget::Union{Nothing, Int}=nothing,
    method::String="approx",
    jl_factor::Float64=4.0,
    solver_tolerance::Float64=1e-2,
    er_seed::Int=42,
    sample_seed::Int=42,
    exact_max_nodes::Int=5_000,
    force_er::Bool=false,
    force_sparsifier::Bool=false,
    progress_every::Int=1,
)
    mkpath(output_dir)
    graph = load_graph(input_path)
    er_path = joinpath(output_dir, "effective_resistance.npz")
    er_meta_path = joinpath(output_dir, "effective_resistance.toml")
    er_settings = _effective_cache_settings(
        graph,
        method,
        jl_factor,
        solver_tolerance,
        er_seed,
        exact_max_nodes,
    )

    er_cached = !force_er && isfile(er_path) && _cache_matches(er_meta_path, er_settings)
    er_seconds = 0.0
    factor_seconds = 0.0
    solve_seconds = 0.0
    dimensions = method == "approx" ? max(1, ceil(Int, jl_factor * log(max(graph.num_nodes, 2)))) : 0

    if er_cached
        println("Effective resistance cache hit: $er_path")
        resistance = load_effective_resistance(er_path, graph)
        cached_meta = TOML.parsefile(er_meta_path)
        er_seconds = Float64(get(cached_meta, "compute_seconds", 0.0))
        factor_seconds = Float64(get(cached_meta, "factor_seconds", 0.0))
        solve_seconds = Float64(get(cached_meta, "solve_seconds", 0.0))
    else
        println("Computing effective resistance: method=$method nodes=$(graph.num_nodes) edges=$(length(graph.src))")
        er_seconds = @elapsed begin
            resistance, dimensions, factor_seconds, solve_seconds = compute_effective_resistance(
                graph;
                method=method,
                jl_factor=jl_factor,
                solver_tolerance=solver_tolerance,
                seed=er_seed,
                exact_max_nodes=exact_max_nodes,
                progress_every=progress_every,
            )
        end
        leverage = graph.weight .* resistance
        _write_npz(
            er_path,
            Dict{String, Any}(
                "src" => Int64.(graph.src .- 1),
                "dst" => Int64.(graph.dst .- 1),
                "conductance" => graph.weight,
                "resistance" => resistance,
                "leverage_score" => leverage,
                "graph_fingerprint" => _byte_array(graph.fingerprint),
            ),
        )
        er_metadata = merge(
            copy(er_settings),
            Dict{String, Any}(
                "created_at_utc" => string(now(UTC)),
                "julia_version" => string(VERSION),
                "num_nodes" => graph.num_nodes,
                "num_undirected_edges" => length(graph.src),
                "jl_dimensions" => dimensions,
                "compute_seconds" => er_seconds,
                "factor_seconds" => factor_seconds,
                "solve_seconds" => solve_seconds,
                "component_count" => _component_count(graph),
                "leverage_sum" => sum(leverage),
            ),
        )
        _write_toml(er_meta_path, er_metadata)
        @printf("Effective resistance completed in %.6f seconds\n", er_seconds)
    end

    directed_path, directed_cached, directed_write_seconds = _ensure_directed_resistance(
        graph,
        resistance,
        output_dir,
        er_settings;
        er_compute_seconds=er_seconds,
        force=force_er,
    )

    if isnothing(budget)
        println("RESULT dataset=$(graph.dataset)")
        println("RESULT er_cached=$er_cached")
        println("RESULT directed_er_cached=$directed_cached")
        @printf("RESULT er_seconds=%.9f\n", er_seconds)
        @printf("RESULT directed_write_seconds=%.9f\n", directed_write_seconds)
        println("RESULT input_undirected_edges=$(length(graph.src))")
        println("RESULT input_directed_edges=$(length(graph.directed_to_undirected))")
        println("RESULT effective_resistance_path=$er_path")
        println("RESULT directed_effective_resistance_path=$directed_path")
        flush(stdout)
        return (
            graph=graph,
            er_cached=er_cached,
            directed_er_cached=directed_cached,
            er_seconds=er_seconds,
            effective_resistance_path=er_path,
            directed_effective_resistance_path=directed_path,
        )
    end

    resolved_budget = something(budget)

    sparsifier_name = _budget_filename(resolved_budget, sample_seed)
    sparsifier_path = joinpath(output_dir, sparsifier_name)
    sparsifier_meta_path = replace(sparsifier_path, r"\.npz$" => ".toml")
    sparsifier_settings = Dict{String, Any}(
        "schema_version" => SCHEMA_VERSION,
        "dataset" => graph.dataset,
        "graph_fingerprint" => graph.fingerprint,
        "effective_resistance_method" => method,
        "effective_resistance_jl_factor" => jl_factor,
        "effective_resistance_solver_tolerance" => solver_tolerance,
        "effective_resistance_seed" => er_seed,
        "budget" => resolved_budget,
        "sample_seed" => sample_seed,
    )
    sparsifier_cached =
        !force_sparsifier && isfile(sparsifier_path) &&
        _cache_matches(sparsifier_meta_path, sparsifier_settings)
    sparsify_seconds = 0.0
    output_edges = 0

    if sparsifier_cached
        println("Sparsifier cache hit: $sparsifier_path")
        cached_meta = TOML.parsefile(sparsifier_meta_path)
        sparsify_seconds = Float64(get(cached_meta, "compute_seconds", 0.0))
        output_edges = Int(get(cached_meta, "num_output_undirected_edges", 0))
    else
        println("Sampling spectral sparsifier: budget=$resolved_budget seed=$sample_seed")
        sparsify_seconds = @elapsed sampled = sample_sparsifier(
            graph,
            resistance;
            budget=resolved_budget,
            seed=sample_seed,
        )
        output_edges = length(sampled.src)
        _write_npz(
            sparsifier_path,
            Dict{String, Any}(
                "num_nodes" => Int64[graph.num_nodes],
                "src" => Int64.(sampled.src .- 1),
                "dst" => Int64.(sampled.dst .- 1),
                "edge_weight" => sampled.edge_weight,
                "sample_count" => sampled.sample_count,
                "selection_probability" => sampled.selection_probability,
                "original_edge_index" => Int64.(sampled.selected .- 1),
                "graph_fingerprint" => _byte_array(graph.fingerprint),
                "dataset" => _byte_array(graph.dataset),
            ),
        )
        merge!(
            sparsifier_settings,
            Dict{String, Any}(
                "created_at_utc" => string(now(UTC)),
                "compute_seconds" => sparsify_seconds,
                "num_input_undirected_edges" => length(graph.src),
                "num_output_undirected_edges" => output_edges,
                "num_output_directed_edges" => 2 * output_edges,
                "sampling" => "independent_with_replacement_then_aggregate",
            ),
        )
        _write_toml(sparsifier_meta_path, sparsifier_settings)
        @printf("Sparsification completed in %.6f seconds (%d unique edges)\n", sparsify_seconds, output_edges)
    end

    println("RESULT dataset=$(graph.dataset)")
    println("RESULT er_cached=$er_cached")
    @printf("RESULT er_seconds=%.9f\n", er_seconds)
    @printf("RESULT sparsify_seconds=%.9f\n", sparsify_seconds)
    println("RESULT input_undirected_edges=$(length(graph.src))")
    println("RESULT output_undirected_edges=$output_edges")
    println("RESULT effective_resistance_path=$er_path")
    println("RESULT directed_effective_resistance_path=$directed_path")
    println("RESULT sparsifier_path=$sparsifier_path")

    return (
        graph=graph,
        er_cached=er_cached,
        er_seconds=er_seconds,
        sparsify_seconds=sparsify_seconds,
        output_edges=output_edges,
        effective_resistance_path=er_path,
        directed_effective_resistance_path=directed_path,
        sparsifier_path=sparsifier_path,
    )
end

precompute_effective_resistance(input_path::AbstractString, output_dir::AbstractString; kwargs...) =
    run_pipeline(input_path, output_dir; budget=nothing, kwargs...)

end
