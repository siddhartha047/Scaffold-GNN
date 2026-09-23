using SpectralSparsification
using Test

@testset "effective resistance" begin
    triangle = GraphData(
        3,
        [1, 1, 2],
        [2, 3, 3],
        ones(3),
        "triangle",
        "triangle",
    )
    resistance, dimensions, _, _ = compute_effective_resistance(
        triangle;
        method="exact",
        exact_max_nodes=10,
    )
    @test dimensions == 0
    @test resistance ≈ fill(2 / 3, 3) atol=1e-10
    @test sum(triangle.weight .* resistance) ≈ 2.0 atol=1e-10
end

@testset "budgeted sampling" begin
    path_graph = GraphData(
        4,
        [1, 2, 3],
        [2, 3, 4],
        ones(3),
        "path",
        "path",
    )
    sampled = sample_sparsifier(path_graph, ones(3); budget=20, seed=7)
    @test sum(sampled.sample_count) == 20
    @test 1 <= length(sampled.src) <= 3
    @test all(sampled.edge_weight .> 0)
    @test all(sampled.src .< sampled.dst)
end
