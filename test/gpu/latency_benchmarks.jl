using ClimaCore
using ClimaCore.CommonSpaces
import ClimaComms
using Test
using CUDA
using BenchmarkTools
import LazyBroadcast: lazy


# the timings for these benchmark are taken using the central cluster
@testset "benchmarks time to kernel launch" begin
    # test to catch regressions and improvement to kernel launch time from ClimaCoreCUDAExt
    # after the inital compilation
    ext = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)
    @assert !isnothing(ext) # cuda must be loaded to test this extension
    space = ExtrudedCubedSphereSpace(Float32;
        z_elem = 10,
        z_min = 0,
        z_max = 1,
        radius = 10,
        h_elem = 10,
        n_quad_points = 4,
        staggering = CellCenter(),
    )

    scalar_field_1 = fill(1.0f0, space)
    scalar_field_2 = fill(1.0f0, space)
    # basic expression
    # intentionally benchmark without a sync between each trial
    # CUDA.synchronize()
    latency = median(@benchmark $scalar_field_1 .= $scalar_field_1 .+ $scalar_field_2).time
    # update this value if the kernel launch time changes significantly and it is expected
    baseline_latency = 20500
    @test latency ≈ baseline_latency atol = 3000
    percent_change_latency =
        round(Int, (latency - baseline_latency) / baseline_latency * 100)
    @info "Latency: $latency ns, Percent change from baseline: $percent_change_latency%"

    # repeated args expression
    CUDA.synchronize()
    latency =
        median(
            @benchmark $scalar_field_1 .=
                $scalar_field_1 .+ $scalar_field_2 .+ $scalar_field_1 .+ $scalar_field_2
        ).time
    # update this value if the kernel launch time changes significantly and it is expected
    baseline_latency = 22500
    @test latency ≈ baseline_latency atol = 3000
    percent_change_latency =
        round(Int, (latency - baseline_latency) / baseline_latency * 100)
    @info "Latency: $latency ns, Percent change from baseline: $percent_change_latency%"

    # nested lazy broadcast
    lazy_sum_1 = @. lazy(scalar_field_1 + scalar_field_2)
    lazy_sum_2 = @. lazy(lazy_sum_1 + lazy_sum_1)
    lazy_sum_3 = @. lazy(lazy_sum_2 + lazy_sum_2)
    CUDA.synchronize()
    latency = median(@benchmark $scalar_field_1 .= $lazy_sum_3).time
    # update this value if the kernel launch time changes significantly and it is expected
    baseline_latency = 29000
    @test latency ≈ baseline_latency atol = 3000
    percent_change_latency =
        round(Int, (latency - baseline_latency) / baseline_latency * 100)
    @info "Latency: $latency ns, Percent change from baseline: $percent_change_latency%"
end
