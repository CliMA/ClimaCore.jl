#=
julia --project=.buildkite
using Revise; include(joinpath("test", "DataLayouts", "benchmark_loops.jl"))
=#
using Test
using BenchmarkTools
import ClimaComms
import ClimaCore: ClimaCore, DataLayouts
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
if ClimaComms.device() isa ClimaComms.CUDADevice
    import CUDA
    device_name = CUDA.name(CUDA.device()) # Move to ClimaComms
else
    device_name = "CPU"
end

include(joinpath(pkgdir(ClimaCore), "benchmarks/scripts/benchmark_utils.jl"))

# Benchmarks for the point-loop machinery: pointwise broadcasts and reductions
# over full layouts, single-field property views (whose constant-stride parents
# have linear fast paths in array_and_index_args and view_point_struct), and
# lazy broadcast expressions with equal shapes (linear point indices) and mixed
# shapes (Cartesian point indices, projected by Broadcast.newindex). Pointwise
# kernels should be limited by memory bandwidth, so the distance between their
# tabulated throughput and the device's peak measures index-arithmetic
# overhead.

function benchmark_pointwise!(bm, device, caller, dest, arg1, arg2)
    @info "Benchmarking $caller..."
    trial = @benchmark ClimaComms.@cuda_sync $device ($dest .= $arg1 .+ 2 .* $arg2)
    kernel_time_s = minimum(trial.times) * 1e-9 # to seconds
    nreps = length(trial.times)
    problem_size = size(dest)
    n_reads_writes = 3
    push_info(bm; kernel_time_s, nreps, caller, problem_size, n_reads_writes)
end

function benchmark_reduce(bm, device, caller, arg)
    @info "Benchmarking $caller..."
    trial = @benchmark ClimaComms.@cuda_sync $device sum(identity, $arg)
    kernel_time_s = minimum(trial.times) * 1e-9 # to seconds
    nreps = length(trial.times)
    problem_size = size(arg)
    n_reads_writes = 1
    push_info(bm; kernel_time_s, nreps, caller, problem_size, n_reads_writes)
end

@testset "pointwise broadcasts and reductions over layouts and views" begin
    device = ClimaComms.device()
    FT = Float64
    A = ClimaComms.array_type(device){FT}
    T2 = @NamedTuple{a::FT, b::FT}
    bm = Benchmark(; float_type = FT, device_name)
    (Nv, Nij, Nh) = (63, 4, 30 * 30 * 6)

    # Full single-component layouts (linear point indices on GPUs).
    dest = DataLayouts.VIJFH{FT, Nv, Nij, Nij, nothing}(A, Nh)
    arg1 = DataLayouts.VIJFH{FT, Nv, Nij, Nij, nothing}(A, Nh)
    arg2 = DataLayouts.VIJFH{FT, Nv, Nij, Nij, nothing}(A, Nh)
    fill!(arg1, 2)
    fill!(arg2, 3)
    benchmark_pointwise!(bm, device, "1-component", dest, arg1, arg2)
    @test all(==(8), parent(dest))

    # Property views of two-component layouts (constant-stride linear paths).
    dest2 = DataLayouts.VIJFH{T2, Nv, Nij, Nij, nothing}(A, Nh)
    args2 = DataLayouts.VIJFH{T2, Nv, Nij, Nij, nothing}(A, Nh)
    fill!(dest2, (; a = FT(0), b = FT(-1)))
    fill!(args2, (; a = FT(2), b = FT(3)))
    benchmark_pointwise!(bm, device, "property views", dest2.a, args2.a, args2.b)
    @test all(==(8), Array(parent(dest2))[:, :, :, 1, :])
    @test all(==(-1), Array(parent(dest2))[:, :, :, 2, :]) # b is untouched

    # Reductions over data, views, and lazy expressions. The integer fill
    # values make the sums exact under any reduction order.
    benchmark_reduce(bm, device, "reduce 1-component", arg1)
    @test sum(identity, arg1) == 2 * length(arg1)
    benchmark_reduce(bm, device, "reduce property view", args2.a)
    @test sum(identity, args2.a) == 2 * length(args2.a)
    equal_bc = Base.broadcasted(+, arg1, arg2)
    benchmark_reduce(bm, device, "reduce equal-shape bc", equal_bc)
    @test sum(identity, equal_bc) == 5 * length(arg1)
    surface = DataLayouts.VIJFH{FT, 1, Nij, Nij, nothing}(A, Nh)
    fill!(surface, 7)
    mixed_bc = Base.broadcasted(+, arg1, surface)
    benchmark_reduce(bm, device, "reduce mixed-shape bc", mixed_bc)
    @test sum(identity, mixed_bc) == 9 * length(arg1)

    tabulate_benchmark(bm)
end
