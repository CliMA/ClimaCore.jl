#=
julia --project
using Revise; include(joinpath("test", "DataLayouts", "benchmark_fill.jl"))
=#
using Test
using ClimaCore
using ClimaCore.DataLayouts
using BenchmarkTools
import ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends

if ClimaComms.device() isa ClimaComms.CUDADevice
    import CUDA
    device_name = CUDA.name(CUDA.device()) # Move to ClimaComms
else
    device_name = "CPU"
end

include(joinpath(pkgdir(ClimaCore), "benchmarks/scripts/benchmark_utils.jl"))

function benchmarkfill!(bm, device, data, val)
    caller = string(nameof(typeof(data)))
    @info "Benchmarking $caller..."
    trial = @benchmark ClimaComms.@cuda_sync $device fill!($data, $val)
    t_min = minimum(trial.times) * 1e-9 # to seconds
    nreps = length(trial.times)
    n_reads_writes = DataLayouts.ncomponents(data)
    push_info(
        bm;
        kernel_time_s = t_min,
        nreps = nreps,
        caller,
        problem_size = DataLayouts.array_size(data),
        n_reads_writes,
    )
end

@testset "fill! with Nf = 1" begin
    device = ClimaComms.device()
    ArrayType = ClimaComms.array_type(device)
    FT = Float64
    S = FT
    Nv = 63
    Ni = Nij = 4
    Nh = 30 * 30 * 6
    Nk = 6
    bm = Benchmark(; float_type = FT, device_name)
    data = DataF{S}(ArrayType{FT}, zeros)
    benchmarkfill!(bm, device, data, 3)
    @test all(parent(data) .== 3)
    data = IJFH{S}(ArrayType{FT}, zeros; Nij, Nh)
    benchmarkfill!(bm, device, data, 3)
    @test all(parent(data) .== 3)
    data = IJHF{S}(ArrayType{FT}, zeros; Nij, Nh)
    benchmarkfill!(bm, device, data, 3)
    @test all(parent(data) .== 3)
    data = IFH{S}(ArrayType{FT}, zeros; Ni, Nh)
    benchmarkfill!(bm, device, data, 3)
    @test all(parent(data) .== 3)
    data = IHF{S}(ArrayType{FT}, zeros; Ni, Nh)
    benchmarkfill!(bm, device, data, 3)
    @test all(parent(data) .== 3)
    data = IJF{S}(ArrayType{FT}, zeros; Nij)
    benchmarkfill!(bm, device, data, 3)
    @test all(parent(data) .== 3)
    data = IF{S}(ArrayType{FT}, zeros; Ni)
    benchmarkfill!(bm, device, data, 3)
    @test all(parent(data) .== 3)
    data = VF{S}(ArrayType{FT}, zeros; Nv)
    benchmarkfill!(bm, device, data, 3)
    @test all(parent(data) .== 3)
    data = VIJFH{S}(ArrayType{FT}, zeros; Nv, Nij, Nh)
    benchmarkfill!(bm, device, data, 3)
    @test all(parent(data) .== 3)
    data = VIJHF{S}(ArrayType{FT}, zeros; Nv, Nij, Nh)
    benchmarkfill!(bm, device, data, 3)
    @test all(parent(data) .== 3)
    data = VIFH{S}(ArrayType{FT}, zeros; Nv, Ni, Nh)
    benchmarkfill!(bm, device, data, 3)
    @test all(parent(data) .== 3)
    data = VIHF{S}(ArrayType{FT}, zeros; Nv, Ni, Nh)
    benchmarkfill!(bm, device, data, 3)
    @test all(parent(data) .== 3)

    # data = DataLayouts.IJKFVH{S}(ArrayType{FT}, zeros; Nij,Nk,Nv,Nh); benchmarkfill!(bm, device, data, 3); @test all(parent(data) .== 3) # TODO: test
    # data = DataLayouts.IH1JH2{S}(ArrayType{FT}, zeros; Nij);          benchmarkfill!(bm, device, data, 3); @test all(parent(data) .== 3) # TODO: test
    tabulate_benchmark(bm)
end
