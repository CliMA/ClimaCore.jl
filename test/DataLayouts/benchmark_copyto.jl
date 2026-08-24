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

function benchmarkcopyto!(bm, device, data, val)
    caller = string(DataLayouts.layout_constructor(data))
    @info "Benchmarking $caller..."
    data_rhs = similar(data)
    fill!(data_rhs, val)
    bc = Base.Broadcast.broadcasted(identity, data_rhs)
    trial = @benchmark ClimaComms.@cuda_sync $device Base.copyto!($data, $bc)
    kernel_time_s = minimum(trial.times) * 1e-9 # to seconds
    nreps = length(trial.times)
    problem_size = size(data)
    n_reads_writes = DataLayouts.ncomponents(data) * 2
    push_info(bm; kernel_time_s, nreps, caller, problem_size, n_reads_writes)
end

@testset "copyto! with Nf = 1" begin
    device = ClimaComms.device()
    FT = Float64
    A = ClimaComms.array_type(device){FT}
    bm = Benchmark(; float_type = FT, device_name)

    data = DataLayouts.DataF{FT}(A)
    benchmarkcopyto!(bm, device, data, 3)
    @test all(parent(data) .== 3)

    (Nv, Nij, Nh) = (63, 4, 30 * 30 * 6)
    for Nv in (1, Nv), (Ni, Nj) in ((1, 1), (Nij, 1), (Nij, Nij)), Nh in (1, Nh)
        for D in (DataLayouts.VIJFH, DataLayouts.VIJHF)
            data = D{FT, Nv, Ni, Nj, Nh == 1 ? 1 : nothing}(A, Nh)
            benchmarkcopyto!(bm, device, data, 3)
            @test all(parent(data) .== 3)
        end
    end
    for Nv in (1, Nv), Ni in (1, Nij), Nh in (1, Nh)
        data = DataLayouts.VIH1{FT, Nv, Ni, Nh == 1 ? 1 : nothing}(A, Nh)
        benchmarkcopyto!(bm, device, data, 3)
        @test all(parent(data) .== 3)
    end
    for (Ni, Nj) in ((1, 1), (Nij, 1), (Nij, Nij)), Nh in (1, Nh)
        data = DataLayouts.IH1JH2{FT, Ni, Nj, Nh == 1 ? 1 : nothing}(A, Nh)
        benchmarkcopyto!(bm, device, data, 3)
        @test all(parent(data) .== 3)
    end

    tabulate_benchmark(bm)
end
