#=
julia --project
ENV["CLIMACOMMS_DEVICE"] = "CPU"; using Revise; include(joinpath("test", "Fields", "benchmark_fieldvectors.jl"))
ENV["CLIMACOMMS_DEVICE"] = "CUDA"; using Revise; include(joinpath("test", "Fields", "benchmark_fieldvectors.jl"))
=#
using Test
using ClimaCore.DataLayouts
using ClimaCore: Spaces, Fields, Geometry
using BenchmarkTools
import ClimaComms
import ClimaCore
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
if ClimaComms.device() isa ClimaComms.CUDADevice
    import CUDA
    device_name = CUDA.name(CUDA.device()) # Move to ClimaComms
else
    device_name = "CPU"
end
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;


include(joinpath(pkgdir(ClimaCore), "benchmarks/scripts/benchmark_utils.jl"))

function benchmarkcopyto!(bm, device, data, val)
    caller = string(nameof(typeof(data)))
    @info "Benchmarking $caller..."
    data_rhs_1 = similar(data)
    data_rhs_2 = similar(data)
    data_rhs_3 = similar(data)
    data_rhs_4 = similar(data)
    T = eltype(parent(data))
    dt = T(0)
    bc1 = Base.Broadcast.broadcasted(+, data_rhs_2, data_rhs_3, data_rhs_4)
    bc2 = Base.Broadcast.broadcasted(*, dt, bc1)
    bc = Base.Broadcast.broadcasted(+, data_rhs_1, bc2)
    trial = @benchmark ClimaComms.@cuda_sync $device Base.copyto!($data, $bc)
    t_min = minimum(trial.times) * 1e-9 # to seconds
    nreps = length(trial.times)
    n_reads_writes = 1 + 4
    push_info(
        bm;
        kernel_time_s = t_min,
        nreps = nreps,
        caller,
        problem_size = size(parent(data)),
        n_reads_writes,
    )
end

function fv_state(cspace, fspace)
    FT = Spaces.undertype(cspace)
    return Fields.FieldVector(
        c = fill(
            (;
                ρ = FT(0),
                uₕ = zero(Geometry.Covariant12Vector{FT}),
                e_int = FT(0),
                q_tot = FT(0),
            ),
            cspace,
        ),
        f = fill((; w = Geometry.Covariant3Vector(FT(0))), fspace),
    )
end

@testset "FieldVector FH" begin
    FT = Float64
    device = ClimaComms.device()

    bm = Benchmark(; float_type = FT, device_name)
    cspace = TU.CenterExtrudedFiniteDifferenceSpace(
        FT;
        zelem = 63,
        helem = 30,
        Nq = 4,
        context = ClimaComms.context(device),
    )
    fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
    X = fv_state(cspace, fspace)
    benchmarkcopyto!(bm, device, X, 3)

    cspace = TU.CenterExtrudedFiniteDifferenceSpace(
        FT;
        zelem = 63,
        helem = 15,
        Nq = 4,
        context = ClimaComms.context(device),
    )
    fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
    X = fv_state(cspace, fspace)
    benchmarkcopyto!(bm, device, X, 3)

    tabulate_benchmark(bm)
    nothing
end

@testset "FieldVector HF" begin
    FT = Float64
    device = ClimaComms.device()

    bm = Benchmark(; float_type = FT, device_name)
    cspace = TU.CenterExtrudedFiniteDifferenceSpace(
        FT;
        zelem = 63,
        helem = 30,
        Nq = 4,
        context = ClimaComms.context(device),
        horizontal_layout_type = DataLayouts.IJHF,
    )
    fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
    X = fv_state(cspace, fspace)
    benchmarkcopyto!(bm, device, X, 3)

    cspace = TU.CenterExtrudedFiniteDifferenceSpace(
        FT;
        zelem = 63,
        helem = 15,
        Nq = 4,
        context = ClimaComms.context(device),
        horizontal_layout_type = DataLayouts.IJHF,
    )
    fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
    X = fv_state(cspace, fspace)
    benchmarkcopyto!(bm, device, X, 3)

    tabulate_benchmark(bm)
    nothing
end
