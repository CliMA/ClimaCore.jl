#=
julia --project
using Revise; include(joinpath("test", "DataLayouts", "benchmark_fill.jl"))
=#
using Test
using ClimaCore.DataLayouts
using BenchmarkTools
import ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends

function benchmarkfill!(device, data, val)
    trial = @benchmark ClimaComms.@cuda_sync $device fill!($data, $val)
    show(stdout, MIME("text/plain"), trial)
    println()
    trial =
        @benchmark ClimaComms.@cuda_sync $device fill!($(parent(data)), $val)
    show(stdout, MIME("text/plain"), trial)
    println()
end

@testset "fill! with Nf = 1" begin
    device = ClimaComms.device()
    device_zeros(args...) = ClimaComms.array_type(device)(zeros(args...))
    FT = Float64
    S = FT
    Nf = 1
    Nv = 63
    Nij = 4
    Nh = 30
    Nk = 6
#! format: off
    data = DataF{S}(device_zeros(FT,Nf));                        benchmarkfill!(device, data, 3); @test all(parent(data) .== 3)
    data = IJFH{S, Nij}(device_zeros(FT,Nij,Nij,Nf,Nh));         benchmarkfill!(device, data, 3); @test all(parent(data) .== 3)
    data = IFH{S, Nij}(device_zeros(FT,Nij,Nf,Nh));              benchmarkfill!(device, data, 3); @test all(parent(data) .== 3)
    data = IJF{S, Nij}(device_zeros(FT,Nij,Nij,Nf));             benchmarkfill!(device, data, 3); @test all(parent(data) .== 3)
    data = IF{S, Nij}(device_zeros(FT,Nij,Nf));                  benchmarkfill!(device, data, 3); @test all(parent(data) .== 3)
    data = VF{S, Nv}(device_zeros(FT,Nv,Nf));                    benchmarkfill!(device, data, 3); @test all(parent(data) .== 3)
    data = VIJFH{S, Nv, Nij}(device_zeros(FT,Nv,Nij,Nij,Nf,Nh)); benchmarkfill!(device, data, 3); @test all(parent(data) .== 3)
    data = VIFH{S, Nv, Nij}(device_zeros(FT,Nv,Nij,Nf,Nh));      benchmarkfill!(device, data, 3); @test all(parent(data) .== 3)
#! format: on

    # data = IJKFVH{S}(device_zeros(FT,Nij,Nij,Nk,Nf,Nh)); benchmarkfill!(device, data, 3); @test all(parent(data) .== 3) # TODO: test
    # data = IH1JH2{S}(device_zeros(FT,Nij,Nij,Nk,Nf,Nh)); benchmarkfill!(device, data, 3); @test all(parent(data) .== 3) # TODO: test
end
