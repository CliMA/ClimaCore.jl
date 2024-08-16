#=
julia --project
ENV["CLIMACOMMS_DEVICE"] = "CPU"
using Revise; include(joinpath("test", "DataLayouts", "opt_similar.jl"))
=#
using Test
using ClimaCore.DataLayouts
using ClimaCore: DataLayouts, Geometry
import ClimaComms
using StaticArrays: SMatrix
ClimaComms.@import_required_backends
using JET

function test_similar!(data)
    if data isa VF || data isa VIFH || data isa VIJFH
        FT = eltype(parent(data))
        CT = Geometry.ZPoint{FT}
        AIdx = (3,)
        LG = Geometry.LocalGeometry{AIdx, CT, FT, SMatrix{1, 1, FT, 1}}
        (_, _, _, Nv, _) = size(data)
        similar(data, LG, Val(Nv))
        @test_opt similar(data, LG, Val(Nv))
    else
        s = similar(data) # test callable
        @test_opt similar(data)
    end
end

@testset "similar" begin
    device = ClimaComms.device()
    device_zeros(args...) = ClimaComms.array_type(device)(zeros(args...))
    FT = Float64
    S = FT
    Nf = 1
    Nv = 4
    Nij = 3
    Nh = 5
    Nk = 6
    data = DataF{S}(device_zeros(FT, Nf))
    test_similar!(data)
    data = IJFH{S, Nij}(device_zeros(FT, Nij, Nij, Nf, Nh))
    test_similar!(data)
    data = IFH{S, Nij}(device_zeros(FT, Nij, Nf, Nh))
    test_similar!(data)
    data = IJF{S, Nij}(device_zeros(FT, Nij, Nij, Nf))
    test_similar!(data)
    data = IF{S, Nij}(device_zeros(FT, Nij, Nf))
    test_similar!(data)
    data = VF{S, Nv}(device_zeros(FT, Nv, Nf))
    test_similar!(data)
    data = VIJFH{S, Nv, Nij}(device_zeros(FT, Nv, Nij, Nij, Nf, Nh))
    test_similar!(data)
    data = VIFH{S, Nv, Nij}(device_zeros(FT, Nv, Nij, Nf, Nh))
    test_similar!(data)
    # data = DataLayouts.IJKFVH{S, Nij, Nk, Nv}(device_zeros(FT,Nij,Nij,Nk,Nf,Nv,Nh)); test_similar!(data) # TODO: test
    # data = DataLayouts.IH1JH2{S, Nij}(device_zeros(FT,2*Nij,3*Nij));                     test_similar!(data) # TODO: test
end
