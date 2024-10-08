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
    if data isa VF || data isa VIHF || data isa VIJHF
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
    ArrayType = ClimaComms.array_type(device)
    FT = Float64
    S = FT
    Nv = 4
    Ni = Nij = 3
    Nh = 5
    Nk = 6
    data = DataF{S}(ArrayType{FT}, zeros)
    test_similar!(data)
    data = IJHF{S}(ArrayType{FT}, zeros; Nij, Nh)
    test_similar!(data)
    data = IHF{S}(ArrayType{FT}, zeros; Ni, Nh)
    test_similar!(data)
    data = IJF{S}(ArrayType{FT}, zeros; Nij)
    test_similar!(data)
    data = IF{S}(ArrayType{FT}, zeros; Ni)
    test_similar!(data)
    data = VF{S}(ArrayType{FT}, zeros; Nv)
    test_similar!(data)
    data = VIJHF{S}(ArrayType{FT}, zeros; Nv, Nij, Nh)
    test_similar!(data)
    data = VIHF{S}(ArrayType{FT}, zeros; Nv, Ni, Nh)
    test_similar!(data)
    # data = DataLayouts.IJKFVH{S}(ArrayType{FT}, zeros; Nij,Nk,Nv,Nh); test_similar!(data) # TODO: test
    # data = DataLayouts.IH1JH2{S}(ArrayType{FT}, zeros; Nij);          test_similar!(data) # TODO: test
end
