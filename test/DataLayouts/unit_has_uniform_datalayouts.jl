#=
julia --project
using Revise; include(joinpath("test", "DataLayouts", "unit_has_uniform_datalayouts.jl"))
=#
using Test
using ClimaCore.DataLayouts
import ClimaCore.Geometry
import ClimaComms
import ClimaCore: @lazy
using StaticArrays
import Random
Random.seed!(1234)

@testset "has_uniform_datalayouts" begin
    device = ClimaComms.device()
    ArrayType = ClimaComms.array_type(device)
    FT = Float64
    S = FT
    Nf = 1
    Nv = 4
    Ni = Nij = 3
    Nh = 5
    Nk = 6
    data_DataF = DataF{S}(ArrayType{FT}, zeros)
    data_IJFH = IJFH{S}(ArrayType{FT}, zeros; Nij, Nh)
    data_IFH = IFH{S}(ArrayType{FT}, zeros; Ni, Nh)
    data_IJF = IJF{S}(ArrayType{FT}, zeros; Nij)
    data_IF = IF{S}(ArrayType{FT}, zeros; Ni)
    data_VF = VF{S}(ArrayType{FT}, zeros; Nv)
    data_VIJFH = VIJFH{S}(ArrayType{FT}, zeros; Nv, Nij, Nh)
    data_VIFH = VIFH{S}(ArrayType{FT}, zeros; Nv, Ni, Nh)

    bc = @lazy @. data_VIFH + data_VIFH
    @test DataLayouts.has_uniform_datalayouts(bc)
    bc = @lazy @. data_IJFH + data_VF
    @test !DataLayouts.has_uniform_datalayouts(bc)

    data_VIJFHᶜ = VIJFH{S}(ArrayType{FT}, zeros; Nv, Nij, Nh)
    data_VIJFHᶠ = VIJFH{S}(ArrayType{FT}, zeros; Nv = Nv + 1, Nij, Nh)

    # This is not a valid broadcast expression,
    # but these two datalayouts can exist in a
    # valid broadcast expression (e.g., interpolation).
    bc = @lazy @. data_VIJFHᶜ + data_VIJFHᶠ
    @test DataLayouts.has_uniform_datalayouts(bc)
end
