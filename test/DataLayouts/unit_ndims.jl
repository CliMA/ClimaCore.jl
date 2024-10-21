#=
julia --project
using Revise; include(joinpath("test", "DataLayouts", "unit_ndims.jl"))
=#
using Test
using ClimaCore.DataLayouts
import ClimaComms
ClimaComms.@import_required_backends

@testset "Base.ndims" begin
    device = ClimaComms.device()
    ArrayType = ClimaComms.array_type(device)
    FT = Float64
    S = FT
    Nv = 4
    Ni = Nij = 3
    Nh = 5
    Nk = 6

    data = DataF{S}(ArrayType{FT}, zeros)
    @test ndims(data) == 1
    @test ndims(typeof(data)) == 1
    data = IF{S}(ArrayType{FT}, zeros; Ni)
    @test ndims(data) == 2
    @test ndims(typeof(data)) == 2
    data = VF{S}(ArrayType{FT}, zeros; Nv)
    @test ndims(data) == 2
    @test ndims(typeof(data)) == 2
    data = IFH{S}(ArrayType{FT}, zeros; Ni, Nh)
    @test ndims(data) == 3
    @test ndims(typeof(data)) == 3
    data = IHF{S}(ArrayType{FT}, zeros; Ni, Nh)
    @test ndims(data) == 3
    @test ndims(typeof(data)) == 3
    data = IJF{S}(ArrayType{FT}, zeros; Nij)
    @test ndims(data) == 3
    @test ndims(typeof(data)) == 3
    data = IJFH{S}(ArrayType{FT}, zeros; Nij, Nh)
    @test ndims(data) == 4
    @test ndims(typeof(data)) == 4
    data = IJHF{S}(ArrayType{FT}, zeros; Nij, Nh)
    @test ndims(data) == 4
    @test ndims(typeof(data)) == 4
    data = VIFH{S}(ArrayType{FT}, zeros; Nv, Ni, Nh)
    @test ndims(data) == 4
    @test ndims(typeof(data)) == 4
    data = VIHF{S}(ArrayType{FT}, zeros; Nv, Ni, Nh)
    @test ndims(data) == 4
    @test ndims(typeof(data)) == 4
    data = VIJFH{S}(ArrayType{FT}, zeros; Nv, Nij, Nh)
    @test ndims(data) == 5
    @test ndims(typeof(data)) == 5
    data = VIJHF{S}(ArrayType{FT}, zeros; Nv, Nij, Nh)
    @test ndims(data) == 5
    @test ndims(typeof(data)) == 5
    data = DataLayouts.IJKFVH{S}(ArrayType{FT}, zeros; Nij, Nk, Nv, Nh)
    @test ndims(data) == 6
    @test ndims(typeof(data)) == 6
    data = DataLayouts.IH1JH2{S}(ArrayType{FT}, zeros; Nij)
    @test ndims(data) == 2
    @test ndims(typeof(data)) == 2
end
