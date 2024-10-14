#=
julia --project
using Revise; include(joinpath("test", "DataLayouts", "opt_universal_size.jl"))
=#
using Test
using ClimaCore.DataLayouts
using ClimaCore: DataLayouts, Geometry
import ClimaComms
using StaticArrays: SMatrix
ClimaComms.@import_required_backends
using JET
using InteractiveUtils: @code_typed

function test_universal_size(data)
    us = DataLayouts.UniversalSize(data)
    # Make sure results is statically returned / constant propagated

    # We cannot statically know Nh or N until we put Nh back
    # into the type space. So some of these tests have been
    # commented out until we add it back in.

    # ct = @code_typed DataLayouts.get_N(us)
    # @test ct.first.code[1] isa Core.ReturnNode
    # @test ct.first.code[end].val == DataLayouts.get_N(us)

    ct = @code_typed DataLayouts.get_Nv(us)
    @test ct.first.code[1] isa Core.ReturnNode
    @test ct.first.code[end].val == DataLayouts.get_Nv(us)

    ct = @code_typed DataLayouts.get_Nij(us)
    @test ct.first.code[1] isa Core.ReturnNode
    @test ct.first.code[end].val == DataLayouts.get_Nij(us)

    # ct = @code_typed DataLayouts.get_Nh(us)
    # @test ct.first.code[1] isa Core.ReturnNode
    # @test ct.first.code[end].val == DataLayouts.get_Nh(us)

    # ct = @code_typed size(data)
    # @test ct.first.code[1] isa Core.ReturnNode
    # @test ct.first.code[end].val == size(data)

    # ct = @code_typed DataLayouts.get_N(data)
    # @test ct.first.code[1] isa Core.ReturnNode
    # @test ct.first.code[end].val == DataLayouts.get_N(data)

    # Demo of failed constant prop:
    ct = @code_typed prod(size(data))
    @test ct.first.code[1] isa Expr # first element is not a return node, but an expression
end

@testset "UniversalSize" begin
    device = ClimaComms.device()
    ArrayType = ClimaComms.array_type(device)
    FT = Float64
    S = FT
    Nv = 4
    Ni = Nij = 3
    Nh = 5
    Nk = 6
    data = DataF{S}(ArrayType{FT}, zeros)
    test_universal_size(data)
    data = IJFH{S}(ArrayType{FT}, zeros; Nij, Nh)
    test_universal_size(data)
    data = IFH{S}(ArrayType{FT}, zeros; Ni, Nh)
    test_universal_size(data)
    data = IJF{S}(ArrayType{FT}, zeros; Nij)
    test_universal_size(data)
    data = IF{S}(ArrayType{FT}, zeros; Ni)
    test_universal_size(data)
    data = VF{S}(ArrayType{FT}, zeros; Nv)
    test_universal_size(data)
    data = VIJFH{S}(ArrayType{FT}, zeros; Nv, Nij, Nh)
    test_universal_size(data)
    data = VIFH{S}(ArrayType{FT}, zeros; Nv, Ni, Nh)
    test_universal_size(data)
    # data = DataLayouts.IJKFVH{S}(ArrayType{FT}, zeros; Nij,Nk,Nv,Nh);  test_universal_size(data) # TODO: test
    # data = DataLayouts.IH1JH2{S}(ArrayType{FT}, zeros; Nij);           test_universal_size(data) # TODO: test
end
