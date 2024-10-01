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
    device_zeros(args...) = ClimaComms.array_type(device)(zeros(args...))
    FT = Float64
    S = FT
    Nf = 1
    Nv = 4
    Nij = 3
    Nh = 5
    Nk = 6
    data = DataF{S}(device_zeros(FT, Nf))
    test_universal_size(data)
    data = IJFH{S, Nij}(device_zeros(FT, Nij, Nij, Nf, Nh))
    test_universal_size(data)
    data = IFH{S, Nij}(device_zeros(FT, Nij, Nf, Nh))
    test_universal_size(data)
    data = IJF{S, Nij}(device_zeros(FT, Nij, Nij, Nf))
    test_universal_size(data)
    data = IF{S, Nij}(device_zeros(FT, Nij, Nf))
    test_universal_size(data)
    data = VF{S, Nv}(device_zeros(FT, Nv, Nf))
    test_universal_size(data)
    data = VIJFH{S, Nv, Nij}(device_zeros(FT, Nv, Nij, Nij, Nf, Nh))
    test_universal_size(data)
    data = VIFH{S, Nv, Nij}(device_zeros(FT, Nv, Nij, Nf, Nh))
    test_universal_size(data)
    # data = DataLayouts.IJKFVH{S, Nij, Nk, Nv}(device_zeros(FT,Nij,Nij,Nk,Nf,Nv,Nh)); test_universal_size(data) # TODO: test
    # data = DataLayouts.IH1JH2{S, Nij}(device_zeros(FT,2*Nij,3*Nij));                     test_universal_size(data) # TODO: test
end
