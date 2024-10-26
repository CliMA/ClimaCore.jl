#=
julia --check-bounds=yes --project
using Revise; include(joinpath("test", "DataLayouts", "unit_non_extruded_broadcast.jl"))
=#
using Test
using ClimaComms
using LazyBroadcast: @lazy
using ClimaCore.DataLayouts
using ClimaCore.Geometry
using ClimaCore: Geometry, Domains, Topologies, Meshes, Spaces, Fields

@testset "unit_non_extruded_broadcast" begin
    a = [1, 2, 3]
    b = [10, 20, 30]
    bc = @lazy @. a + b
    bc = Base.Broadcast.instantiate(bc)
    @test !DataLayouts.isascalar(bc)
    bc = DataLayouts.to_non_extruded_broadcasted(bc)
    @test !DataLayouts.isascalar(bc)
    @test bc[1] == 11.0
    @test bc[2] == 22.0
    @test bc[3] == 33.0
    @test_throws BoundsError bc[4]
end

@testset "unit_non_extruded_broadcast DataF" begin
    device = ClimaComms.device()
    ArrayType = ClimaComms.array_type(device)
    FT = Float64
    S = FT
    data = DataF{S}(ArrayType{FT}, zeros)
    data[] = 5.0

    bc = @lazy @. data + data
    bc = Base.Broadcast.instantiate(bc)
    @test !DataLayouts.isascalar(bc)
    bc = DataLayouts.to_non_extruded_broadcasted(bc)
    @test !DataLayouts.isascalar(bc)
    @test_throws MethodError bc[1]
    @test bc[] == 10.0
end

foo(a, b, c) = a
@testset "unit_non_extruded_broadcast empty field" begin
    device = ClimaComms.device()
    ArrayType = ClimaComms.array_type(device)
    FT = Float64
    S = FT
    data = DataF{S}(ArrayType{FT}, zeros)
    data_empty = similar(data, typeof(()))

    bc = @lazy @. foo(data_empty, data_empty, ())
    bc = Base.Broadcast.instantiate(bc)
    @test !DataLayouts.isascalar(bc)
    bc = DataLayouts.to_non_extruded_broadcasted(bc)
    @test !DataLayouts.isascalar(bc)
    # In the case of the empty field, one
    # cannot get anything from getindex:
    @test_throws BoundsError bc[1]
    @test_throws BoundsError bc[]
end

@testset "unit_non_extruded_broadcast fields" begin
    FT = Float64
    nelems = 3
    zspan = (0, 1)
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(first(zspan)),
        Geometry.ZPoint{FT}(last(zspan));
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain; nelems)
    context = ClimaComms.SingletonCommsContext()
    z_topology = Topologies.IntervalTopology(context, z_mesh)
    cspace = Spaces.CenterFiniteDifferenceSpace(z_topology)
    f = Fields.Field(FT, cspace)
    @. f = FT(2.0)
    tup = (2.0,)
    @. f = tup
    bc = @lazy @. f = FT(2.0)
    bc = Base.Broadcast.instantiate(bc)
    bc = DataLayouts.to_non_extruded_broadcasted(bc)
    @test bc[] == 2.0
end

@testset "Conceptual test (to compare against Base)" begin
    foo(a, b) = a
    bc = @lazy @. foo((), ())
    bc = Base.Broadcast.instantiate(bc)
    @test_throws BoundsError bc[]
end
