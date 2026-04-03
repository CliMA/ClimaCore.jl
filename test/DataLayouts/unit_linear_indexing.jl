using Test
using IntervalSets
using ClimaComms
ClimaComms.@import_required_backends

import ClimaCore: Geometry, Domains, Meshes, Spaces
import ClimaCore.DataLayouts: get_struct

struct Foo{T}
    x::T
    y::T
end

@testset "get_struct - IHF indexing (float)" begin
    FT = Float64
    S = FT
    a = reshape(FT.(1:12), 3, 4, 1)
    for i in 1:12
        @test get_struct(a, S, i) == FT(i)
    end
    @test_throws BoundsError get_struct(a, S, 13)
end

@testset "get_struct - IHF indexing" begin
    FT = Float64
    S = Foo{FT}
    a = reshape(FT.(1:24), 3, 4, 2)
    for i in 1:12
        @test get_struct(a, S, i) == Foo{FT}(i, 12 + i)
    end
    @test_throws BoundsError get_struct(a, S, 13)
end

@testset "get_struct - IJF indexing" begin
    FT = Float64
    S = Foo{FT}
    a = reshape(FT.(1:24), 3, 4, 2)
    for i in 1:12
        @test get_struct(a, S, i) == Foo{FT}(i, 12 + i)
    end
    @test_throws BoundsError get_struct(a, S, 13)
end

@testset "get_struct - VIJHF indexing" begin
    FT = Float64
    S = Foo{FT}
    a = reshape(FT.(1:32), 2, 2, 2, 2, 2)
    for i in 1:16
        @test get_struct(a, S, i) == Foo{FT}(i, 16 + i)
    end
    @test_throws BoundsError get_struct(a, S, 17)
end

@testset "get_struct - example" begin
    FT = Float64
    stretch_fn = Meshes.Uniform()
    interval = Geometry.ZPoint(FT(0.0)) .. Geometry.ZPoint(FT(1.0))
    domain = Domains.IntervalDomain(interval; boundary_names = (:left, :right))
    mesh = Meshes.IntervalMesh(domain, stretch_fn, nelems = 5)
    space = Spaces.FaceFiniteDifferenceSpace(ClimaComms.device(), mesh)
    lg_data = Spaces.local_geometry_data(space)
    a = parent(lg_data)
    S = eltype(lg_data)
    for i in 1:6
        @test get_struct(a, S, i) == get_struct(a, S, CartesianIndex(i), Val(2))
    end
    @test_throws BoundsError get_struct(a, S, 7)
end

# TODO: add set_struct!
