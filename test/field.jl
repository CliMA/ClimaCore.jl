using Test
using StaticArrays, IntervalSets
import ClimaCore.DataLayouts: IJFH
import ClimaCore:
    Fields, slab, Domains, Topologies, Meshes, Operators, Spaces, Geometry
using LinearAlgebra: norm

function spectral_space_2D(; n1 = 1, n2 = 1, Nij = 4)
    domain = Domains.RectangleDomain(
        Geometry.XPoint(-3.0)..Geometry.XPoint(5.0),
        Geometry.YPoint(-2.0)..Geometry.YPoint(8.0),
        x1periodic = false,
        x2periodic = false,
        x1boundary = (:east, :west),
        x2boundary = (:south, :north),
    )
    mesh = Meshes.EquispacedRectangleMesh(domain, n1, n2)
    grid_topology = Topologies.GridTopology(mesh)

    quad = Spaces.Quadratures.GLL{Nij}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    return space
end

@testset "1×1 2D domain space" begin
    Nij = 4
    n1 = n2 = 1
    space = spectral_space_2D(n1 = n1, n2 = n2, Nij = Nij)

    field =
        Fields.Field(IJFH{ComplexF64, Nij}(ones(Nij, Nij, 2, n1 * n2)), space)

    @test sum(field) ≈ Complex(1.0, 1.0) * 8.0 * 10.0 rtol = 10eps()
    @test sum(x -> 3.0, field) ≈ 3 * 8.0 * 10.0 rtol = 10eps()
    @test norm(field) ≈ sqrt(2.0 * 8.0 * 10.0) rtol = 10eps()
    @test norm(field, 1) ≈ norm(Complex(1.0, 1.0)) * 8.0 * 10.0 rtol = 10eps()
    @test norm(field, Inf) ≈ norm(Complex(1.0, 1.0)) rtol = 10eps()

    @test extrema(real, field) == (1.0, 1.0)

    @test Operators.matrix_interpolate(field, 4) ≈
          [Complex(1.0, 1.0) for i in 1:(4 * n1), j in 1:(4 * n2)]


    field_sin = map(x -> sin((x.x) / 2), Fields.coordinate_field(space))
    M = Operators.matrix_interpolate(field_sin, 20)
    @test size(M) == (20, 20)  # 20 x 20 for a 1 element field

    real_field = field.re

    # test broadcasting
    res = field .+ 1
    @test parent(Fields.field_values(res)) == Float64[
        f == 1 ? 2 : 1 for i in 1:Nij, j in 1:Nij, f in 1:2, h in 1:(n1 * n2)
    ]

    res = field.re .+ 1
    @test parent(Fields.field_values(res)) ==
          Float64[2 for i in 1:Nij, j in 1:Nij, f in 1:1, h in 1:(n1 * n2)]
end

@testset "Broadcasting interception for tuple-valued fields" begin
    n1 = n2 = 1
    Nij = 4
    space = spectral_space_2D(n1 = n1, n2 = n2, Nij = Nij)

    nt_field = Fields.Field(
        IJFH{NamedTuple{(:a, :b), Tuple{Float64, Float64}}, Nij}(
            ones(Nij, Nij, 2, n1 * n2),
        ),
        space,
    )
    nt_sum = sum(nt_field)
    @test nt_sum isa NamedTuple{(:a, :b), Tuple{Float64, Float64}}
    @test nt_sum.a ≈ 8.0 * 10.0 rtol = 10eps()
    @test nt_sum.b ≈ 8.0 * 10.0 rtol = 10eps()
    @test norm(nt_field) ≈ sqrt(2.0 * 8.0 * 10.0) rtol = 10eps()
end

@testset "Special case handling for broadcased norm to pass through space local geometry" begin
    space = spectral_space_2D()
    u = Geometry.Covariant12Vector.(ones(space), ones(space))
    @test norm.(u) ≈ hypot(4 / 8 / 2, 4 / 10 / 2) .* ones(space)
end

@testset "FieldVector" begin
    space = spectral_space_2D()
    u = Geometry.Covariant12Vector.(ones(space), ones(space))
    x = Fields.coordinate_field(space)
    y = [1.0, 2.0, 3.0]
    z = 1.0
    Y = Fields.FieldVector(u = u, x = x, y = y, z = z)

    @test propertynames(Y) == (:u, :x, :y, :z)
    @test Y.u === u
    @test Y.y === y
    @test Y.z === z

    Y1 = 2 .* Y
    @test parent(Y1.u) == 2 .* parent(u)
    @test parent(Y1.x) == 2 .* parent(x)
    @test Y1.y == 2 .* y
    @test Y1.z === 2 * z

    Y1 .= Y1 .+ 2 .* Y
    @test parent(Y1.u) == 4 .* parent(u)
    @test parent(Y1.x) == 4 .* parent(x)
    @test Y1.y == 4 .* y
    @test Y1.z === 4 * z

    Y.z = 3.0
    @test Y.z === 3.0
end
