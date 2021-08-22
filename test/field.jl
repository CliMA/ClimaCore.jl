
using Test
using StaticArrays, IntervalSets
import ClimaCore.DataLayouts: IJFH
import ClimaCore:
    Fields, slab, Domains, Topologies, Meshes, Operators, Spaces, Geometry
using LinearAlgebra: norm

using UnicodePlots

domain = Domains.RectangleDomain(
    -3..5,
    -2..8,
    x1periodic = false,
    x2periodic = false,
    x1boundary = (:east, :west),
    x2boundary = (:south, :north),
)
n1, n2 = 1, 1
Nij = 4
mesh = Meshes.EquispacedRectangleMesh(domain, n1, n2)
grid_topology = Topologies.GridTopology(mesh)

quad = Spaces.Quadratures.GLL{Nij}()
points, weights = Spaces.Quadratures.quadrature_points(Float64, quad)

space = Spaces.SpectralElementSpace2D(grid_topology, quad)


@testset "1×1 domain space" begin

    field =
        Fields.Field(IJFH{ComplexF64, Nij}(ones(Nij, Nij, 2, n1 * n2)), space)

    @test sum(field) ≈ Complex(1.0, 1.0) * 8.0 * 10.0 rtol = 10eps()
    @test sum(x -> 3.0, field) ≈ 3 * 8.0 * 10.0 rtol = 10eps()
    @test norm(field) ≈ sqrt(2.0 * 8.0 * 10.0) rtol = 10eps()


    @test Operators.matrix_interpolate(field, 4) ≈
          [Complex(1.0, 1.0) for i in 1:(4 * n1), j in 1:(4 * n2)]



    #@test parent(Fields.field_values(3 .* nt_field)) ==

    field_sin = map(x -> sin((x.x1) / 2), Fields.coordinate_field(space))

    heatmap(field_sin)

    Operators.matrix_interpolate(field_sin, 20)
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
    u = Geometry.Covariant12Vector.(ones(space), ones(space))
    @test norm.(u) ≈ hypot(4 / 8 / 2, 4 / 10 / 2) .* ones(space)
end

@testset "FieldVector" begin
    u = Geometry.Covariant12Vector.(ones(space), ones(space))
    x = Fields.coordinate_field(space)
    Y = Fields.FieldVector(u = u, x = x)

    Y1 = 2 .* Y
    @test parent(Y1.u) == 2 .* parent(u)
    @test parent(Y1.x) == 2 .* parent(x)

    Y1 .= Y1 .+ 2 .* Y
    @test parent(Y1.u) == 4 .* parent(u)
    @test parent(Y1.x) == 4 .* parent(x)

end
