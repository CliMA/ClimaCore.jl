using Test
using StaticArrays
import ClimateMachineCore.DataLayouts: IJFH
import ClimateMachineCore: Fields, slab, Domains, Topologies, Meshes


using UnicodePlots


@testset "1×1 domain mesh" begin
    domain = Domains.RectangleDomain(
        x1min = -3.0,
        x1max = 5.0,
        x2min = -2.0,
        x2max = 8.0,
        x1periodic = false,
        x2periodic = false,
    )
    n1, n2 = 1, 1
    Nij = 4
    discretization = Domains.EquispacedRectangleDiscretization(domain, n1, n2)
    grid_topology = Topologies.GridTopology(discretization)

    quad = Meshes.Quadratures.GLL{Nij}()
    points, weights = Meshes.Quadratures.quadrature_points(Float64, quad)

    mesh = Meshes.Mesh2D(grid_topology, quad)

    field =
        Fields.Field(IJFH{ComplexF64, Nij}(ones(Nij, Nij, 2, n1 * n2)), mesh)

    @test Fields.matrix_interpolate(field, 4) ≈
          [Complex(1.0, 1.0) for i in 1:(4 * n1), j in 1:(4 * n2)]

    f(x) = sin((x.x1) / 2)
    field_sin = f.(Fields.coordinate_field(mesh))

    heatmap(field_sin)

    Fields.matrix_interpolate(field_sin, 20)
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
