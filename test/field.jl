using Test
using StaticArrays
import ClimateMachineCore.DataLayouts: IJFH
import ClimateMachineCore: Fields, slab, Domains, Topologies, Meshes

@testset "1×1 domain mesh" begin
    domain = Domains.RectangleDomain(
        x1min = -3.0,
        x1max = 5.0,
        x2min = -2.0,
        x2max = 8.0,
        x1periodic = false,
        x2periodic = false,
    )
    discretiation = Domains.EquispacedRectangleDiscretization(domain, 1, 1)
    grid_topology = Topologies.GridTopology(discretiation)

    quad = Meshes.Quadratures.GLL{4}()
    points, weights = Meshes.Quadratures.quadrature_points(Float64, quad)

    mesh = Meshes.Mesh2D(grid_topology, quad)

    field = Fields.Field(IJFH{ComplexF64, 4}(ones(4, 4, 2, 1)), mesh)
    real_field = field.re
end
