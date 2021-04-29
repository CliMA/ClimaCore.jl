using Test
using StaticArrays
import ClimateMachineCore.DataLayouts: IJFH
import ClimateMachineCore: Fields, Domains, Topologies, Meshes
import ClimateMachineCore.Operators
import ClimateMachineCore.Geometry
using LinearAlgebra

@testset "gradient on 1×1 domain mesh" begin
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

    Nq = 3
    quad = Meshes.Quadratures.GLL{Nq}()
    points, weights = Meshes.Quadratures.quadrature_points(Float64, quad)

    mesh = Meshes.Mesh2D(grid_topology, quad)
    f(x) = (x.x1, x.x2)
    field = f.(Fields.coordinate_field(mesh))

    data = Fields.field_values(field)
    ∇data = Operators.volume_gradient!(
        similar(data, NTuple{2, Geometry.Cartesian2DVector{Float64}}),
        data,
        Fields.field_mesh(field),
    )
    @test parent(∇data) ≈
          Float64[f == 1 || f == 4 for i in 1:Nq, j in 1:Nq, f in 1:4, h in 1:1]
end

@testset "gradient on -π : π domain mesh" begin
    FT = Float64
    domain = Domains.RectangleDomain(
        x1min = FT(-π),
        x1max = FT(π),
        x2min = FT(-π),
        x2max = FT(π),
        x1periodic = false,
        x2periodic = false,
    )
    discretiation = Domains.EquispacedRectangleDiscretization(domain, 5, 5)
    grid_topology = Topologies.GridTopology(discretiation)

    Nq = 6
    quad = Meshes.Quadratures.GLL{Nq}()
    points, weights = Meshes.Quadratures.quadrature_points(Float64, quad)
    mesh = Meshes.Mesh2D(grid_topology, quad)
    field = sin.(Fields.coordinate_field(mesh).x1)

    data = Fields.field_values(field)
    ∇data = Operators.volume_gradient!(
        similar(data, Geometry.Cartesian2DVector{Float64}),
        data,
        Fields.field_mesh(field),
    )
    @test parent(∇data.u1) ≈
          parent(Fields.field_values(cos.(Fields.coordinate_field(mesh).x1))) rtol =
        1e-3
    Operators.horizontal_dss!(∇data, mesh)

    S = similar(data, Float64)
    S .= 1.0
    Operators.horizontal_dss!(S, mesh)
    S .= inv.(S)

    ∇data .= S .* ∇data

    @test parent(∇data.u1) ≈
          parent(Fields.field_values(cos.(Fields.coordinate_field(mesh).x1))) rtol =
        1e-3
end

@testset "divergence of a constant vector field is zero" begin
    FT = Float64
    domain = Domains.RectangleDomain(
        x1min = FT(-π),
        x1max = FT(π),
        x2min = FT(-π),
        x2max = FT(π),
        x1periodic = true,
        x2periodic = true,
    )
    discretiation = Domains.EquispacedRectangleDiscretization(domain, 5, 5)
    grid_topology = Topologies.GridTopology(discretiation)

    Nq = 6
    quad = Meshes.Quadratures.GLL{Nq}()
    points, weights = Meshes.Quadratures.quadrature_points(Float64, quad)
    mesh = Meshes.Mesh2D(grid_topology, quad)
    f(x) = Geometry.Cartesian2DVector{Float64}(
        sin(x.x1) * sin(x.x2),
        sin(x.x1) * sin(x.x2),
    )
    # ∂_x1 f + ∂_x2 f = cos(x1)*sin(x2) x̂ + sin(x1)*cos(x2) ŷ
    field = f.(Fields.coordinate_field(mesh))

    data = Fields.field_values(field)
    div_data = Operators.volume_divergence!(
        similar(data, Float64),
        data,
        Fields.field_mesh(field),
    )
    divf(x) = sin(x.x1 + x.x2)
    @test parent(div_data) ≈
          parent(Fields.field_values(divf.(Fields.coordinate_field(mesh)))) rtol =
        1e-3

    # Jacobian-weighted DSS
    SJ = copy(mesh.local_geometry.J)
    Operators.horizontal_dss!(SJ, mesh)
    dss_div_data =
        Operators.horizontal_dss!(mesh.local_geometry.J .* div_data, mesh) ./ SJ
    @test parent(div_data) ≈
          parent(Fields.field_values(divf.(Fields.coordinate_field(mesh)))) rtol =
        1e-3
end
