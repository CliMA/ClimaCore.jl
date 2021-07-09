using Test
using StaticArrays
import ClimaCore.DataLayouts: IJFH, VF
import ClimaCore: Fields, Domains, Topologies, Meshes, Spaces
import ClimaCore.Operators
import ClimaCore.Geometry
using LinearAlgebra, IntervalSets

@testset "gradient on 1×1 domain SE space" begin
    domain = Domains.RectangleDomain(
        -3..5,
        -2..8,
        x1periodic = true,
        x2periodic = true,
    )
    mesh = Meshes.EquispacedRectangleMesh(domain, 1, 1)
    grid_topology = Topologies.GridTopology(mesh)

    Nq = 3
    quad = Spaces.Quadratures.GLL{Nq}()
    points, weights = Spaces.Quadratures.quadrature_points(Float64, quad)

    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    f(x) = (x.x1, x.x2)
    field = f.(Fields.coordinate_field(space))

    data = Fields.field_values(field)
    ∇data = Operators.slab_gradient!(
        similar(data, NTuple{2, Geometry.Cartesian12Vector{Float64}}),
        data,
        axes(field),
    )
    @test parent(∇data) ≈
          Float64[f == 1 || f == 4 for i in 1:Nq, j in 1:Nq, f in 1:4, h in 1:1]
end

@testset "gradient on -π : π domain SE space" begin
    FT = Float64
    domain = Domains.RectangleDomain(
        FT(-π)..FT(π),
        FT(-π)..FT(π),
        x1periodic = true,
        x2periodic = true,
    )
    mesh = Meshes.EquispacedRectangleMesh(domain, 5, 5)
    grid_topology = Topologies.GridTopology(mesh)

    Nq = 6
    quad = Spaces.Quadratures.GLL{Nq}()
    points, weights = Spaces.Quadratures.quadrature_points(Float64, quad)
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    field = sin.(Fields.coordinate_field(space).x1)

    data = Fields.field_values(field)
    ∇data = Operators.slab_gradient!(
        similar(data, Geometry.Cartesian12Vector{Float64}),
        data,
        axes(field),
    )
    @test parent(∇data.u1) ≈
          parent(Fields.field_values(cos.(Fields.coordinate_field(space).x1))) rtol =
        1e-3
    Spaces.horizontal_dss!(∇data, space)

    S = similar(data, Float64)
    S .= 1.0
    Spaces.horizontal_dss!(S, space)
    S .= inv.(S)

    ∇data .= S .* ∇data

    @test parent(∇data.u1) ≈
          parent(Fields.field_values(cos.(Fields.coordinate_field(space).x1))) rtol =
        1e-3
end

@testset "divergence of a constant vector field is zero" begin
    FT = Float64
    domain = Domains.RectangleDomain(
        -3..5,
        -2..8,
        x1periodic = true,
        x2periodic = true,
    )
    mesh = Meshes.EquispacedRectangleMesh(domain, 5, 5)
    grid_topology = Topologies.GridTopology(mesh)

    Nq = 6
    quad = Spaces.Quadratures.GLL{Nq}()
    points, weights = Spaces.Quadratures.quadrature_points(Float64, quad)
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    f(x) = Geometry.Cartesian12Vector{Float64}(
        sin(x.x1) * sin(x.x2),
        sin(x.x1) * sin(x.x2),
    )
    # ∂_x1 f + ∂_x2 f = cos(x1)*sin(x2) x̂ + sin(x1)*cos(x2) ŷ
    field = f.(Fields.coordinate_field(space))

    data = Fields.field_values(field)
    div_data =
        Operators.slab_divergence!(similar(data, Float64), data, axes(field))
    divf(x) = sin(x.x1 + x.x2)
    @test parent(div_data) ≈
          parent(Fields.field_values(divf.(Fields.coordinate_field(space)))) rtol =
        1e-3

    # Jacobian-weighted DSS
    SJ = copy(space.local_geometry.J)
    Spaces.horizontal_dss!(SJ, space)
    dss_div_data =
        Spaces.horizontal_dss!(space.local_geometry.J .* div_data, space) ./ SJ
    @test parent(div_data) ≈
          parent(Fields.field_values(divf.(Fields.coordinate_field(space)))) rtol =
        1e-3
end
