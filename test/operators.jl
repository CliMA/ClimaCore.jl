using Test
using StaticArrays
import ClimaCore.DataLayouts: IJFH, VF
import ClimaCore: Fields, Domains, Topologies, Meshes, Spaces
import ClimaCore.Operators
import ClimaCore.Geometry
using LinearAlgebra, IntervalSets


#@testset "gradient on 1×1 domain SE space" begin
    FT = Float64
    domain = Domains.RectangleDomain(-3..5, -2..8)
    mesh = Meshes.EquispacedRectangleMesh(domain, 1, 1)
    grid_topology = Topologies.GridTopology(mesh)

    Nq = 3
    quad = Spaces.Quadratures.GLL{Nq}()
    points, weights = Spaces.Quadratures.quadrature_points(Float64, quad)

    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    ∇ = Operators.Gradient()
    x1 = map(x -> x.x1, Fields.coordinate_field(space))
    ∇x1 = @. ∇(x1)

    @test ∇x1.u1 ≈ ones(FT, space)
    @test ∇x1.u2 ≈ zeros(FT, space)

    #=
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
          =#
#end

#=
@testset "gradient on -π : π domain SE space" begin
    FT = Float64
    domain = Domains.RectangleDomain(FT(-π)..FT(π), FT(-π)..FT(π))
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
=#
#@testset "divergence of a constant vector field is zero" begin
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
    X = Fields.coordinate_field(space)
    F = f.(X)

    x1 = X.x1

    div = Operators.StrongDivergence()

    divF = @. div(F)
    divF_ref = sin.(X.x1 .+ X.x2)

    @test divF ≈ divF_ref rtol=1e-3

    divgradF = @. div(∇(x1))
    @test divgradF ≈ zeros(FT, space) atol=1e-10

    V = ones(FT, space)
    ndivgradF = @. -div(∇(x1))
    ndivgradF = @. -div(V*∇(x1))
    #=
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
        =#
#end

#=
function boundary_value(
    f::Fields.Field,
    cs::Spaces.FaceFiniteDifferenceSpace,
    boundary::Spaces.ColumnMinMax,
)
    FT = Spaces.undertype(cs)
    column_array = parent(f)
    if length(column_array) == Spaces.n_cells(cs)
        column_array = parent(f)
        ghost_index = Spaces.ghost_index(cs, Spaces.CellCent(), boundary)
        ghost_value = column_array[ghost_index]
        first_interior_index =
            Spaces.interior_index(cs, Spaces.CellCent(), boundary)
        interior_value = column_array[first_interior_index]
        return FT((ghost_value + interior_value) / 2)
    elseif length(column_array) == Spaces.n_faces(cs)
        boundary_index = Spaces.boundary_index(cs, Spaces.CellFace(), boundary)
        return column_array[boundary_index]
    else
        error("Bad field")
    end
end

function ∇boundary_value(
    f::Fields.Field,
    cs::Spaces.FaceFiniteDifferenceSpace,
    boundary::Spaces.ColumnMinMax,
)
    FT = Spaces.undertype(cs)
    column_array = parent(f)
    n_cells = Spaces.n_cells(cs)
    n_faces = Spaces.n_faces(cs)
    if length(column_array) == n_cells
        j = boundary isa Spaces.ColumnMin ? 2 : n_faces - 1
        i = j - 1
        local_operator = parent(cs.∇_cent_to_face)[j]
        local_stencil = column_array[i:(i + 1)] # TODO: remove hard-coded stencil size 2
        return convert(
            FT,
            LinearAlgebra.dot(parent(local_operator), local_stencil),
        )
    elseif length(column_array) == n_faces
        i = boundary isa Spaces.ColumnMin ? 2 : n_faces - 1
        local_operator = parent(cs.∇_face_to_face)[i]
        local_stencil = column_array[(i - 1):(i + 1)] # TODO: remove hard-coded stencil size 2
        return convert(
            FT,
            LinearAlgebra.dot(parent(local_operator), local_stencil),
        )
    else
        error("Bad field")
    end
end

@testset "Test vertical column dirchlet boundry conditions" begin
    FT = Float32
    a = FT(0.0)
    b = FT(1.0)
    n = 10
    cs = Spaces.FaceFiniteDifferenceSpace(a, b, n)nothing)

    vert_cent = Spaces.coordinates(cs, Spaces.CellCent())
    vert_face = Spaces.coordinates(cs, Spaces.CellFace())

    cent_field = Fields.Field(VF{FT}(zeros(FT, Spaces.n_cells(cs), 1)), cs)
    face_field = Fields.Field(VF{FT}(zeros(FT, Spaces.n_faces(cs), 1)), cs)

    value = one(FT)
    Operators.apply_dirichlet!(face_field, value, cs, Spaces.ColumnMax())
    Operators.apply_dirichlet!(face_field, value, cs, Spaces.ColumnMin())

    @test boundary_value(face_field, cs, Spaces.ColumnMin()) ≈ value
    @test boundary_value(face_field, cs, Spaces.ColumnMax()) ≈ value

    value = one(FT)
    Operators.apply_dirichlet!(cent_field, value, cs, Spaces.ColumnMax())
    Operators.apply_dirichlet!(cent_field, value, cs, Spaces.ColumnMin())

    @test boundary_value(cent_field, cs, Spaces.ColumnMin()) ≈ value
    @test boundary_value(cent_field, cs, Spaces.ColumnMax()) ≈ value
end

@testset "Test vertical column neumann boundry conditions" begin
    FT = Float64
    a = FT(0.0)
    b = FT(1.0)
    n = 10
    cs = Spaces.FaceFiniteDifferenceSpace(a, b, n)

    vert_cent = Spaces.coordinates(cs, Spaces.CellCent())
    vert_face = Spaces.coordinates(cs, Spaces.CellFace())

    cent_field = Fields.Field(VF{FT}(zeros(FT, Spaces.n_cells(cs), 1)), cs)
    face_field = Fields.Field(VF{FT}(zeros(FT, Spaces.n_faces(cs), 1)), cs)

    value = one(FT)
    Operators.apply_neumann!(face_field, value, cs, Spaces.ColumnMin())
    Operators.apply_neumann!(face_field, value, cs, Spaces.ColumnMax())

    @test ∇boundary_value(face_field, cs, Spaces.ColumnMin()) ≈ value
    @test ∇boundary_value(face_field, cs, Spaces.ColumnMax()) ≈ value

    value = one(FT)
    Operators.apply_neumann!(cent_field, value, cs, Spaces.ColumnMin())
    Operators.apply_neumann!(cent_field, value, cs, Spaces.ColumnMax())

    @test ∇boundary_value(cent_field, cs, Spaces.ColumnMin()) ≈ value
    @test ∇boundary_value(cent_field, cs, Spaces.ColumnMax()) ≈ value
end
=#
