using Test
using StaticArrays
import ClimaCore.DataLayouts: IJFH, VF
import ClimaCore: Geometry, Fields, Domains, Topologies, Meshes, Spaces, Operators
using LinearAlgebra, IntervalSets

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
    cs = Spaces.FaceFiniteDifferenceSpace(a, b, n)

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

