using Test
using StaticArrays
import ClimateMachineCore.DataLayouts: VF
import ClimateMachineCore: Fields, Spaces
import ClimateMachineCore.Operators
import ClimateMachineCore.Geometry
using LinearAlgebra, IntervalSets

"""
    convergence_rate(err, Δh)

Estimate convergence rate given
vectors `err` and `Δh`

    err = C Δh^p+ H.O.T
    err_k ≈ C Δh_k^p
    err_k/err_m ≈ Δh_k^p/Δh_m^p
    log(err_k/err_m) ≈ log((Δh_k/Δh_m)^p)
    log(err_k/err_m) ≈ p*log(Δh_k/Δh_m)
    log(err_k/err_m)/log(Δh_k/Δh_m) ≈ p

Assumes a uniform grid spacing.
"""
convergence_rate(err, Δh) =
    [log(err[i] / err[i - 1]) / log(Δh[i] / Δh[i - 1]) for i in 2:length(Δh)]

#####
##### Convergence of interpolation operators
#####

@testset "Face -> Center interpolation (uniform and stretched)" begin
    FT = Float64
    a, b = FT(0.0), FT(1.0)
    nr = 2 .^ (5, 6, 7, 8)
    warp_fns =
        (coord -> coord, coord -> a + (b - a) * expm1(2.5 * coord) / expm1(2.5))

    for (i, warp_fn) in enumerate(warp_fns)
        err, Δh = zeros(length(nr)), zeros(length(nr))
        for (k, n) in enumerate(nr)
            cs = Spaces.warp_mesh(
                warp_fn,
                Spaces.FaceFiniteDifferenceSpace(a, b, n),
            )

            vert_cent = Spaces.coordinates(cs, Spaces.CellCent())
            vert_face = Spaces.coordinates(cs, Spaces.CellFace())

            cent_field_exact = Fields.CentField(cs)
            cent_field = Fields.CentField(cs)
            face_field = Fields.FaceField(cs)

            parent(face_field) .= sin.(3 * π * vert_face)
            parent(cent_field_exact) .= sin.(3 * π * vert_cent)

            Operators.vertical_interp!(cent_field, face_field, cs)

            Δh[k] = first(Spaces.Δcoordinates(cs, Spaces.CellCent()))
            err[k] =
                norm(parent(cent_field) .- parent(cent_field_exact)) /
                length(parent(cent_field_exact))
        end
        conv = convergence_rate(err, Δh)
        # conv should be approximately 2 for second order-accurate stencil.
        @test 1.5 ≤ conv[1] ≤ 3
        @test 1.5 ≤ conv[3] ≤ 3
        if i == 1 # TODO: should warped functions converge monotonically, using this definitions?
            @test conv[1] ≤ conv[2] ≤ conv[3]
        end
        @test err[3] ≤ err[2] ≤ err[1] ≤ 1e-2
    end
end

@testset "Center -> Face interpolation (uniform and stretched)" begin
    FT = Float64
    a, b = FT(0.0), FT(1.0)
    nr = 2 .^ (5, 6, 7, 8)
    warp_fns =
        (coord -> coord, coord -> a + (b - a) * expm1(2.5 * coord) / expm1(2.5))

    for warp_fn in warp_fns
        err, Δh = zeros(length(nr)), zeros(length(nr))
        for (k, n) in enumerate(nr)
            cs = Spaces.warp_mesh(
                warp_fn,
                Spaces.FaceFiniteDifferenceSpace(a, b, n),
            )

            vert_cent = Spaces.coordinates(cs, Spaces.CellCent())
            vert_face = Spaces.coordinates(cs, Spaces.CellFace())

            face_field_exact = Fields.FaceField(cs)
            face_field = Fields.FaceField(cs)
            cent_field = Fields.CentField(cs)

            parent(cent_field) .= sin.(3 * π * vert_cent)
            parent(face_field_exact) .= sin.(3 * π * vert_face)

            Operators.vertical_interp!(face_field, cent_field, cs)

            idx_interior = Spaces.interior_face_range(cs)
            Δh[k] = first(Spaces.Δcoordinates(cs, Spaces.CellFace()))
            err_field =
                parent(face_field)[idx_interior] .-
                parent(face_field_exact)[idx_interior]
            err[k] = norm(err_field) / length(parent(face_field_exact))
        end
        conv = convergence_rate(err, Δh)
        # conv should be approximately 2 for second order-accurate stencil.
        @test 1.4 ≤ conv[1] ≤ 3
        @test 1.4 ≤ conv[3] ≤ 3
        @test conv[1] ≤ conv[2] ≤ conv[3]
        @test err[3] ≤ err[2] ≤ err[1] ≤ 4 * 1e-2
    end
end

#####
##### Convergence of derivative operators
#####

@testset "∇_face_to_cent (uniform and stretched)" begin
    FT = Float64
    a, b = FT(0.0), FT(1.0)
    nr = 2 .^ (5, 6, 7, 8)
    warp_fns = (
        :uniform => coord -> coord,
        :stretched =>
            coord -> a + (b - a) * expm1(2.5 * coord) / expm1(2.5),
    )

    for (warp_name, warp_fn) in warp_fns
        err, Δh = zeros(length(nr)), zeros(length(nr))
        for (k, n) in enumerate(nr)
            cs = Spaces.warp_mesh(
                warp_fn,
                Spaces.FaceFiniteDifferenceSpace(a, b, n),
            )

            vert_cent = Spaces.coordinates(cs, Spaces.CellCent())
            vert_face = Spaces.coordinates(cs, Spaces.CellFace())

            cent_field_exact = Fields.CentField(cs)
            cent_field = Fields.CentField(cs)
            face_field = Fields.FaceField(cs)

            parent(face_field) .= sin.(3 * π * vert_face)
            parent(cent_field_exact) .= 3 * π * cos.(3 * π * vert_cent)

            Operators.vertical_gradient!(cent_field, face_field, cs)

            Δh[k] = first(Spaces.Δcoordinates(cs, Spaces.CellCent()))
            err[k] =
                norm(parent(cent_field) .- parent(cent_field_exact)) /
                length(parent(cent_field_exact))
        end
        conv = convergence_rate(err, Δh)
        # conv should be approximately 2 for second order-accurate stencil.
        @test 1.5 ≤ conv[1] ≤ 3
        @test 1.5 ≤ conv[3] ≤ 3
        if warp_name == :uniform # TODO: should warped functions converge monotonically, using this definitions?
            @test conv[1] ≤ conv[2] ≤ conv[3]
        end
        @test err[3] ≤ err[2] ≤ err[1] ≤ 2 * 1e-2
    end
end

@testset "∇_cent_to_face (uniform and stretched)" begin
    FT = Float64
    a, b = FT(0.0), FT(1.0)
    nr = 2 .^ (5, 6, 7, 8)
    warp_fns = (
        :uniform => coord -> coord,
        :stretched =>
            coord -> a + (b - a) * expm1(2.5 * coord) / expm1(2.5),
    )
    for (warp_name, warp_fn) in warp_fns
        err, Δh = zeros(length(nr)), zeros(length(nr))
        for (k, n) in enumerate(nr)
            cs = Spaces.warp_mesh(
                warp_fn,
                Spaces.FaceFiniteDifferenceSpace(a, b, n),
            )

            vert_cent = Spaces.coordinates(cs, Spaces.CellCent())
            vert_face = Spaces.coordinates(cs, Spaces.CellFace())

            face_field_exact = Fields.FaceField(cs)
            face_field = Fields.FaceField(cs)
            cent_field = Fields.CentField(cs)

            parent(cent_field) .= sin.(3 * π * vert_cent)
            parent(face_field_exact) .= 3 * π * cos.(3 * π * vert_face)

            Operators.vertical_gradient!(face_field, cent_field, cs)

            idx_interior = Spaces.interior_face_range(cs)
            Δh[k] = first(Spaces.Δcoordinates(cs, Spaces.CellFace()))
            err_field =
                parent(face_field)[idx_interior] .-
                parent(face_field_exact)[idx_interior]
            err[k] = norm(err_field) / length(parent(face_field_exact))
        end
        conv = convergence_rate(err, Δh)
        # conv should be approximately 2 for second order-accurate stencil.
        @test 1.4 ≤ conv[1] ≤ 3
        @test 1.4 ≤ conv[3] ≤ 3
        if warp_name == :uniform # TODO: should warped functions converge monotonically, using this definitions?
            @test conv[1] ≤ conv[2] ≤ conv[3]
        end
        @test err[3] ≤ err[2] ≤ err[1] ≤ 4 * 1e-2
    end
end

@testset "∇_face_to_face (uniform and stretched)" begin
    FT = Float64
    a, b = FT(0.0), FT(1.0)
    nr = 2 .^ (5, 6, 7, 8)
    warp_fns = (
        :uniform => coord -> coord,
        :stretched =>
            coord -> a + (b - a) * expm1(2.5 * coord) / expm1(2.5),
    )
    for (warp_name, warp_fn) in warp_fns
        err, Δh = zeros(length(nr)), zeros(length(nr))
        for (k, n) in enumerate(nr)
            cs = Spaces.warp_mesh(
                warp_fn,
                Spaces.FaceFiniteDifferenceSpace(a, b, n),
            )

            vert_face = Spaces.coordinates(cs, Spaces.CellFace())

            face_field = Fields.FaceField(cs)
            ∇face_field_exact = Fields.FaceField(cs)
            ∇face_field = Fields.FaceField(cs)

            parent(face_field) .= sin.(3 * π * vert_face)
            parent(∇face_field_exact) .= 3 * π * cos.(3 * π * vert_face)

            Operators.vertical_gradient!(∇face_field, face_field, cs)

            idx_interior = Spaces.interior_face_range(cs)
            Δh[k] = first(Spaces.Δcoordinates(cs, Spaces.CellFace()))
            err_field =
                parent(∇face_field)[idx_interior] .-
                parent(∇face_field_exact)[idx_interior]
            err[k] = norm(err_field) / length(parent(∇face_field_exact))
        end
        conv = convergence_rate(err, Δh)
        # conv should be approximately 2 for second order-accurate stencil.
        @test 1.4 ≤ conv[1] ≤ 3
        @test 1.4 ≤ conv[3] ≤ 3
        if warp_name == :uniform # TODO: should warped functions converge monotonically, using this definitions?
            @test conv[1] ≤ conv[2] ≤ conv[3]
        end
        @test err[3] ≤ err[2] ≤ err[1] ≤ 5 * 1e-2
    end
end

@testset "∇_cent_to_cent (uniform and stretched)" begin
    FT = Float64
    a, b = FT(0.0), FT(1.0)
    nr = 2 .^ (5, 6, 7, 8)
    warp_fns = (
        :uniform => coord -> coord,
        :stretched =>
            coord -> a + (b - a) * expm1(2.5 * coord) / expm1(2.5),
    )
    for (warp_name, warp_fn) in warp_fns
        err, Δh = zeros(length(nr)), zeros(length(nr))
        for (k, n) in enumerate(nr)
            cs = Spaces.warp_mesh(
                warp_fn,
                Spaces.FaceFiniteDifferenceSpace(a, b, n),
            )

            vert_cent = Spaces.coordinates(cs, Spaces.CellCent())

            cent_field = Fields.CentField(cs)
            ∇cent_field_exact = Fields.CentField(cs)
            ∇cent_field = Fields.CentField(cs)

            parent(cent_field) .= sin.(3 * π * vert_cent)
            parent(∇cent_field_exact) .= 3 * π * cos.(3 * π * vert_cent)

            Operators.vertical_gradient!(∇cent_field, cent_field, cs)

            idx_interior = Spaces.interior_cent_range(cs)
            Δh[k] = first(Spaces.Δcoordinates(cs, Spaces.CellCent()))
            err_field =
                parent(∇cent_field)[idx_interior] .-
                parent(∇cent_field_exact)[idx_interior]
            err[k] =
                norm(err_field) /
                length(parent(∇cent_field_exact)[idx_interior])
        end
        conv = convergence_rate(err, Δh)
        # conv should be approximately 2 for second order-accurate stencil.
        @test 1.4 ≤ conv[1] ≤ 3
        @test 1.4 ≤ conv[3] ≤ 3
        if warp_name == :uniform # TODO: should warped functions converge monotonically, using this definitions?
            @test conv[1] ≤ conv[2] ≤ conv[3]
        end
        @test err[3] ≤ err[2] ≤ err[1] ≤ 4 * 1e-2
    end
end

#####
##### Boundary conditions
#####

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

    cent_field = Fields.CentField(cs)
    face_field = Fields.FaceField(cs)

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

    cent_field = Fields.CentField(cs)
    face_field = Fields.FaceField(cs)

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
