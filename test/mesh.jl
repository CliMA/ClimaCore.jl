using Test
using StaticArrays, IntervalSets, LinearAlgebra
import ClimateMachineCore: slab, Domains, Topologies, Meshes
import ClimateMachineCore.Domains.Geometry: Cartesian2DPoint

@testset "1×1 domain mesh" begin
    domain = Domains.RectangleDomain(
        -3..5,
        -2..8,
        x1periodic = true,
        x2periodic = false,
    )
    discretization = Domains.EquispacedRectangleDiscretization(domain, 1, 1)
    grid_topology = Topologies.GridTopology(discretization)

    quad = Meshes.Quadratures.GLL{4}()
    points, weights = Meshes.Quadratures.quadrature_points(Float64, quad)

    mesh = Meshes.Mesh2D(grid_topology, quad)

    array = getfield(mesh.coordinates, :array)
    @test size(array) == (4, 4, 2, 1)
    coord_slab = slab(mesh.coordinates, 1)
    @test coord_slab[1, 1] ≈ Cartesian2DPoint(-3.0, -2.0)
    @test coord_slab[4, 1] ≈ Cartesian2DPoint(5.0, -2.0)
    @test coord_slab[1, 4] ≈ Cartesian2DPoint(-3.0, 8.0)
    @test coord_slab[4, 4] ≈ Cartesian2DPoint(5.0, 8.0)

    local_geometry_slab = slab(mesh.local_geometry, 1)
    for i in 1:4, j in 1:4
        @test local_geometry_slab[i, j].∂ξ∂x ≈ @SMatrix [2/8 0; 0 2/10]
        @test local_geometry_slab[i, j].J ≈ (10 / 2) * (8 / 2)
        @test local_geometry_slab[i, j].WJ ≈
              (10 / 2) * (8 / 2) * weights[i] * weights[j]
        if i in (1, 4)
            @test 2 *
                  local_geometry_slab[i, j].invM *
                  local_geometry_slab[i, j].WJ ≈ 1
        else
            @test local_geometry_slab[i, j].invM *
                  local_geometry_slab[i, j].WJ ≈ 1
        end
    end

    @test length(mesh.boundary_surface_geometries) == 2
    @test keys(mesh.boundary_surface_geometries) == (:south, :north)
    @test sum(parent(mesh.boundary_surface_geometries.north.sWJ)) ≈ 8
    @test parent(mesh.boundary_surface_geometries.north.normal)[1, :, 1] ≈
          [0.0, 1.0]
end

@testset "ColumnMesh" begin
    for FT in (Float32, Float64)
        a = FT(0.0)
        b = FT(1.0)
        n = 10
        cm = Meshes.FaceColumnMesh(a, b, n)
        @test cm.cent.Δh[1] ≈ FT(1 / 10)
        @test cm.face.Δh[1] ≈ FT(1 / 10)
        sprint(show, cm)
        @test cm.cent == Meshes.coords(cm, Meshes.CellCent())
        @test cm.face == Meshes.coords(cm, Meshes.CellFace())
    end
    @test Meshes.n_hat(Meshes.ColumnMin()) == -1
    @test Meshes.binary(Meshes.ColumnMin()) == 0

    @test Meshes.n_hat(Meshes.ColumnMax()) == 1
    @test Meshes.binary(Meshes.ColumnMax()) == 1

    # @show collect(Meshes.column(cm, Meshes.CellCent()))
    # @show collect(Meshes.column(cm, Meshes.CellFace()))

    # https://github.com/charleskawczynski/MOONS.jl/blob/main/test/Fields/interpolations_convergence.jl#L21-L50
    # for i_cent in column(cm, Meshes.CellCent())
    #     cent_stencil = i_cent-1:i_cent+1
    #     cent_view = @view cent_data[cent_stencil]
    #     face_view = @view face_data[face_stencil]
    #     face_data[face_stencil] = cm.interp_cent_to_face
    # end
end


#TODO:  move to operators
import ClimateMachineCore.Operators
import ClimateMachineCore.Fields: Fields
import ClimateMachineCore.DataLayouts: VF
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

"""
convergence_rate(err, Δh) =
    [log(err[i] / err[i - 1]) / log(Δh[i] / Δh[i - 1]) for i in 2:length(Δh)]

@testset "Face -> Center interpolation (uniform and stretched)" begin
    FT = Float64
    a, b = FT(0.0), FT(1.0)
    nr = 2 .^ (5, 6, 7, 8)
    warp_fns =
        (coord -> coord, coord -> a + (b - a) * expm1(2.5 * coord) / expm1(2.5))

    for (i, warp_fn) in enumerate(warp_fns)
        err, Δh = zeros(length(nr)), zeros(length(nr))
        for (k, n) in enumerate(nr)
            cm = Meshes.warp_mesh(warp_fn, Meshes.FaceColumnMesh(a, b, n))

            vert_cent = Meshes.coordinates(cm, Meshes.CellCent())
            vert_face = Meshes.coordinates(cm, Meshes.CellFace())

            cent_field_exact =
                Fields.Field(VF{FT}(zeros(FT, Meshes.n_cells(cm), 1)), cm)
            cent_field =
                Fields.Field(VF{FT}(zeros(FT, Meshes.n_cells(cm), 1)), cm)

            face_field =
                Fields.Field(VF{FT}(zeros(FT, Meshes.n_faces(cm), 1)), cm)

            parent(face_field) .= sin.(3 * π * vert_face)
            parent(cent_field_exact) .= sin.(3 * π * vert_cent)

            Operators.vertical_interp!(cent_field, face_field, cm)

            Δh[k] = first(Meshes.Δcoordinates(cm, Meshes.CellCent()))
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
            cm = Meshes.warp_mesh(warp_fn, Meshes.FaceColumnMesh(a, b, n))

            vert_cent = Meshes.coordinates(cm, Meshes.CellCent())
            vert_face = Meshes.coordinates(cm, Meshes.CellFace())

            face_field_exact =
                Fields.Field(VF{FT}(zeros(FT, Meshes.n_faces(cm), 1)), cm)
            face_field =
                Fields.Field(VF{FT}(zeros(FT, Meshes.n_faces(cm), 1)), cm)

            cent_field =
                Fields.Field(VF{FT}(zeros(FT, Meshes.n_cells(cm), 1)), cm)

            parent(cent_field) .= sin.(3 * π * vert_cent)
            parent(face_field_exact) .= sin.(3 * π * vert_face)

            Operators.vertical_interp!(face_field, cent_field, cm)

            idx_interior = Meshes.interior_face_range(cm)
            Δh[k] = first(Meshes.Δcoordinates(cm, Meshes.CellFace()))
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
            cm = Meshes.warp_mesh(warp_fn, Meshes.FaceColumnMesh(a, b, n))

            vert_cent = Meshes.coordinates(cm, Meshes.CellCent())
            vert_face = Meshes.coordinates(cm, Meshes.CellFace())

            cent_field_exact =
                Fields.Field(VF{FT}(zeros(FT, Meshes.n_cells(cm), 1)), cm)
            cent_field =
                Fields.Field(VF{FT}(zeros(FT, Meshes.n_cells(cm), 1)), cm)

            face_field =
                Fields.Field(VF{FT}(zeros(FT, Meshes.n_faces(cm), 1)), cm)

            parent(face_field) .= sin.(3 * π * vert_face)
            parent(cent_field_exact) .= 3 * π * cos.(3 * π * vert_cent)

            Operators.vertical_gradient!(cent_field, face_field, cm)

            Δh[k] = first(Meshes.Δcoordinates(cm, Meshes.CellCent()))
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
            cm = Meshes.warp_mesh(warp_fn, Meshes.FaceColumnMesh(a, b, n))

            vert_cent = Meshes.coordinates(cm, Meshes.CellCent())
            vert_face = Meshes.coordinates(cm, Meshes.CellFace())

            face_field_exact =
                Fields.Field(VF{FT}(zeros(FT, Meshes.n_faces(cm), 1)), cm)
            face_field =
                Fields.Field(VF{FT}(zeros(FT, Meshes.n_faces(cm), 1)), cm)

            cent_field =
                Fields.Field(VF{FT}(zeros(FT, Meshes.n_cells(cm), 1)), cm)

            parent(cent_field) .= sin.(3 * π * vert_cent)
            parent(face_field_exact) .= 3 * π * cos.(3 * π * vert_face)

            Operators.vertical_gradient!(face_field, cent_field, cm)

            idx_interior = Meshes.interior_face_range(cm)
            Δh[k] = first(Meshes.Δcoordinates(cm, Meshes.CellFace()))
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
