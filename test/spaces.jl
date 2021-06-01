using Test
using StaticArrays, IntervalSets, LinearAlgebra

import ClimateMachineCore: slab, Domains, Meshes, Topologies, Spaces
import ClimateMachineCore.Domains.Geometry: Cartesian2DPoint

@testset "1×1 domain space" begin
    domain = Domains.RectangleDomain(
        -3..5,
        -2..8,
        x1periodic = true,
        x2periodic = false,
    )
    mesh = Meshes.EquispacedRectangleMesh(domain, 1, 1)
    grid_topology = Topologies.GridTopology(mesh)

    quad = Spaces.Quadratures.GLL{4}()
    points, weights = Spaces.Quadratures.quadrature_points(Float64, quad)

    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    array = getfield(space.coordinates, :array)
    @test size(array) == (4, 4, 2, 1)
    coord_slab = slab(space.coordinates, 1)
    @test coord_slab[1, 1] ≈ Cartesian2DPoint(-3.0, -2.0)
    @test coord_slab[4, 1] ≈ Cartesian2DPoint(5.0, -2.0)
    @test coord_slab[1, 4] ≈ Cartesian2DPoint(-3.0, 8.0)
    @test coord_slab[4, 4] ≈ Cartesian2DPoint(5.0, 8.0)

    local_geometry_slab = slab(space.local_geometry, 1)
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

    @test length(space.boundary_surface_geometries) == 2
    @test keys(space.boundary_surface_geometries) == (:south, :north)
    @test sum(parent(space.boundary_surface_geometries.north.sWJ)) ≈ 8
    @test parent(space.boundary_surface_geometries.north.normal)[1, :, 1] ≈
          [0.0, 1.0]
end

@testset "Column FiniteDifferenceSpace" begin
    for FT in (Float32, Float64)
        a = FT(0.0)
        b = FT(1.0)
        n = 10
        cs = Spaces.FaceFiniteDifferenceSpace(a, b, n)
        @test cs.cent.Δh[1] ≈ FT(1 / 10)
        @test cs.face.Δh[1] ≈ FT(1 / 10)
        @test cs.cent == Spaces.coords(cs, Spaces.CellCent())
        @test cs.face == Spaces.coords(cs, Spaces.CellFace())
    end
    @test Spaces.n_hat(Spaces.ColumnMin()) == -1
    @test Spaces.binary(Spaces.ColumnMin()) == 0

    @test Spaces.n_hat(Spaces.ColumnMax()) == 1
    @test Spaces.binary(Spaces.ColumnMax()) == 1
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
