using Test
using SparseArrays

import ClimaCore: ClimaCore, Domains, Meshes, Geometry

function unit_intervalmesh(;
    stretching = nothing,
    nelems = 20,
    periodic = false,
    reverse_mode = false,
)
    if periodic
        dom = ClimaCore.Domains.IntervalDomain(
            reverse_mode ? ClimaCore.Geometry.ZPoint(-1.0) :
            ClimaCore.Geometry.ZPoint(0.0),
            reverse_mode ? ClimaCore.Geometry.ZPoint(0.0) :
            ClimaCore.Geometry.ZPoint(1.0);
            periodic = true,
        )
    else
        dom = ClimaCore.Domains.IntervalDomain(
            reverse_mode ? ClimaCore.Geometry.ZPoint(-1.0) :
            ClimaCore.Geometry.ZPoint(0.0),
            reverse_mode ? ClimaCore.Geometry.ZPoint(0.0) :
            ClimaCore.Geometry.ZPoint(1.0),
            boundary_names = (:left, :right),
        )
    end
    if stretching !== nothing
        return dom,
        ClimaCore.Meshes.IntervalMesh(
            dom,
            stretching,
            nelems = nelems,
            reverse_mode = reverse_mode,
        )
    else
        return dom, ClimaCore.Meshes.IntervalMesh(dom, nelems = nelems)
    end
end

@testset "IntervalMesh" begin
    for nelems in (-1, 0)
        @test_throws ArgumentError unit_intervalmesh(nelems = nelems)
    end
    for nelems in (1, 3)
        dom, mesh = unit_intervalmesh(nelems = nelems)
        @test length(mesh.faces) == nelems + 1 # number of faces is +1 elements
        @test Meshes.nelements(mesh) == nelems
        @test Meshes.elements(mesh) == UnitRange(1, nelems)
        @test Meshes.element_horizontal_length_scale(mesh) ≈ 1 / nelems
    end
    dom, mesh = unit_intervalmesh(nelems = 2)
    @test Meshes.domain(mesh) isa Domains.IntervalDomain
    @test Meshes.is_boundary_face(mesh, 1, 1) == true
    @test Meshes.boundary_face_name(mesh, 1, 1) == :left
    @test Meshes.is_boundary_face(mesh, 1, 2) == false
    @test Meshes.boundary_face_name(mesh, 1, 2) === nothing
    @test Meshes.is_boundary_face(mesh, 2, 1) == false
    @test Meshes.boundary_face_name(mesh, 2, 1) === nothing
    @test Meshes.is_boundary_face(mesh, 2, 2) == true
    @test Meshes.boundary_face_name(mesh, 2, 2) == :right

    @test_throws BoundsError Meshes.coordinates(mesh, 0, 1)
    @test Meshes.coordinates(mesh, 1, 1) == Geometry.ZPoint(0.0)
    @test Meshes.coordinates(mesh, 2, 2) == Geometry.ZPoint(1.0)
    # an element of a 1D mesh has 2 vertices; `vert = 3` used to index past
    # them into the next element's faces, which returned a plausible-looking
    # coordinate rather than erroring (#1026)
    @test_throws ArgumentError Meshes.coordinates(mesh, 1, 3)
    @test_throws ArgumentError Meshes.coordinates(mesh, 2, 3)
    @test_throws ArgumentError Meshes.coordinates(mesh, 3, 3)
    @test_throws ArgumentError Meshes.coordinates(mesh, 1, 0)
end


@testset "IntervalMesh periodic" begin
    for nelems in (-1, 0)
        @test_throws ArgumentError unit_intervalmesh(
            nelems = nelems,
            periodic = true,
        )
    end
    for nelems in (1, 3)
        dom, mesh = unit_intervalmesh(nelems = nelems, periodic = true)
        @test length(mesh.faces) == nelems + 1 # number of faces is +1 elements
        @test Meshes.nelements(mesh) == nelems
        @test Meshes.elements(mesh) == UnitRange(1, nelems)
    end
    dom, mesh = unit_intervalmesh(nelems = 2, periodic = true)
    @test Meshes.domain(mesh) isa Domains.IntervalDomain
    @test Meshes.is_boundary_face(mesh, 1, 1) == false
    @test Meshes.boundary_face_name(mesh, 1, 1) === nothing
    @test Meshes.is_boundary_face(mesh, 1, 2) == false
    @test Meshes.boundary_face_name(mesh, 1, 2) === nothing
    @test Meshes.is_boundary_face(mesh, 2, 1) == false
    @test Meshes.boundary_face_name(mesh, 2, 1) === nothing
    @test Meshes.is_boundary_face(mesh, 2, 2) == false
    @test Meshes.boundary_face_name(mesh, 2, 2) === nothing

    @test_throws BoundsError Meshes.coordinates(mesh, 0, 1)
    @test Meshes.coordinates(mesh, 1, 1) == Geometry.ZPoint(0.0)
    @test Meshes.coordinates(mesh, 2, 2) == Geometry.ZPoint(1.0)
    # an element of a 1D mesh has 2 vertices; `vert = 3` used to index past
    # them into the next element's faces, which returned a plausible-looking
    # coordinate rather than erroring (#1026)
    @test_throws ArgumentError Meshes.coordinates(mesh, 1, 3)
    @test_throws ArgumentError Meshes.coordinates(mesh, 2, 3)
    @test_throws ArgumentError Meshes.coordinates(mesh, 3, 3)
    @test_throws ArgumentError Meshes.coordinates(mesh, 1, 0)
end

@testset "IntervalMesh ExponentialStretching" begin
    FT = Float64
    @test_throws ArgumentError unit_intervalmesh(
        stretching = Meshes.ExponentialStretching(0.25),
        nelems = 0,
    )
    dom, mesh = unit_intervalmesh(
        stretching = Meshes.ExponentialStretching(0.25),
        nelems = 1,
    )
    @test Meshes.coordinates(mesh, 1, 1) == Geometry.ZPoint(0.0)
    @test Meshes.coordinates(mesh, 1, 2) ≈ Geometry.ZPoint(1.0)
    # Approx equal to gcm ref profile height fraction given the vert domain
    # height of 45 km and scale height of ~7.5 km
    # we normalize to unit length for easy comparison
    H = 7.5 / 45.0
    nelems = 10
    dom, mesh = unit_intervalmesh(
        stretching = Meshes.ExponentialStretching(H),
        nelems = nelems,
    )
    # Check against a reference profile
    ref_profile = [
        Geometry.ZPoint(0.0),
        Geometry.ZPoint(0.017514189444930356),
        Geometry.ZPoint(0.03708734253289887),
        Geometry.ZPoint(0.05926886424040023),
        Geometry.ZPoint(0.08486241436551151),
        Geometry.ZPoint(0.11511191590370247),
        Geometry.ZPoint(0.15209658312699387),
        Geometry.ZPoint(0.1997009518240897),
        Geometry.ZPoint(0.2665952891528509),
        Geometry.ZPoint(0.3800869206593279),
        Geometry.ZPoint(1.0000000000000000),
    ]
    @test Meshes.coordinates(mesh, 1, 1) == ref_profile[1]
    for eidx in 1:Meshes.nelements(mesh)
        @test Meshes.coordinates(mesh, eidx, 1) == ref_profile[eidx]
        @test Meshes.coordinates(mesh, eidx, 2) == ref_profile[eidx + 1]
    end
    # Check, without a reference profile, that the inverse transformation gives us equispaced points
    cmin = Geometry.component(dom.coord_min, 1)
    cmax = Geometry.component(dom.coord_max, 1)
    R = cmax - cmin
    h = H / R
    ζ(η, h) = (-expm1(-η / h)) / (-expm1(-1 / h))
    face_values = [mesh.faces[f].z for f in 1:length(mesh.faces)]
    inverse_faces = [ζ(η, h) for η in face_values]
    for idx in eachindex(inverse_faces)
        @test inverse_faces[idx] ==
              range(FT(0), FT(1); length = nelems + 1)[idx]
    end
end

@testset "IntervalMesh ExponentialStretching reverse" begin
    FT = Float64
    @test_throws ArgumentError unit_intervalmesh(
        stretching = Meshes.ExponentialStretching(0.25),
        nelems = 0,
        reverse_mode = true,
    )
    dom, mesh = unit_intervalmesh(
        stretching = Meshes.ExponentialStretching(0.25),
        nelems = 1,
        reverse_mode = true,
    )
    @test Meshes.coordinates(mesh, 1, 1) == Geometry.ZPoint(-1.0)
    @test Meshes.coordinates(mesh, 1, 2) ≈ Geometry.ZPoint(0.0)
    # Approx equal to gcm ref profile height fraction given the vert domain
    # height of 45 km and scale height of ~7.5 km
    # we normalize to unit length for easy comparison
    H = 7.5 / 45.0
    nelems = 10
    reverse_mode = true
    dom, mesh = unit_intervalmesh(
        stretching = Meshes.ExponentialStretching(7.5 / 45.0),
        nelems = nelems,
        reverse_mode = reverse_mode,
    )
    # Check against a reference profile
    ref_profile = [
        Geometry.ZPoint(-1.0000000000000000),
        Geometry.ZPoint(-0.3800869206593279),
        Geometry.ZPoint(-0.2665952891528509),
        Geometry.ZPoint(-0.1997009518240897),
        Geometry.ZPoint(-0.15209658312699387),
        Geometry.ZPoint(-0.11511191590370247),
        Geometry.ZPoint(-0.08486241436551151),
        Geometry.ZPoint(-0.05926886424040023),
        Geometry.ZPoint(-0.03708734253289887),
        Geometry.ZPoint(-0.017514189444930356),
        Geometry.ZPoint(0.0),
    ]
    @test Meshes.coordinates(mesh, 1, 1) == ref_profile[1]
    for eidx in 1:Meshes.nelements(mesh)
        @test Meshes.coordinates(mesh, eidx, 1) == ref_profile[eidx]
        @test Meshes.coordinates(mesh, eidx, 2) == ref_profile[eidx + 1]
    end
    # Check, without a reference profile, that the inverse transformation gives us equispaced points
    cmin = Geometry.component(dom.coord_min, 1)
    cmax = Geometry.component(dom.coord_max, 1)
    R = cmax - cmin
    h = H / R
    ζ(η, h) = (-expm1(-η / h)) / (-expm1(-1 / h))
    face_values = [-mesh.faces[f].z for f in 1:length(mesh.faces)]
    reverse!(face_values)
    inverse_faces = [ζ(η, h) for η in face_values]
    for idx in eachindex(inverse_faces)
        @test inverse_faces[idx] ==
              range(FT(0), FT(1); length = nelems + 1)[idx]
    end
end

@testset "IntervalMesh GeneralizedExponentialStretching" begin
    # Use normalized GCM profile heights (45km)
    @test_throws ArgumentError unit_intervalmesh(
        stretching = Meshes.GeneralizedExponentialStretching(
            0.02 / 45.0,
            7.0 / 45.0,
        ),
        nelems = 0,
    )
    @test_throws ArgumentError unit_intervalmesh(
        stretching = Meshes.GeneralizedExponentialStretching(
            0.02 / 45.0,
            7.0 / 45.0,
        ),
        nelems = 1,
    )
    # Test a gcm like configuration
    dom, mesh = unit_intervalmesh(
        stretching = Meshes.GeneralizedExponentialStretching(
            0.02 / 45.0,
            7.0 / 45.0,
        ),
        nelems = 45, # 46 face levels
    )
    # Test the mesh coordinates are eqv to dz_bottom
    @test Meshes.coordinates(mesh, 1, 1) == Geometry.ZPoint(0.0)
    @test isapprox(
        Meshes.coordinates(mesh, 1, 2),
        Geometry.ZPoint(0.02 / 45.0),
        rtol = 1e-3,
    )

    # Test the face element distance at the top of the domain is dz_top
    fₑ₋₁ = Geometry.component(Meshes.coordinates(mesh, 45, 1), 1)
    fₑ = Geometry.component(Meshes.coordinates(mesh, 45, 2), 1)
    # A residual tol of ~1e-1 or 1e-2 is fine for typical use cases
    @test fₑ - fₑ₋₁ ≈ 7.0 / 45.0 rtol = 1e-2
end


@testset "IntervalMesh GeneralizedExponentialStretching reverse" begin
    # Use normalized GCM profile heights (45km)
    @test_throws ArgumentError unit_intervalmesh(
        stretching = Meshes.GeneralizedExponentialStretching(
            7.0 / 45.0,
            0.02 / 45.0,
        ),
        nelems = 0,
        reverse_mode = true,
    )
    @test_throws ArgumentError unit_intervalmesh(
        stretching = Meshes.GeneralizedExponentialStretching(
            7.0 / 45.0,
            0.02 / 45.0,
        ),
        nelems = 1,
        reverse_mode = true,
    )
    # Test a gcm like configuration, for land
    nelems = 45
    dom, mesh = unit_intervalmesh(
        stretching = Meshes.GeneralizedExponentialStretching(
            7.0 / 45.0,
            0.02 / 45.0,
        ),
        nelems = nelems, # 46 face levels
        reverse_mode = true,
    )
    # Test the mesh coordinates are eqv to dz_bottom
    @test Meshes.coordinates(mesh, 1, 1) == Geometry.ZPoint(-1.0)
    # Test the face element distance at the top of the domain is dz_top
    @test isapprox(
        Meshes.coordinates(mesh, nelems, 1),
        Geometry.ZPoint(-Geometry.ZPoint(0.02 / 45.0).z),
        rtol = 1e-3,
    )
    fₑ₋₁ = Geometry.component(Meshes.coordinates(mesh, nelems, 1), 1)
    fₑ = Geometry.component(Meshes.coordinates(mesh, nelems, 2), 1)
    # A residual tol of ~1e-1 or 1e-2 is fine for typical use cases
    @test fₑ - fₑ₋₁ ≈ 0.02 / 45.0 rtol = 1e-2
end

@testset "IntervalMesh GeneralizedExponentialStretching default tolerance" begin
    # The default tolerance scales with the finer spacing, so both end
    # spacings match their targets on a unit domain, with a GCM-like profile
    # and a scaled 2 m soil column, where a residual tolerance of 1e-3 leaves
    # end-spacing errors of 0.3% and 1.2%.
    for (reverse_mode, dz_bottom, dz_top, nelems) in (
        (false, 0.02 / 45.0, 7.0 / 45.0, 45),
        (true, 0.3 / 2.0, 0.05 / 2.0, 20),
    )
        _, mesh = unit_intervalmesh(;
            stretching = Meshes.GeneralizedExponentialStretching(
                dz_bottom,
                dz_top,
            ),
            nelems,
            reverse_mode,
        )
        z = [f.z for f in mesh.faces]
        @test isapprox(z[2] - z[1], dz_bottom; rtol = 1e-6)
        @test isapprox(z[end] - z[end - 1], dz_top; rtol = 1e-6)
    end
end

@testset "IntervalMesh HyperbolicTangentStretching" begin
    # Use normalized GCM profile heights (75km)
    @test_throws ArgumentError unit_intervalmesh(
        stretching = Meshes.HyperbolicTangentStretching(0.03 / 75.0),
        nelems = 0,
    )
    @test_throws ArgumentError unit_intervalmesh(
        stretching = Meshes.HyperbolicTangentStretching(0.03 / 75.0),
        nelems = 1,
    )
    # A surface spacing of at least `H / nelems` has no stretched solution.
    @test_throws ArgumentError unit_intervalmesh(
        stretching = Meshes.HyperbolicTangentStretching(1 / 20),
        nelems = 20,
    )
    # Test a gcm like configuration
    nelems = 75
    dom, mesh = unit_intervalmesh(
        stretching = Meshes.HyperbolicTangentStretching(0.03 / 75.0),
        nelems = nelems, # 76 face levels
    )
    # Test the bottom and top of the mesh coordinates are correct
    @test Meshes.coordinates(mesh, 1, 1) == Geometry.ZPoint(0.0)
    @test Meshes.coordinates(mesh, nelems, 2) ≈ Geometry.ZPoint(1.0)
    # Test the interval of the mesh coordinates at the surface is the same as dz_surface
    @test isapprox(
        Meshes.coordinates(mesh, 1, 2),
        Geometry.ZPoint(0.03 / 75.0),
        rtol = 1e-3,
    )
end

@testset "IntervalMesh HyperbolicTangentStretching reverse" begin
    # Use normalized GCM profile heights (75km)
    @test_throws ArgumentError unit_intervalmesh(
        stretching = Meshes.HyperbolicTangentStretching(0.03 / 75.0),
        nelems = 0,
        reverse_mode = true,
    )
    @test_throws ArgumentError unit_intervalmesh(
        stretching = Meshes.HyperbolicTangentStretching(0.03 / 75.0),
        nelems = 1,
        reverse_mode = true,
    )
    # Test a gcm like configuration, for land
    nelems = 75
    dom, mesh = unit_intervalmesh(
        stretching = Meshes.HyperbolicTangentStretching(0.03 / 75.0),
        nelems = nelems, # 76 face levels
        reverse_mode = true,
    )
    # Test the bottom and top of the mesh coordinates are correct
    @test Meshes.coordinates(mesh, 1, 1) == Geometry.ZPoint(-1.0)
    @test Meshes.coordinates(mesh, nelems, 2) ≈ Geometry.ZPoint(0.0)
    # Test the interval of the mesh coordinates at the surface is the same as dz_surface
    @test isapprox(
        Meshes.coordinates(mesh, nelems, 1),
        Geometry.ZPoint(-0.03 / 75.0),
        rtol = 1e-3,
    )
end

@testset "Stretched IntervalMesh on shifted domains" begin
    # Every rule depends on the domain only through its height: shifting the
    # bounds translates the faces, and `reverse_mode` mirrors them about the
    # domain midpoint, with the surface spacing at the corresponding end.
    FT = Float64
    H = FT(30000)
    z_lo = FT(1000)
    base = Domains.IntervalDomain(
        Geometry.ZPoint(FT(0)),
        Geometry.ZPoint(H);
        boundary_names = (:bottom, :top),
    )
    shifted = Domains.IntervalDomain(
        Geometry.ZPoint(z_lo),
        Geometry.ZPoint(z_lo + H);
        boundary_names = (:bottom, :top),
    )
    faces_z(mesh) = [f.z for f in mesh.faces]
    for (stretch, rev_stretch, dz_surface, nelems) in (
        (
            Meshes.ExponentialStretching(FT(7500)),
            Meshes.ExponentialStretching(FT(7500)),
            nothing,
            10,
        ),
        (
            Meshes.GeneralizedExponentialStretching(FT(30), FT(5000)),
            Meshes.GeneralizedExponentialStretching(FT(5000), FT(30)),
            FT(30),
            20,
        ),
        (
            Meshes.HyperbolicTangentStretching(FT(30)),
            Meshes.HyperbolicTangentStretching(FT(30)),
            FT(30),
            20,
        ),
        (
            Meshes.BoundaryLayerHyperbolicTangentStretching(
                FT(30),
                FT(3000),
                FT(2200),
            ),
            Meshes.BoundaryLayerHyperbolicTangentStretching(
                FT(30),
                FT(3000),
                FT(2200),
            ),
            FT(30),
            43,
        ),
    )
        z_base = faces_z(Meshes.IntervalMesh(base, stretch; nelems))
        z_shift = faces_z(Meshes.IntervalMesh(shifted, stretch; nelems))
        @test z_shift[1] == z_lo
        @test z_shift[end] == z_lo + H
        @test z_shift ≈ z_lo .+ z_base
        if !isnothing(dz_surface)
            @test isapprox(z_shift[2] - z_shift[1], dz_surface; rtol = 1e-3)
        end

        z_rev = faces_z(
            Meshes.IntervalMesh(
                shifted,
                rev_stretch;
                nelems,
                reverse_mode = true,
            ),
        )
        @test z_rev[1] == z_lo
        @test z_rev[end] == z_lo + H
        @test z_rev ≈ (z_lo + H) .- reverse(z_base)
    end

    # Float32 solves reach the default tolerance and place the end spacings,
    # including a surface spacing four orders of magnitude below `H / nelems`.
    domain32 = Domains.IntervalDomain(
        Geometry.ZPoint(0.0f0),
        Geometry.ZPoint(45000.0f0);
        boundary_names = (:bottom, :top),
    )
    for (dz_surface, nelems) in ((30.0f0, 20), (0.2f0, 400))
        mesh32 = Meshes.IntervalMesh(
            domain32,
            Meshes.HyperbolicTangentStretching(dz_surface);
            nelems,
            FT_solve = Float32,
        )
        @test isapprox(mesh32.faces[2].z, dz_surface; rtol = 1e-5)
    end
    # The default tolerance of `GeneralizedExponentialStretching` is
    # `sqrt(eps(Float32)) * min(dz_bottom, dz_top)`, which places the end
    # spacings to within 2.1e-4 in a 12-configuration sweep.
    for (dz_bottom, dz_top, nelems) in ((30.0f0, 5000.0f0, 20), (700.0f0, 800.0f0, 63))
        mesh32 = Meshes.IntervalMesh(
            domain32,
            Meshes.GeneralizedExponentialStretching(dz_bottom, dz_top);
            nelems,
            FT_solve = Float32,
        )
        z32 = [f.z for f in mesh32.faces]
        @test isapprox(z32[2] - z32[1], dz_bottom; rtol = 1e-3)
        @test isapprox(z32[end] - z32[end - 1], dz_top; rtol = 1e-3)
    end

    # Faces solved in Float32 on a Float64 domain end at the domain bounds,
    # which Float32 does not represent.
    offset = Domains.IntervalDomain(
        Geometry.ZPoint(0.1),
        Geometry.ZPoint(60000.1);
        boundary_names = (:bottom, :top),
    )
    bl_stretch =
        Meshes.BoundaryLayerHyperbolicTangentStretching(28.0, 3000.0, 2200.0)
    for (stretch, nelems, reverse_mode) in (
        (Meshes.HyperbolicTangentStretching(30.0), 45, false),
        (Meshes.HyperbolicTangentStretching(30.0), 45, true),
        (Meshes.GeneralizedExponentialStretching(30.0, 5000.0), 45, false),
        (Meshes.GeneralizedExponentialStretching(5000.0, 30.0), 45, true),
        (bl_stretch, 75, false),
        (bl_stretch, 75, true),
    )
        mesh = Meshes.IntervalMesh(
            offset,
            stretch;
            nelems,
            FT_solve = Float32,
            reverse_mode,
        )
        @test mesh.faces[1] == offset.coord_min
        @test mesh.faces[end] == offset.coord_max
    end
end

@testset "IntervalMesh from explicit faces" begin
    domain = Domains.IntervalDomain(
        Geometry.ZPoint(0.0),
        Geometry.ZPoint(1000.0);
        boundary_names = (:bottom, :top),
    )
    @test_throws ArgumentError Meshes.IntervalMesh(domain, [Geometry.ZPoint(0.0)])
    mesh = Meshes.IntervalMesh(domain, Geometry.ZPoint.([0.0, 1000.0]))
    @test Meshes.nelements(mesh) == 1
end

@testset "IntervalMesh BoundaryLayerHyperbolicTangentStretching" begin
    FT = Float64
    z_max = FT(60000)
    domain = Domains.IntervalDomain(
        Geometry.ZPoint(FT(0)),
        Geometry.ZPoint(z_max);
        boundary_names = (:bottom, :top),
    )
    stretch = Meshes.BoundaryLayerHyperbolicTangentStretching(
        FT(28.0),
        FT(3000.0),
        FT(2200.0),
    )
    @test_throws ArgumentError Meshes.IntervalMesh(domain, stretch; nelems = 0)
    @test_throws ArgumentError Meshes.IntervalMesh(domain, stretch; nelems = 1)
    # Too few levels for the boundary layer, and too many for the background.
    @test_throws ArgumentError Meshes.IntervalMesh(domain, stretch; nelems = 61)
    @test_throws ArgumentError Meshes.IntervalMesh(domain, stretch; nelems = 110)
    err = try
        Meshes.IntervalMesh(domain, stretch; nelems = 25)
    catch e
        e
    end
    @test occursin("admits 62 ≤ nelems ≤ 109", err.msg)
    # A boundary layer this shallow leaves no admissible `nelems`.
    err = try
        Meshes.IntervalMesh(
            domain,
            Meshes.BoundaryLayerHyperbolicTangentStretching(28.0, 1000.0, 500.0);
            nelems = 100,
        )
    catch e
        e
    end
    @test occursin("admits no value of nelems", err.msg)

    adaption = ClimaCore.Hypsography.SLEVEAdaption(
        Geometry.ZPoint(FT(5500.0)),
        FT(0.7),
        FT(10.0),
    )
    z_top_pt = Geometry.ZPoint(z_max)

    # `find_gamma` has two roots for `nelems ≥ 98`.
    for nelems in (63, 75, 79, 87, 95, 105)
        mesh = Meshes.IntervalMesh(domain, stretch; nelems)
        @test Meshes.coordinates(mesh, 1, 1) == Geometry.ZPoint(FT(0))
        @test Meshes.coordinates(mesh, nelems, 2) == Geometry.ZPoint(z_max)
        @test isapprox(
            Meshes.coordinates(mesh, 1, 2).z,
            FT(28.0);
            rtol = 1e-5,
        )
        dz_ref = diff([f.z for f in mesh.faces])
        @test all(>(0), dz_ref)
        @test all(>(0), diff(dz_ref))
        @test minimum(dz_ref) >= FT(28.0) * (1 - 1e-5)
        @test isapprox(dz_ref[end], FT(3000.0); rtol = 1e-3)

        # Physical spacing stays monotonic and above 20 m under SLEVE over
        # 5500 m terrain.
        phys_faces = [
            ClimaCore.Hypsography.ref_z_to_physical_z(adaption, f, z_top_pt).z
            for f in mesh.faces
        ]
        dz_phys = diff(phys_faces)
        @test all(>(0), dz_phys)
        @test all(>(0), diff(dz_phys))
        @test minimum(dz_phys) >= FT(20.0)
    end

    # Float32 solves reach the default tolerance.
    mesh32 =
        Meshes.IntervalMesh(domain, stretch; nelems = 75, FT_solve = Float32)
    mesh64 = Meshes.IntervalMesh(domain, stretch; nelems = 75)
    @test isapprox(
        [f.z for f in mesh32.faces],
        [f.z for f in mesh64.faces];
        rtol = 1e-4,
    )
    # A 1 m surface spacing in a 45 km domain, where `γ ≈ 4.5`.
    domain45 = Domains.IntervalDomain(
        Geometry.ZPoint(FT(0)),
        Geometry.ZPoint(FT(45000));
        boundary_names = (:bottom, :top),
    )
    thin_stretch = Meshes.BoundaryLayerHyperbolicTangentStretching(
        FT(1),
        FT(2000),
        FT(1000),
    )
    mesh32 = Meshes.IntervalMesh(
        domain45,
        thin_stretch;
        nelems = 200,
        FT_solve = Float32,
    )
    mesh64 = Meshes.IntervalMesh(domain45, thin_stretch; nelems = 200)
    @test isapprox(
        [f.z for f in mesh32.faces],
        [f.z for f in mesh64.faces];
        rtol = 1e-3,
    )

    # A surface-to-top spacing ratio of 3e-8, for which `atanh` of the bound
    # on `tanh γ` overflows in Float32.
    tiny_stretch = Meshes.BoundaryLayerHyperbolicTangentStretching(
        1.0f-4,
        3000.0f0,
        2200.0f0,
    )
    domain32 = Domains.IntervalDomain(
        Geometry.ZPoint(0.0f0),
        Geometry.ZPoint(60000.0f0);
        boundary_names = (:bottom, :top),
    )
    mesh32 = Meshes.IntervalMesh(
        domain32,
        tiny_stretch;
        nelems = 200,
        FT_solve = Float32,
    )
    @test isapprox(mesh32.faces[2].z, 1.0f-4; rtol = 1e-4)

    # The argmin of a monotonic function is the end of the interval.
    @test Meshes._golden_section_argmin(identity, 0.0, 1.0) == 0.0
    @test Meshes._golden_section_argmin(-, 0.0, 1.0) == 1.0
    @test Meshes._golden_section_argmin(x -> (x - 0.3)^2, 0.0, 1.0) ≈ 0.3

    # `reverse_mode` mirrors the profile about the domain midpoint, for a
    # domain whose top is away from zero as well.
    soil_stretch = Meshes.BoundaryLayerHyperbolicTangentStretching(
        FT(0.05),
        FT(2.0),
        FT(1.0),
    )
    for (z_lo, z_hi) in ((FT(-10), FT(0)), (FT(-5), FT(5)))
        soil_domain = Domains.IntervalDomain(
            Geometry.ZPoint(z_lo),
            Geometry.ZPoint(z_hi);
            boundary_names = (:bottom, :top),
        )
        up = Meshes.IntervalMesh(soil_domain, soil_stretch; nelems = 20)
        down = Meshes.IntervalMesh(
            soil_domain,
            soil_stretch;
            nelems = 20,
            reverse_mode = true,
        )
        z_up = [f.z for f in up.faces]
        z_down = [f.z for f in down.faces]
        @test z_down[1] == z_lo
        @test z_down[end] == z_hi
        @test z_down ≈ z_lo .+ z_hi .- reverse(z_up)
        @test isapprox(z_hi - z_down[end - 1], FT(0.05); rtol = 1e-5)
        @test all(<(0), diff(diff(z_down)))
    end
end

@testset "Truncated IntervalMesh" begin
    FT = Float64
    nz = 55
    Δz_bottom = FT(30.0)
    Δz_top = FT(8000.0)
    z_bottom = FT(0.0)
    z_top_parent = FT(45000.0)
    z_top = FT(4000.0)
    stretch = Meshes.GeneralizedExponentialStretching(Δz_bottom, Δz_top)
    parent_domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(z_bottom),
        Geometry.ZPoint{FT}(z_top_parent),
        boundary_names = (:bottom, :top),
    )
    parent_mesh = Meshes.IntervalMesh(parent_domain, stretch, nelems = nz)
    trunc_domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(z_bottom),
        Geometry.ZPoint{FT}(z_top),
        boundary_names = (:bottom, :top),
    )
    trunc_mesh = Meshes.truncate_mesh(parent_mesh, trunc_domain)

    @test Meshes.coordinates(trunc_mesh, 1, 1) == Geometry.ZPoint(z_bottom)
    @test Meshes.coordinates(trunc_mesh, Meshes.nelements(trunc_mesh), 2) ==
          Geometry.ZPoint(z_top)

    # A raised bottom keeps the parent's faces and spacings between the bounds.
    faces_z(mesh) = [f.z for f in mesh.faces]
    z_parent = faces_z(parent_mesh)
    raised_domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(1000),
        Geometry.ZPoint{FT}(10000),
        boundary_names = (:bottom, :top),
    )
    raised = faces_z(Meshes.truncate_mesh(parent_mesh, raised_domain))
    k_bottom = findlast(≤(1000), z_parent)
    k_top = findfirst(≥(10000), z_parent)
    @test length(raised) == k_top - k_bottom + 1
    @test (raised[1], raised[end]) == (1000, 10000)
    @test isapprox(
        raised[2] - raised[1],
        z_parent[k_bottom + 1] - z_parent[k_bottom];
        rtol = 1e-6,
    )

    # A land mesh, with its fine end at the top, truncates from below the
    # surface.
    land_domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(-10),
        Geometry.ZPoint{FT}(0),
        boundary_names = (:bottom, :top),
    )
    land_mesh = Meshes.IntervalMesh(
        land_domain,
        Meshes.GeneralizedExponentialStretching(FT(1.5), FT(0.1));
        nelems = 20,
        reverse_mode = true,
    )
    trunc_land_domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(-10),
        Geometry.ZPoint{FT}(-2),
        boundary_names = (:bottom, :top),
    )
    trunc_land = Meshes.truncate_mesh(land_mesh, trunc_land_domain)
    z_land = faces_z(land_mesh)
    z_trunc = faces_z(trunc_land)
    k_top = findfirst(≥(-2), z_land)
    @test trunc_land.reverse_mode
    @test length(z_trunc) == k_top
    @test isapprox(
        z_trunc[end] - z_trunc[end - 1],
        z_land[k_top] - z_land[k_top - 1];
        rtol = 1e-3,
    )
end

@testset "IntervalMesh stretching parameter promotion" begin
    # Stretching parameters that are not already of the domain's floating point
    # type used to raise a `MethodError` (#1434).
    @test Meshes.GeneralizedExponentialStretching(30, 5000.0) ===
          Meshes.GeneralizedExponentialStretching(30.0, 5000.0)
    @test Meshes.GeneralizedExponentialStretching(30.0, 5000) ===
          Meshes.GeneralizedExponentialStretching(30.0, 5000.0)
    @test Meshes.GeneralizedExponentialStretching(30.0f0, 5000.0) ===
          Meshes.GeneralizedExponentialStretching(30.0, 5000.0)

    for FT in (Float32, Float64)
        domain = Domains.IntervalDomain(
            Geometry.ZPoint(FT(0)),
            Geometry.ZPoint(FT(30000));
            boundary_names = (:bottom, :top),
        )
        # the reproducer from the issue
        int_mesh = Meshes.IntervalMesh(
            domain,
            Meshes.GeneralizedExponentialStretching(30, 5000.0);
            nelems = 20,
        )
        float_mesh = Meshes.IntervalMesh(
            domain,
            Meshes.GeneralizedExponentialStretching(FT(30), FT(5000));
            nelems = 20,
        )
        @test int_mesh.faces == float_mesh.faces
        # the stored rule is converted to the domain's float type
        @test int_mesh.stretch ===
              Meshes.GeneralizedExponentialStretching{FT}(30, 5000)

        for (int_stretch, float_stretch) in (
            (
                Meshes.ExponentialStretching(7500),
                Meshes.ExponentialStretching(FT(7500)),
            ),
            (
                Meshes.HyperbolicTangentStretching(30),
                Meshes.HyperbolicTangentStretching(FT(30)),
            ),
            (
                Meshes.BoundaryLayerHyperbolicTangentStretching(30, 3000, 2200),
                Meshes.BoundaryLayerHyperbolicTangentStretching(
                    FT(30),
                    FT(3000),
                    FT(2200),
                ),
            ),
        )
            nelems =
                int_stretch isa Meshes.BoundaryLayerHyperbolicTangentStretching ?
                43 : 20
            int_mesh =
                Meshes.IntervalMesh(domain, int_stretch; nelems = nelems)
            float_mesh =
                Meshes.IntervalMesh(domain, float_stretch; nelems = nelems)
            @test int_mesh.faces == float_mesh.faces
            @test typeof(int_mesh.stretch) === typeof(float_stretch)
        end
    end
end

@testset "monotonic_check - dispatch" begin
    faces = range(Geometry.XPoint(0), Geometry.XPoint(10); length = 11)
    @test Meshes.monotonic_check(faces) == :no_check
    @test Meshes.monotonic_check(collect(faces)) == :no_check
end

@testset "monotonic_check" begin
    faces = range(Geometry.ZPoint(0), Geometry.ZPoint(10); length = 11)
    @test Meshes.monotonic_check(faces) == :pass # monotonic increasing
    @test Meshes.monotonic_check(collect(faces)) == :pass # monotonic increasing
    @test Meshes.monotonic_check(map(x -> x.z, faces)) == :pass # monotonic increasing

    faces = range(Geometry.ZPoint(0), Geometry.ZPoint(-10); length = 11)
    @test Meshes.monotonic_check(faces) == :pass # monotonic decreasing
    @test Meshes.monotonic_check(collect(faces)) == :pass # monotonic decreasing
    @test Meshes.monotonic_check(map(x -> x.z, faces)) == :pass # monotonic decreasing

    faces = map(z -> Geometry.ZPoint(1), 1:10)
    @test_throws ErrorException Meshes.monotonic_check(faces) # non-monotonic

    faces = range(Geometry.ZPoint(0), Geometry.ZPoint(10); length = 11)
    cfaces = collect(faces)
end
