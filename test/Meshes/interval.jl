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
    @test_throws BoundsError Meshes.coordinates(mesh, 2, 3)
    @test_throws BoundsError Meshes.coordinates(mesh, 3, 3)
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
    @test Meshes.boundary_face_name(mesh, 1, 1) == nothing
    @test Meshes.is_boundary_face(mesh, 1, 2) == false
    @test Meshes.boundary_face_name(mesh, 1, 2) === nothing
    @test Meshes.is_boundary_face(mesh, 2, 1) == false
    @test Meshes.boundary_face_name(mesh, 2, 1) === nothing
    @test Meshes.is_boundary_face(mesh, 2, 2) == false
    @test Meshes.boundary_face_name(mesh, 2, 2) == nothing

    @test_throws BoundsError Meshes.coordinates(mesh, 0, 1)
    @test Meshes.coordinates(mesh, 1, 1) == Geometry.ZPoint(0.0)
    @test Meshes.coordinates(mesh, 2, 2) == Geometry.ZPoint(1.0)
    @test_throws BoundsError Meshes.coordinates(mesh, 2, 3)
    @test_throws BoundsError Meshes.coordinates(mesh, 3, 3)
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
    # approx equal to gcm ref profile height fraction given the vert domain
    # height of 45 km and scale height of ~7.5 km
    # we normalize to unit length for easy comparison
    H = 7.5 / 45.0
    nelems = 10
    dom, mesh = unit_intervalmesh(
        stretching = Meshes.ExponentialStretching(H),
        nelems = nelems,
    )
    # check against a reference profile
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
    for eidx in Meshes.nelements(mesh)
        @test Meshes.coordinates(mesh, eidx, 1) == ref_profile[eidx]
        @test Meshes.coordinates(mesh, eidx, 2) == ref_profile[eidx + 1]
    end
    # check, without a reference profile, that the inverse transformation gives us equispaced points
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
    # approx equal to gcm ref profile height fraction given the vert domain
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
    # check against a reference profile
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
    for eidx in Meshes.nelements(mesh)
        @test Meshes.coordinates(mesh, eidx, 1) == ref_profile[eidx]
        @test Meshes.coordinates(mesh, eidx, 2) == ref_profile[eidx + 1]
    end
    # check, without a reference profile, that the inverse transformation gives us equispaced points
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
    # use normalized GCM profile heights (45km)
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
    # test a gcm like configuration
    dom, mesh = unit_intervalmesh(
        stretching = Meshes.GeneralizedExponentialStretching(
            0.02 / 45.0,
            7.0 / 45.0,
        ),
        nelems = 45, # 46 face levels
    )
    # test the mesh coordinates are eqv to dz_bottom
    @test Meshes.coordinates(mesh, 1, 1) == Geometry.ZPoint(0.0)
    @test isapprox(Meshes.coordinates(mesh, 1, 2), Geometry.ZPoint(0.02 / 45.0), rtol=1e-3)

    # test the face element distance at the top of the domain is dz_top
    fₑ₋₁ = Geometry.component(Meshes.coordinates(mesh, 45, 1), 1)
    fₑ = Geometry.component(Meshes.coordinates(mesh, 45, 2), 1)
    # a residual tol of ~1e-1 or 1e-2 is fine for typical use cases
    @test fₑ - fₑ₋₁ ≈ 7.0 / 45.0 rtol = 1e-2
end


@testset "IntervalMesh GeneralizedExponentialStretching reverse" begin
    # use normalized GCM profile heights (45km)
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
    # test a gcm like configuration, for land
    nelems = 45
    dom, mesh = unit_intervalmesh(
        stretching = Meshes.GeneralizedExponentialStretching(
            7.0 / 45.0,
            0.02 / 45.0,
        ),
        nelems = nelems, # 46 face levels
        reverse_mode = true,
    )
    # test the mesh coordinates are eqv to dz_bottom
    @test Meshes.coordinates(mesh, 1, 1) == Geometry.ZPoint(-1.0)
    # test the face element distance at the top of the domain is dz_top
    @test isapprox(Meshes.coordinates(mesh, 1, nelems), Geometry.ZPoint(-Geometry.ZPoint(0.02 / 45.0).z), rtol = 1e-3)
    fₑ₋₁ = Geometry.component(Meshes.coordinates(mesh, nelems, 1), 1)
    fₑ = Geometry.component(Meshes.coordinates(mesh, nelems, 2), 1)
    # a residual tol of ~1e-1 or 1e-2 is fine for typical use cases
    @test fₑ - fₑ₋₁ ≈ 0.02 / 45.0 rtol = 1e-2
end

@testset "IntervalMesh HyperbolicTangentStretching" begin
    # use normalized GCM profile heights (75km)
    @test_throws ArgumentError unit_intervalmesh(
        stretching = Meshes.HyperbolicTangentStretching(0.03 / 75.0),
        nelems = 0,
    )
    @test_throws ArgumentError unit_intervalmesh(
        stretching = Meshes.HyperbolicTangentStretching(0.03 / 75.0),
        nelems = 1,
    )
    # test a gcm like configuration
    nelems = 75
    dom, mesh = unit_intervalmesh(
        stretching = Meshes.HyperbolicTangentStretching(0.03 / 75.0),
        nelems = nelems, # 76 face levels
    )
    # test the bottom and top of the mesh coordinates are correct
    @test Meshes.coordinates(mesh, 1, 1) == Geometry.ZPoint(0.0)
    @test Meshes.coordinates(mesh, 1, nelems + 1) ≈ Geometry.ZPoint(1.0)
    # test the interval of the mesh coordinates at the surface is the same as dz_surface
    @test isapprox(Meshes.coordinates(mesh, 1, 2), Geometry.ZPoint(0.03 / 75.0), rtol = 1e-3)
end

@testset "IntervalMesh HyperbolicTangentStretching reverse" begin
    # use normalized GCM profile heights (75km)
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
    # test a gcm like configuration, for land
    nelems = 75
    dom, mesh = unit_intervalmesh(
        stretching = Meshes.HyperbolicTangentStretching(0.03 / 75.0),
        nelems = nelems, # 76 face levels
        reverse_mode = true,
    )
    # test the bottom and top of the mesh coordinates are correct
    @test Meshes.coordinates(mesh, 1, 1) == Geometry.ZPoint(-1.0)
    @test Meshes.coordinates(mesh, 1, nelems + 1) ≈ Geometry.ZPoint(0.0)
    # test the interval of the mesh coordinates at the surface is the same as dz_surface
    @test isapprox(Meshes.coordinates(mesh, 1, nelems), Geometry.ZPoint(-0.03 / 75.0), rtol=1e-3)
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
    @test Meshes.coordinates(trunc_mesh, 1, length(trunc_mesh.faces)) ==
          Geometry.ZPoint(z_top)
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
