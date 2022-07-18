using Test
using SparseArrays

import ClimaCore: ClimaCore, Domains, Meshes, Geometry

function unit_intervalmesh(;
    stretching = nothing,
    nelems = 20,
    periodic = false,
)
    if periodic
        dom = ClimaCore.Domains.IntervalDomain(
            ClimaCore.Geometry.ZPoint(0.0),
            ClimaCore.Geometry.ZPoint(1.0);
            periodic = true,
        )
    else
        dom = ClimaCore.Domains.IntervalDomain(
            ClimaCore.Geometry.ZPoint(0.0),
            ClimaCore.Geometry.ZPoint(1.0),
            boundary_names = (:left, :right),
        )
    end
    if stretching !== nothing
        return ClimaCore.Meshes.IntervalMesh(dom, stretching, nelems = nelems)
    else
        return ClimaCore.Meshes.IntervalMesh(dom, nelems = nelems)
    end
end

@testset "IntervalMesh" begin
    for nelems in (-1, 0)
        @test_throws ArgumentError unit_intervalmesh(nelems = nelems)
    end
    for nelems in (1, 3)
        mesh = unit_intervalmesh(nelems = nelems)
        @test length(mesh.faces) == nelems + 1 # number of faces is +1 elements
        @test Meshes.nelements(mesh) == nelems
        @test Meshes.elements(mesh) == UnitRange(1, nelems)
    end
    mesh = unit_intervalmesh(nelems = 2)
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
        mesh = unit_intervalmesh(nelems = nelems, periodic = true)
        @test length(mesh.faces) == nelems + 1 # number of faces is +1 elements
        @test Meshes.nelements(mesh) == nelems
        @test Meshes.elements(mesh) == UnitRange(1, nelems)
    end
    mesh = unit_intervalmesh(nelems = 2, periodic = true)
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
    @test_throws ArgumentError unit_intervalmesh(
        stretching = Meshes.ExponentialStretching(0.25),
        nelems = 0,
    )
    mesh = unit_intervalmesh(
        stretching = Meshes.ExponentialStretching(0.25),
        nelems = 1,
    )
    @test Meshes.coordinates(mesh, 1, 1) == Geometry.ZPoint(0.0)
    @test Meshes.coordinates(mesh, 1, 2) ≈ Geometry.ZPoint(1.0)
    # approx equal to gcm ref profile height fraction given the vert domain
    # ~7.5 km of 45 km vertical domain extent
    # we normalize to unit length for easy comparison
    mesh = unit_intervalmesh(
        stretching = Meshes.ExponentialStretching(7.5 / 45.0),
        nelems = 10,
    )
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
end

@testset "IntervalMesh GeneralizedExponentialStretching" begin
    # use normalized GCM profile heights (7.5km)
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
    # 45 km vertical domain height Δzₛ = 20m, Δzₜ = 7km
    mesh = unit_intervalmesh(
        stretching = Meshes.GeneralizedExponentialStretching(
            0.02 / 45.0,
            7.0 / 45.0,
        ),
        nelems = 45, # 46 face levels
    )
    # test the mesh coordinates are eqv to Δzₛ
    @test Meshes.coordinates(mesh, 1, 1) == Geometry.ZPoint(0.0)
    @test Meshes.coordinates(mesh, 1, 2) ≈ Geometry.ZPoint(0.02 / 45.0)

    # test the face element distance at the top of the domain is Δzₜ
    fₑ₋₁ = Geometry.component(Meshes.coordinates(mesh, 45, 1), 1)
    fₑ = Geometry.component(Meshes.coordinates(mesh, 45, 2), 1)
    # a residual tol of ~1e-1 or 1e-2 is fine for typical use cases
    @test fₑ - fₑ₋₁ ≈ 7.0 / 45.0 rtol = 1e-2
end
