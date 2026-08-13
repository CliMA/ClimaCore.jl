using Test
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Spaces,
    Quadratures,
    Topologies,
    DataLayouts

import ClimaCore  # for `pkgdir` below
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU

function make_2x2_rect_space(
    ::Type{FT};
    x1periodic = false,
    x2periodic = false,
    Nq = 3,
    context = ClimaComms.SingletonCommsContext(),
) where {FT}
    domain = Domains.RectangleDomain(
        Domains.IntervalDomain(
            Geometry.XPoint(FT(-1)),
            Geometry.XPoint(FT(1)),
            periodic = x1periodic,
            boundary_names = x1periodic ? nothing : (:west, :east),
        ),
        Domains.IntervalDomain(
            Geometry.YPoint(FT(-1)),
            Geometry.YPoint(FT(1)),
            periodic = x2periodic,
            boundary_names = x2periodic ? nothing : (:south, :north),
        ),
    )
    mesh = Meshes.RectilinearMesh(domain, 2, 2)
    topology = Topologies.Topology2D(context, mesh)
    quad = Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(topology, quad)
    return space
end

@testset "2x2 Rectangular Mesh Exact DSS Semantics (Non-Periodic)" begin
    FT = Float64
    Nq = 3
    space = make_2x2_rect_space(FT; x1periodic = false, x2periodic = false, Nq = Nq)
    field = zeros(FT, space)

    # Fill each element with distinct sequential integers: elem 1 has 1:9, elem 2 has 10:18, etc.
    # Node indexing in slab (i, j) with i along x1 (fast), j along x2 (slow):
    # Elem 1: [(-1,0) x (-1,0)] -> bottom-left
    # Elem 2: [(0,1) x (-1,0)]  -> bottom-right
    # Elem 3: [(-1,0) x (0,1)]  -> top-left
    # Elem 4: [(0,1) x (0,1)]   -> top-right
    arr_cpu = zeros(FT, 1, Nq, Nq, 1, 4)
    for elem in 1:4
        for j in 1:Nq, i in 1:Nq
            idx = (elem - 1) * (Nq * Nq) + (j - 1) * Nq + i
            arr_cpu[1, i, j, 1, elem] = FT(idx)
        end
    end
    copyto!(parent(field), arr_cpu)
    # Compare on the host: `parent(field)` is device memory under CUDA, and
    # indexing it elementwise would need `allowscalar`.
    raw_before = Array(parent(field))
    dss_buffer = Spaces.create_dss_buffer(field)
    Spaces.weighted_dss!(field, dss_buffer)
    arr = Array(parent(field))

    # 1. Unshared corner nodes:
    # Elem 1 bottom-left (i=1, j=1): index 1
    @test arr[1, 1, 1, 1, 1] == raw_before[1, 1, 1, 1, 1] == 1.0
    # Elem 2 bottom-right (i=3, j=1): index 12
    @test arr[1, 3, 1, 1, 2] == raw_before[1, 3, 1, 1, 2] == 12.0
    # Elem 3 top-left (i=1, j=3): index 25
    @test arr[1, 1, 3, 1, 3] == raw_before[1, 1, 3, 1, 3] == 25.0
    # Elem 4 top-right (i=3, j=3): index 36
    @test arr[1, 3, 3, 1, 4] == raw_before[1, 3, 3, 1, 4] == 36.0

    # 2. Interior nodes (e.g. i=2, j=2 in any element) are never shared:
    for elem in 1:4
        @test arr[1, 2, 2, 1, elem] == raw_before[1, 2, 2, 1, elem]
    end

    # 3. Two-element shared edge nodes:
    # Vertical interface between Elem 1 (i=3, j=2) and Elem 2 (i=1, j=2):
    # Elem 1 (3, 2) is idx 6; Elem 2 (1, 2) is idx 13
    expected_v_edge = (6.0 + 13.0) / 2.0
    @test arr[1, 3, 2, 1, 1] == arr[1, 1, 2, 1, 2] == expected_v_edge

    # Horizontal interface between Elem 1 (i=2, j=3) and Elem 3 (i=2, j=1):
    # Elem 1 (2, 3) is idx 8; Elem 3 (2, 1) is idx 20
    expected_h_edge = (8.0 + 20.0) / 2.0
    @test arr[1, 2, 3, 1, 1] == arr[1, 2, 1, 1, 3] == expected_h_edge

    # 4. Central cross node (shared by all 4 elements):
    # Elem 1 (3, 3) is idx 9
    # Elem 2 (1, 3) is idx 16
    # Elem 3 (3, 1) is idx 21
    # Elem 4 (1, 1) is idx 28
    expected_center = (9.0 + 16.0 + 21.0 + 28.0) / 4.0
    @test arr[1, 3, 3, 1, 1] == expected_center
    @test arr[1, 1, 3, 1, 2] == expected_center
    @test arr[1, 3, 1, 1, 3] == expected_center
    @test arr[1, 1, 1, 1, 4] == expected_center

    # 5. Allocation check. Host-side only: a CUDA launch allocates wrappers,
    #    so the sentinel is meaningless there (same split as
    #    `unit_sphere_dg_fluxes.jl`).
    allocs = @allocated Spaces.weighted_dss!(field, dss_buffer)
    ClimaComms.device() isa ClimaComms.CUDADevice ||
        !TU.allocation_checks_meaningful() ||
        @test allocs == 0
end

@testset "2x2 Rectangular Mesh Exact DSS Semantics (Fully Periodic)" begin
    FT = Float64
    Nq = 3
    space = make_2x2_rect_space(FT; x1periodic = true, x2periodic = true, Nq = Nq)
    field = zeros(FT, space)

    arr_cpu = zeros(FT, 1, Nq, Nq, 1, 4)
    for elem in 1:4
        for j in 1:Nq, i in 1:Nq
            idx = (elem - 1) * (Nq * Nq) + (j - 1) * Nq + i
            arr_cpu[1, i, j, 1, elem] = FT(idx)
        end
    end
    copyto!(parent(field), arr_cpu)
    # Compare on the host: `parent(field)` is device memory under CUDA, and
    # indexing it elementwise would need `allowscalar`.
    raw_before = Array(parent(field))
    dss_buffer = Spaces.create_dss_buffer(field)
    Spaces.weighted_dss!(field, dss_buffer)
    arr = Array(parent(field))

    # In a 2x2 fully periodic mesh, all 4 corner nodes of all elements wrap to meet at corners!
    # Central cross node is still shared by 4 elements:
    expected_center = (9.0 + 16.0 + 21.0 + 28.0) / 4.0
    @test arr[1, 3, 3, 1, 1] == expected_center
    @test arr[1, 1, 3, 1, 2] == expected_center
    @test arr[1, 3, 1, 1, 3] == expected_center
    @test arr[1, 1, 1, 1, 4] == expected_center

    # The outer 4 corners wrap to form one shared 4-node vertex:
    # Elem 1 (1,1)[1], Elem 2 (3,1)[12], Elem 3 (1,3)[25], Elem 4 (3,3)[36]
    expected_outer_corner = (1.0 + 12.0 + 25.0 + 36.0) / 4.0
    @test arr[1, 1, 1, 1, 1] == expected_outer_corner
    @test arr[1, 3, 1, 1, 2] == expected_outer_corner
    @test arr[1, 1, 3, 1, 3] == expected_outer_corner
    @test arr[1, 3, 3, 1, 4] == expected_outer_corner

    # Every element interior node (2, 2) remains unshared
    for elem in 1:4
        @test arr[1, 2, 2, 1, elem] == raw_before[1, 2, 2, 1, elem]
    end
end
