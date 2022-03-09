using ClimaCore: Fields, Domains, Geometry, Topologies, Meshes, Spaces, Limiters
using Test

# 2D mesh setup
function rectangular_mesh(
    n1,
    n2,
    x1periodic,
    x2periodic;
    x1min = 0.0,
    x1max = 1.0,
    x2min = 0.0,
    x2max = 1.0,
)
    domain = Domains.RectangleDomain(
        Domains.IntervalDomain(
            Geometry.XPoint(x1min),
            Geometry.XPoint(x1max);
            periodic = x1periodic,
            boundary_names = boundary =
                x1periodic ? nothing : (:west, :east),
        ),
        Domains.IntervalDomain(
            Geometry.YPoint(x2min),
            Geometry.YPoint(x2max);
            periodic = x2periodic,
            boundary_names = boundary =
                x2periodic ? nothing : (:south, :north),
        ),
    )
    return Meshes.RectilinearMesh(domain, n1, n2)
end

# 2D x 1D hybrid function space setup
function hvspace_3D(
    FT = Float64;
    xlim = (-2π, 2π),
    ylim = (-2π, 2π),
    zlim = (0, 4π),
    xelems = 4,
    yelems = 8,
    zelems = 16,
    Nij = 4,
)

    xdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1]),
        Geometry.XPoint{FT}(xlim[2]),
        periodic = true,
    )
    ydomain = Domains.IntervalDomain(
        Geometry.YPoint{FT}(ylim[1]),
        Geometry.YPoint{FT}(ylim[2]),
        periodic = true,
    )

    horzdomain = Domains.RectangleDomain(xdomain, ydomain)
    horzmesh = Meshes.RectilinearMesh(horzdomain, xelems, yelems)
    horztopology = Topologies.Topology2D(horzmesh)

    zdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(zdomain, nelems = zelems)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

    quad = Spaces.Quadratures.GLL{Nij}()
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (horzspace, hv_center_space, hv_face_space)
end

@testset "Optimization-based limiter on a 1×1 elem 2D domain space" begin
    FT = Float64
    lim_tol = 5e-14
    Nij = 2
    n1 = n2 = 1

    mesh = rectangular_mesh(1, 1, false, false)
    topology = Topologies.Topology2D(mesh)
    quad = Spaces.Quadratures.GLL{Nij}()
    space = Spaces.SpectralElementSpace2D(topology, quad)

    # Initialize fields
    ρ = ones(space)
    q = ones(space)
    parent(q)[:, :, 1, 1] = [-0.2 0.00001; 1.1 1]

    ρq = ρ .* q
    initial_Q_mass = sum(ρq)

    # Initialize variables needed for limiters
    n_elems = Topologies.nlocalelems(space.topology)
    min_ρq = zeros(n_elems)
    max_ρq = ones(n_elems)

    Limiters.quasimonotone_limiter!(ρq, ρ, min_ρq, max_ρq, rtol = lim_tol)
    @test parent(ρq)[:, :, 1, 1] ≈ [0.0 0.0; 0.950005 0.950005] rtol = 10eps()
    # Check mass conservation after application of limiter
    @test sum(ρq) ≈ initial_Q_mass rtol = 10eps()
end

@testset "Optimization-based limiter on a 3×3 elem 2D domain space" begin
    FT = Float64
    lim_tol = 5e-14
    Nij = 2
    n1 = n2 = 1

    mesh = rectangular_mesh(3, 3, false, false)
    topology = Topologies.Topology2D(mesh)
    quad = Spaces.Quadratures.GLL{Nij}()
    space = Spaces.SpectralElementSpace2D(topology, quad)

    # Initialize fields
    ρ = ones(space)
    q = ones(space)
    parent(q)[:, :, 1, 1] = [-0.2 0.00001; 1.1 1]
    parent(q)[:, :, 1, 2] = [-0.2 0.00001; 1.1 1]

    ρq = ρ .* q
    initial_Q_mass = sum(ρq)

    # Initialize variables needed for limiters
    n_elems = Topologies.nlocalelems(space.topology)
    min_ρq = zeros(n_elems)
    max_ρq = ones(n_elems)
    max_ρq[2] = 0.5

    Limiters.quasimonotone_limiter!(ρq, ρ, min_ρq, max_ρq, rtol = lim_tol)
    # Check elem 1 values, with min_ρq = 0, max_ρq = 1
    @test parent(ρq)[:, :, 1, 1] ≈ [0.0 0.0; 0.950005 0.950005] rtol = 10eps()
    # Check elem 2 values, with min_ρq = 0, max_ρq = 0.5
    @test parent(ρq)[:, :, 1, 2] ≈ [0.45 0.45001; 0.5 0.5] rtol = 10eps()
    # Check elem 3, vertex 1 value that was between min_ρq = 0, max_ρq = 1 bounds
    @test parent(ρq)[9] ≈ 1 rtol = 10eps()
    # Check mass conservation after application of limiter
    @test sum(ρq) ≈ initial_Q_mass rtol = 10eps()
end

@testset "Optimization-based limiter on a doubly-periodic 1x1×1 elem 2D x 1D hybrid domain space" begin
    FT = Float64
    lim_tol = 5e-14
    Nij = 2
    n1 = n2 = n3 = 1

    horzspace, hv_center_space, hv_face_space =
        hvspace_3D(FT, Nij = Nij, xelems = n1, yelems = n2, zelems = n3)

    # Initialize fields
    ρ = ones(hv_center_space)
    q = ones(hv_center_space)
    parent(q)[1, :, :, 1, 1] = [-0.2 0.00001; 1.1 1]

    ρq = ρ .* q
    initial_Q_mass = sum(ρq)

    # Initialize variables needed for limiters
    horz_n_elems = Topologies.nlocalelems(horzspace.topology)
    min_ρq = zeros(horz_n_elems, n3)
    max_ρq = ones(horz_n_elems, n3)

    Limiters.quasimonotone_limiter!(ρq, ρ, min_ρq, max_ρq, rtol = lim_tol)
    @test parent(ρq)[1, :, :, 1, 1] ≈ [0.0 0.0; 0.950005 0.950005] rtol =
        10eps()

    # Check mass conservation after application of limiter
    @test sum(ρq) ≈ initial_Q_mass rtol = 10eps()
end

@testset "Optimization-based limiter on a doubly-periodic 3x3×3 elem 2D x 1D hybrid domain space" begin
    FT = Float64
    lim_tol = 5e-14
    Nij = 2
    n1 = n2 = 3
    n3 = 3

    horzspace, hv_center_space, hv_face_space =
        hvspace_3D(FT, Nij = Nij, xelems = n1, yelems = n2, zelems = n3)

    # Initialize fields
    ρ = ones(hv_center_space)
    q = ones(hv_center_space)
    parent(q)[1, :, :, 1, 1] = [-0.2 0.00001; 1.1 1]
    parent(q)[1, :, :, 1, 2] = [-0.2 0.00001; 1.1 1]
    parent(q)[2, :, :, 1, 5] = [-0.2 0.00001; 1.1 1]
    parent(q)[3, :, :, 1, 7] = [1 1; 1 1]

    ρq = ρ .* q
    initial_Q_mass = sum(ρq)

    # Initialize variables needed for limiters
    horz_n_elems = Topologies.nlocalelems(horzspace.topology)
    min_ρq = zeros(horz_n_elems, n3)
    max_ρq = ones(horz_n_elems, n3)
    # Change max only for level 1, elem 2
    max_ρq[2, 1] = 0.5

    Limiters.quasimonotone_limiter!(ρq, ρ, min_ρq, max_ρq, rtol = lim_tol)
    # Check level 1, elem 1 values, with min_ρq = 0, max_ρq = 1
    @test parent(ρq)[1, :, :, 1, 1] ≈ [0.0 0.0; 0.950005 0.950005] rtol =
        10eps()
    # Check level 1, elem 2 values, with min_ρq = 0, max_ρq = 0.5
    @test parent(ρq)[1, :, :, 1, 2] ≈ [0.45 0.45001; 0.5 0.5] rtol = 10eps()
    # Check level 2, elem 5 values, with min_ρq = 0, max_ρq = 1
    @test parent(ρq)[2, :, :, 1, 5] ≈ [0.0 0.0; 0.950005 0.950005] rtol =
        10eps()
    # Check level 3, elem 7 values, with unvaried entries
    @test parent(ρq)[3, :, :, 1, 7] ≈ [1 1; 1 1] rtol = 10eps()
    # Check mass conservation after application of limiter
    @test sum(ρq) ≈ initial_Q_mass rtol = 10eps()
end
