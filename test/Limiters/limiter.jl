using ClimaCore: Fields, Domains, Geometry, Topologies, Meshes, Spaces, Limiters
using Test

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

@testset "Optimization-based limiter on a 1×1 2D domain space" begin
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

@testset "Optimization-based limiter on a 3×3 2D domain space" begin
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
    @test parent(ρq)[:, :, 1, 1] ≈ [0.0 0.0; 0.950005 0.950005] rtol = 10eps()
    @test parent(ρq)[:, :, 1, 2] ≈ [0.45 0.45001; 0.5 0.5] rtol = 10eps()
    @test parent(ρq)[9] ≈ 1 rtol = 10eps()
    # Check mass conservation after application of limiter
    @test sum(ρq) ≈ initial_Q_mass rtol = 10eps()
end
