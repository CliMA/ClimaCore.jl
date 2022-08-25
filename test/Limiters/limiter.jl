using ClimaCore:
    DataLayouts, Fields, Domains, Geometry, Topologies, Meshes, Spaces, Limiters
using ClimaCore.RecursiveApply
using ClimaCore: slab
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


@testset "compute_bounds" begin
    FT = Float64
    lim_tol = 5e-14
    Nij = 4
    n1 = n2 = 5

    mesh = rectangular_mesh(
        n1,
        n2,
        false,
        false,
        x1min = 0.0,
        x1max = 2.0 * n1,
        x2min = 0.0,
        x2max = 3.0 * n2,
    )
    topology = Topologies.Topology2D(mesh)
    quad = Spaces.Quadratures.GLL{Nij}()
    space = Spaces.SpectralElementSpace2D(topology, quad)

    # Initialize fields
    ρ = map(coord -> exp(-coord.y), Fields.coordinate_field(space))
    q = map(coord -> (; coord.x, coord.y), Fields.coordinate_field(space))
    ρq = ρ .* q

    limiter = Limiters.QuasiMonotoneLimiter(ρq)
    Limiters.compute_bounds!(limiter, ρq, ρ)

    for h2 in 1:n2
        for h1 in 1:n1
            s = slab(limiter.q_bounds, h1 + n1 * (h2 - 1))
            q_min = s[1]
            q_max = s[2]
            @test q_min.x ≈ 2 * (h1 - 1)
            @test q_min.y ≈ 3 * (h2 - 1)
            @test q_max.x ≈ 2 * h1
            @test q_max.y ≈ 3 * h2

            s_nbr = slab(limiter.q_bounds_nbr, h1 + n1 * (h2 - 1))
            q_min = s_nbr[1]
            q_max = s_nbr[2]
            @test q_min.x ≈ 2 * max(h1 - 2, 0)
            @test q_min.y ≈ 3 * max(h2 - 2, 0)
            @test q_max.x ≈ 2 * min(h1 + 1, n1)
            @test q_max.y ≈ 3 * min(h2 + 1, n2)
        end
    end
end


@testset "apply_limit_slab!" begin
    q = DataLayouts.IJF{Tuple{Float64, Float64}, 5}(
        Float64[i + f for i in 1:5, j in 1:5, f in 1:2],
    )
    ρ = DataLayouts.IJF{Float64, 5}(
        Float64[j / 2 for i in 1:5, j in 1:5, f in 1:1],
    )
    ρq = ρ .⊠ q
    WJ = DataLayouts.IJF{Float64, 5}(ones(5, 5, 1))
    q_min = (3.2, 3.0)
    q_max = (5.2, 5.0)
    q_bounds = DataLayouts.IF{Tuple{Float64, Float64}, 2}(zeros(2, 2))
    q_bounds[1] = q_min
    q_bounds[2] = q_max


    ρq_new = deepcopy(ρq)
    Limiters.apply_limit_slab!(ρq_new, ρ, WJ, q_bounds, eps(Float64))


    q_new = RecursiveApply.rdiv.(ρq_new, ρ)
    for j in 1:5, i in 1:5
        @test q_min[1] <= q_new[i, j][1] <= q_max[1]
        @test q_min[2] <= q_new[i, j][2] <= q_max[2]
    end
    @test sum(ρq_new.:1 .* WJ) ≈ sum(ρq.:1 .* WJ)
    @test sum(ρq_new.:2 .* WJ) ≈ sum(ρq.:2 .* WJ)
end



@testset "Optimization-based limiter on a 1×1 elem 2D domain space" begin
    FT = Float64
    lim_tol = 5e-14
    Nij = 2

    mesh = rectangular_mesh(1, 1, false, false)
    topology = Topologies.Topology2D(mesh)
    quad = Spaces.Quadratures.GLL{Nij}()
    space = Spaces.SpectralElementSpace2D(topology, quad)

    # Initialize fields
    ρ = ones(space)
    q = ones(space)
    parent(q)[:, :, 1, 1] = [-0.2 0.00001; 1.1 1]

    ρq = ρ .* q

    limiter = Limiters.QuasiMonotoneLimiter(ρq)
    initial_Q_mass = sum(ρq)

    # Initialize variables needed for limiters
    q_ref = ones(space)
    parent(q_ref)[:, :, 1, 1] = [0 0.00001; 1 1]
    ρq_ref = ρ .* q_ref

    Limiters.compute_bounds!(limiter, ρq_ref, ρ)
    Limiters.apply_limiter!(ρq, ρ, limiter)

    @test parent(ρq)[:, :, 1, 1] ≈ [0.0 0.0; 0.950005 0.950005] rtol = 10eps()
    # Check mass conservation after application of limiter
    @test sum(ρq) ≈ initial_Q_mass rtol = 10eps()
end

@testset "Optimization-based limiter on a 3×3 elem 2D domain space" begin
    FT = Float64
    Nij = 5

    mesh = rectangular_mesh(3, 3, false, false)
    topology = Topologies.Topology2D(mesh)
    quad = Spaces.Quadratures.GLL{Nij}()
    space = Spaces.SpectralElementSpace2D(topology, quad)

    # Initialize fields
    ρ = map(coord -> exp(-coord.y), Fields.coordinate_field(space))
    q = map(
        coord -> (x = 1.2 * coord.x, y = 1.5 * coord.y),
        Fields.coordinate_field(space),
    )
    ρq = ρ .⊠ q
    q_ref =
        map(coord -> (x = coord.x, y = coord.y), Fields.coordinate_field(space))
    ρq_ref = ρ .⊠ q_ref

    total_ρq = sum(ρq)

    limiter = Limiters.QuasiMonotoneLimiter(ρq)

    Limiters.compute_bounds!(limiter, ρq_ref, ρ)
    Limiters.apply_limiter!(ρq, ρ, limiter)
    q = RecursiveApply.rdiv.(ρq, ρ)

    @test sum(ρq.x) ≈ total_ρq.x
    @test sum(ρq.y) ≈ total_ρq.y
    @test all(0 .<= parent(ρq) .<= 1)
end


@testset "Optimization-based limiter on a doubly-periodic 3x3×3 elem 2D x 1D hybrid domain space" begin
    FT = Float64
    Nij = 5

    n1 = n2 = 3
    n3 = 3

    horzspace, hv_center_space, hv_face_space = hvspace_3D(
        FT,
        xlim = (0, 1),
        ylim = (0, 1),
        zlim = (0, 2),
        Nij = Nij,
        xelems = n1,
        yelems = n2,
        zelems = n3,
    )

    # Initialize fields
    ρ = map(
        coord -> exp(-coord.x - coord.z),
        Fields.coordinate_field(hv_center_space),
    )
    q = map(
        coord -> (x = 1.2 * coord.x, y = 1.5 * coord.y),
        Fields.coordinate_field(hv_center_space),
    )
    ρq = ρ .⊠ q
    q_ref = map(
        coord -> (x = coord.x, y = coord.y),
        Fields.coordinate_field(hv_center_space),
    )
    ρq_ref = ρ .⊠ q_ref

    total_ρq = sum(ρq)

    limiter = Limiters.QuasiMonotoneLimiter(ρq)

    Limiters.compute_bounds!(limiter, ρq_ref, ρ)
    Limiters.apply_limiter!(ρq, ρ, limiter)
    q = RecursiveApply.rdiv.(ρq, ρ)

    @test sum(ρq.x) ≈ total_ρq.x
    @test sum(ρq.y) ≈ total_ρq.y
    @test all(0 .<= parent(ρq) .<= 1)
end
