#=
julia --project=test
using Revise; include(joinpath("test", "Limiters", "limiter.jl"))
=#
using ClimaComms
ClimaComms.@import_required_backends
using ClimaCore:
    DataLayouts,
    Fields,
    Domains,
    Geometry,
    Topologies,
    Meshes,
    Spaces,
    Limiters,
    Quadratures
import ClimaCore.DataLayouts: slab_index
using ClimaCore: slab
using Test

si = slab_index
# 2D mesh setup
function rectangular_mesh_space(
    n1,
    n2,
    x1periodic,
    x2periodic;
    FT = Float32,
    x1min = 0.0,
    x1max = 1.0,
    x2min = 0.0,
    x2max = 1.0,
    Nij,
    device = ClimaComms.device(),
    comms_ctx = ClimaComms.SingletonCommsContext(device),
)
    domain = Domains.RectangleDomain(
        Domains.IntervalDomain(
            Geometry.XPoint{FT}(x1min),
            Geometry.XPoint{FT}(x1max);
            periodic = x1periodic,
            boundary_names = boundary =
                x1periodic ? nothing : (:west, :east),
        ),
        Domains.IntervalDomain(
            Geometry.YPoint{FT}(x2min),
            Geometry.YPoint{FT}(x2max);
            periodic = x2periodic,
            boundary_names = boundary =
                x2periodic ? nothing : (:south, :north),
        ),
    )
    mesh = Meshes.RectilinearMesh(domain, n1, n2)
    topology = Topologies.Topology2D(comms_ctx, mesh)
    quad = Quadratures.GLL{Nij}()
    return Spaces.SpectralElementSpace2D(topology, quad)
end

# 2D x 1D hybrid function space setup
function hvspace_3D(
    FT = Float32;
    xlim = (-2π, 2π),
    ylim = (-2π, 2π),
    zlim = (0, 4π),
    xelems = 4,
    yelems = 8,
    zelems = 16,
    Nij = 4,
    device = ClimaComms.device(),
    comms_ctx = ClimaComms.SingletonCommsContext(device),
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
    horztopology = Topologies.Topology2D(comms_ctx, horzmesh)

    zdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(zdomain, nelems = zelems)
    z_topology = Topologies.IntervalTopology(comms_ctx, vertmesh)

    vert_center_space = Spaces.CenterFiniteDifferenceSpace(z_topology)

    quad = Quadratures.GLL{Nij}()
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (horzspace, hv_center_space, hv_face_space)
end


@testset "compute_bounds" begin

    Nij = 4
    n1 = n2 = 5
    device = ClimaComms.device()
    comms_ctx = ClimaComms.SingletonCommsContext(device)

    for FT in (Float64, Float32)
        lim_tol = FT(5e-14)
        space = rectangular_mesh_space(
            n1,
            n2,
            false,
            false;
            FT = FT,
            x1min = 0.0,
            x1max = 2.0 * n1,
            x2min = 0.0,
            x2max = 3.0 * n2,
            Nij,
            comms_ctx,
        )

        # Initialize fields
        ρ = map(coord -> exp(-coord.y), Fields.coordinate_field(space))
        q = map(coord -> (; coord.x, coord.y), Fields.coordinate_field(space))
        ρq = ρ .* q

        limiter = Limiters.QuasiMonotoneLimiter(ρq)
        Limiters.compute_element_bounds!(limiter, ρq, ρ)

        S = map(Iterators.product(1:n1, 1:n2)) do (h1, h2)
            (h1, h2, slab(limiter.q_bounds, h1 + n1 * (h2 - 1)))
        end
        ClimaComms.allowscalar(device) do
            @test all(map(T -> T[3][si(1)].x ≈ 2 * (T[1] - 1), S)) # q_min
            @test all(map(T -> T[3][si(1)].y ≈ 3 * (T[2] - 1), S)) # q_min
            @test all(map(T -> T[3][si(2)].x ≈ 2 * T[1], S)) # q_max
            @test all(map(T -> T[3][si(2)].y ≈ 3 * T[2], S)) # q_max
        end

        Limiters.compute_neighbor_bounds_local!(limiter, ρ)
        SN = map(Iterators.product(1:n1, 1:n2)) do (h1, h2)
            (h1, h2, slab(limiter.q_bounds_nbr, h1 + n1 * (h2 - 1)))
        end
        ClimaComms.allowscalar(device) do
            @test all(map(T -> T[3][si(1)].x ≈ 2 * max(T[1] - 2, 0), SN))  # q_min
            @test all(map(T -> T[3][si(1)].y ≈ 3 * max(T[2] - 2, 0), SN))  # q_min
            @test all(map(T -> T[3][si(2)].x ≈ 2 * min(T[1] + 1, n1), SN))  # q_max
            @test all(map(T -> T[3][si(2)].y ≈ 3 * min(T[2] + 1, n2), SN))  # q_max
        end
    end
end


@testset "apply_limit_slab!" begin
    for FT in (Float64, Float32)
        q = DataLayouts.IJF{Tuple{FT, FT}, 5}(
            FT[i + f for i in 1:5, j in 1:5, f in 1:2],
        )
        ρ = DataLayouts.IJF{FT, 5}(FT[j / 2 for i in 1:5, j in 1:5, f in 1:1])
        ρq = ρ .* q
        WJ = DataLayouts.IJF{FT, 5}(ones(FT, 5, 5, 1))
        q_min = (FT(3.2), FT(3.0))
        q_max = (FT(5.2), FT(5.0))
        q_bounds = DataLayouts.IF{Tuple{FT, FT}, 2}(zeros(FT, 2, 2))
        q_bounds[si(1)] = q_min
        q_bounds[si(2)] = q_max


        ρq_new = deepcopy(ρq)
        Limiters.apply_limit_slab!(ρq_new, ρ, WJ, q_bounds, eps(FT))

        q_new = ρq_new ./ ρ
        for j in 1:5, i in 1:5
            @test q_min[1] <= q_new[si(i, j)][1] <= q_max[1]
            @test q_min[2] <= q_new[si(i, j)][2] <= q_max[2]
        end
        @test sum(ρq_new.:1 .* WJ) ≈ sum(ρq.:1 .* WJ)
        @test sum(ρq_new.:2 .* WJ) ≈ sum(ρq.:2 .* WJ)
    end
end

@testset "Optimization-based limiter on a 1×1 elem 2D domain space" begin

    Nij = 2
    device = ClimaComms.device()
    comms_ctx = ClimaComms.SingletonCommsContext(device)
    gpu_broken = device isa ClimaComms.CUDADevice

    for FT in (Float32,)
        lim_tol = FT(5e-14)
        space =
            rectangular_mesh_space(1, 1, false, false; FT = FT, Nij, comms_ctx)

        # Initialize fields
        ρ = ones(FT, space)
        q = ones(FT, space)
        parent(q)[:, :, 1, 1] = [FT(-0.2) FT(0.00001); FT(1.1) FT(1)]

        ρq = @. ρ .* q

        limiter = Limiters.QuasiMonotoneLimiter(ρq)
        initial_Q_mass = sum(ρq)

        # Initialize variables needed for limiters
        q_ref = ones(FT, space)
        parent(q_ref)[:, :, 1, 1] = [FT(0) FT(0.00001); FT(1) FT(1)]
        ρq_ref = ρ .* q_ref

        Limiters.compute_bounds!(limiter, ρq_ref, ρ)
        Limiters.apply_limiter!(ρq, ρ, limiter)

        @test Array(parent(ρq))[:, :, 1, 1] ≈
              [FT(0.0) FT(0.0); FT(0.950005) FT(0.950005)] rtol = 10eps(FT)
        # Check mass conservation after application of limiter
        @test sum(ρq) ≈ initial_Q_mass rtol = 10eps(FT)
    end
end

@testset "Optimization-based limiter on a 3×3 elem 2D domain space" begin

    Nij = 5
    device = ClimaComms.device()
    comms_ctx = ClimaComms.SingletonCommsContext(device)

    for FT in (Float64, Float32)
        space =
            rectangular_mesh_space(3, 3, false, false; FT = FT, Nij, comms_ctx)

        x_scale = FT(1.2)
        y_scale = FT(1.5)

        # Initialize fields
        ρ = map(coord -> exp(-coord.y), Fields.coordinate_field(space))
        coords = Fields.coordinate_field(space)
        q₀(coords, x_scale, y_scale) =
            (x = x_scale * coords.x, y = y_scale * coords.y)
        q = @. q₀(coords, x_scale, y_scale)
        ρq = ρ .* q
        q_ref = map(
            coord -> (x = coord.x, y = coord.y),
            Fields.coordinate_field(space),
        )
        ρq_ref = ρ .* q_ref

        total_ρq = (; x = sum(ρq.x), y = sum(ρq.y))

        limiter = Limiters.QuasiMonotoneLimiter(ρq)

        Limiters.compute_bounds!(limiter, ρq_ref, ρ)
        Limiters.apply_limiter!(ρq, ρ, limiter)
        q = ρq ./ ρ

        @test sum(ρq.x) ≈ total_ρq.x
        @test sum(ρq.y) ≈ total_ρq.y
        @test maximum(Array(parent(ρq))) ≈ 1 rtol = eps(FT)
        @test minimum(Array(parent(ρq))) ≈ 0 rtol = eps(FT)
    end
end

@testset "Optimization-based limiter on a doubly-periodic 3x3×3 elem 2D x 1D hybrid domain space" begin

    Nij = 5
    n1 = n2 = 3
    n3 = 3
    device = ClimaComms.device()
    comms_ctx = ClimaComms.SingletonCommsContext(device)

    for FT in (Float64, Float32)
        horzspace, hv_center_space, hv_face_space = hvspace_3D(
            FT;
            xlim = (FT(0), FT(1)),
            ylim = (FT(0), FT(1)),
            zlim = (FT(0), FT(2)),
            Nij = Nij,
            xelems = n1,
            yelems = n2,
            zelems = n3,
            comms_ctx,
        )

        x_scale = FT(1.2)
        y_scale = FT(1.5)

        # Initialize fields
        ρ = map(
            coord -> exp(-coord.x - coord.z),
            Fields.coordinate_field(hv_center_space),
        )
        coords = Fields.coordinate_field(hv_center_space)
        q₀(coords, x_scale, y_scale) =
            (x = x_scale * coords.x, y = y_scale * coords.y)
        q = @. q₀(coords, x_scale, y_scale)
        ρq = ρ .* q
        q_ref = map(
            coord -> (x = coord.x, y = coord.y),
            Fields.coordinate_field(hv_center_space),
        )
        ρq_ref = ρ .* q_ref

        total_ρq = (; x = sum(ρq.x), y = sum(ρq.y))

        limiter = Limiters.QuasiMonotoneLimiter(ρq)

        Limiters.compute_bounds!(limiter, ρq_ref, ρ)
        Limiters.apply_limiter!(ρq, ρ, limiter)
        q = ρq ./ ρ

        @test sum(ρq.x) ≈ total_ρq.x
        @test sum(ρq.y) ≈ total_ρq.y
        @test maximum(Array(parent(ρq))) ≤ 1
        @test minimum(Array(parent(ρq))) ≥ 0
    end
end
