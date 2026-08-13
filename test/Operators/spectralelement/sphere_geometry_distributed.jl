using LinearAlgebra, IntervalSets
import ClimaCore:
    Domains,
    Topologies,
    Meshes,
    Spaces,
    Geometry,
    Operators,
    Fields,
    Quadratures

using Test

using StaticArrays, LinearAlgebra
using Logging
using ClimaComms
ClimaComms.@import_required_backends

function rotational_field(space, α0 = 45.0)
    coords = Fields.coordinate_field(space)
    map(coords) do coord
        ϕ = coord.lat
        λ = coord.long
        uu = (cosd(α0) * cosd(ϕ) + sind(α0) * cosd(λ) * sind(ϕ))
        uv = -sind(α0) * sind(λ)
        Geometry.UVVector(uu, uv)
    end
end

@testset "Spherical geometry properties" begin
    context = ClimaComms.MPICommsContext()
    pid, nprocs = ClimaComms.init(context)
    iamroot = ClimaComms.iamroot(context)
    # Log output only from root process
    logger_stream = iamroot ? stderr : devnull
    prev_logger = global_logger(ConsoleLogger(logger_stream, Logging.Info))
    atexit() do
        global_logger(prev_logger)
    end

    if iamroot
        println("running sphere_geometry_distributed using $nprocs processes")
    end
    # Test different combinations of odd/even to ensure pole is correctly
    # handled
    for Ne in (4, 5), Nq in (4, 5)
        FT = Float64
        radius = FT(3)

        domain = Domains.SphereDomain(radius)
        mesh = Meshes.EquiangularCubedSphere(domain, Ne)
        grid_topology = Topologies.Topology2D(
            context,
            mesh,
            Topologies.spacefillingcurve(mesh),
        )
        quad = Quadratures.GLL{Nq}()
        space = Spaces.SpectralElementSpace2D(grid_topology, quad)
        # For comparison with serial results
        grid_topology_serial =
            Topologies.Topology2D(ClimaComms.SingletonCommsContext(), mesh)
        space_serial = Spaces.SpectralElementSpace2D(grid_topology_serial, quad)

        surface_area = sum(ones(space))
        surface_area_serial = sum(ones(space_serial))
        @test surface_area ≈ 4pi * radius^2 rtol = 1e-3
        # Check if distributed and serial results match
        @test surface_area ≈ surface_area_serial
        for α in [0.0, 45.0, 90.0]
            # Test on divergence
            div = Operators.Divergence()
            u = rotational_field(space, α)
            divu2 = Spaces.weighted_dss!(div.(deepcopy(u)))
            norm_divu2 = norm(divu2)

            u_serial = rotational_field(space_serial, α)
            divu_serial = Spaces.weighted_dss!(div.(u_serial))
            norm_divu_serial = norm(divu_serial)
            @test norm_divu2 < 1e-2
            # Check if distributed and serial results match
            @test norm_divu2 ≈ norm_divu_serial

            # Test dss on UVcoordinates
            uu2 = Spaces.weighted_dss!(deepcopy(u))
            @test norm(uu2 .- u) < 1e-14

            # Test dss on Covariant12Vector
            uᵢ = Geometry.transform.(Ref(Geometry.Covariant12Axis()), u)
            uuᵢ2 = Spaces.weighted_dss!(deepcopy(uᵢ))
            @test norm(uuᵢ2 .- uᵢ) < 1e-14
        end
    end
end
