using Test
using CUDA
using ClimaComms
using Statistics
using LinearAlgebra

import ClimaCore:
    Device,
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Spaces,
    Topologies,
    DataLayouts

# set initial condition for steady-state test
function set_initial_condition(space)
    Y = map(Fields.local_geometry_field(space)) do local_geometry
        h = 1.0
        return h
    end
    return Y
end

function set_elevation(space, h₀)
    Y = map(Fields.local_geometry_field(space)) do local_geometry
        coord = local_geometry.coordinates

        ϕ = coord.lat
        λ = coord.long
        FT = eltype(λ)
        ϕₘ = FT(0) # degrees (equator)
        λₘ = FT(3 / 2 * 180)  # degrees
        rₘ =
            FT(acos(sind(ϕₘ) * sind(ϕ) + cosd(ϕₘ) * cosd(ϕ) * cosd(λ - λₘ))) # Great circle distance (rads)
        Rₘ = FT(3π / 4) # Moutain radius
        ζₘ = FT(π / 16) # Mountain oscillation half-width
        zₛ = ifelse(
            rₘ < Rₘ,
            FT(h₀ / 2) * (1 + cospi(rₘ / Rₘ)) * (cospi(rₘ / ζₘ))^2,
            FT(0),
        )
        return zₛ
    end
    return Y
end

@testset "test cuda reduction op on surface of sphere" begin
    FT = Float64

    device = Device.device()
    context = ClimaComms.SingletonCommsContext(device)
    context_cpu = ClimaComms.SingletonCommsContext() # CPU context for comparison

    # Set up discretization
    ne = 72
    Nq = 4
    ndof = ne * ne * 6 * Nq * Nq
    println(
        "running reduction test on $device device; problem size Ne = $ne; Nq = $Nq; ndof = $ndof; FT = $FT",
    )
    R = FT(6.37122e6) # radius of earth
    domain = Domains.SphereDomain(R)
    mesh = Meshes.EquiangularCubedSphere(domain, ne)
    quad = Spaces.Quadratures.GLL{Nq}()
    grid_topology = Topologies.Topology2D(context, mesh)
    grid_topology_cpu = Topologies.Topology2D(context_cpu, mesh)
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    space_cpu = Spaces.SpectralElementSpace2D(grid_topology_cpu, quad)

    coords = Fields.coordinate_field(space)
    Y = set_initial_condition(space)
    Y_cpu = set_initial_condition(space_cpu)

    h₀ = FT(200)
    Z = set_elevation(space, h₀)
    Z_cpu = set_elevation(space_cpu, h₀)

    result = Base.sum(Y)
    result_cpu = Base.sum(Y_cpu)

    local_max = Base.maximum(identity, Z)
    local_max_cpu = Base.maximum(identity, Z_cpu)

    local_min = Base.minimum(identity, Z)
    local_min_cpu = Base.minimum(identity, Z_cpu)
    # test weighted sum
    @test result[1] ≈ 4 * pi * R^2 rtol = 1e-5
    @test result[1] ≈ result_cpu[1]
    # test maximum
    @test local_max[1] ≈ h₀
    @test local_max[1] ≈ local_max_cpu
    # test minimum
    @test local_min[1] ≈ FT(0)
    @test local_min[1] ≈ local_min_cpu
    # testing mean
    meanz = Statistics.mean(Z)
    meanz_cpu = Statistics.mean(Z_cpu)
    @test meanz[] ≈ meanz_cpu[]
    # testing norm
    norm1z = LinearAlgebra.norm(Z, 1)
    norm1z_cpu = LinearAlgebra.norm(Z_cpu, 1)
    @test norm1z[] ≈ norm1z_cpu[]

    norm2z = LinearAlgebra.norm(Z, 2)
    norm2z_cpu = LinearAlgebra.norm(Z_cpu, 2)
    @test norm2z[] ≈ norm2z_cpu[]

    norm3z = LinearAlgebra.norm(Z, 3)
    norm3z_cpu = LinearAlgebra.norm(Z_cpu, 3)
    @test norm3z[] ≈ norm3z_cpu[]

    norminfz = LinearAlgebra.norm(Z, Inf)
    norminfz_cpu = LinearAlgebra.norm(Z_cpu, Inf)
    @test norminfz[] ≈ norminfz_cpu[]
end
