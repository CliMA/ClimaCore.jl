#=
julia --project
using Revise; include(joinpath("test", "Topologies", "unit_dss_transform.jl"))
=#
using Test
using ClimaComms
using ClimaCore.Topologies: dss_transform, dss_untransform
using Random
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

function get_space(::Type{FT}; context) where {FT}
    R = FT(6.371229e6)
    npoly = 2
    z_max = FT(30e3)
    z_elem = 3
    h_elem = 2
    device = ClimaComms.device(context)
    @info "running dss-Covariant123Vector test on $(device)" h_elem z_elem npoly R z_max FT
    # horizontal space
    domain = Domains.SphereDomain{FT}(R)
    horizontal_mesh = Meshes.EquiangularCubedSphere(domain, h_elem)
    horizontal_topology = Topologies.Topology2D(
        context,
        horizontal_mesh,
        Topologies.spacefillingcurve(horizontal_mesh),
    )
    quad = Quadratures.GLL{npoly + 1}()
    h_space = Spaces.SpectralElementSpace2D(horizontal_topology, quad)
    # vertical space
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zero(z_max)),
        Geometry.ZPoint{FT}(z_max);
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain, nelems = z_elem)
    z_topology = Topologies.IntervalTopology(context, z_mesh)
    z_center_space = Spaces.CenterFiniteDifferenceSpace(z_topology)
    space = Spaces.ExtrudedFiniteDifferenceSpace(h_space, z_center_space)
    return space
end

@testset "dss_transform" begin
    device = ClimaComms.device()
    space = get_space(Float64; context = ClimaComms.context(device))

    local_geometry = Fields.local_geometry_field(space)
    map(local_geometry) do lg
        FT = Geometry.undertype(typeof(lg))
        (; lat, long, z) = lg.coordinates
        # Test that vertical component is treated as a scalar:

        arg = Geometry.Covariant123Vector(FT(lat), FT(long), FT(z))
        weight = 2
        dss_t = dss_transform(arg, lg, weight)
        dss_ut = dss_untransform(Geometry.Covariant123Vector{FT}, dss_t, lg)
        @test dss_t isa Geometry.UVWVector
        @test typeof(arg) == typeof(dss_ut)
        @test arg ≈ dss_ut / weight

        arg = Geometry.Covariant12Vector(FT(lat), FT(long))
        weight = 2
        dss_t = dss_transform(arg, lg, weight)
        dss_ut = dss_untransform(Geometry.Covariant12Vector{FT}, dss_t, lg)
        @test dss_t isa Geometry.UVWVector
        @test typeof(arg) == typeof(dss_ut)
        @test arg ≈ dss_ut / weight

        arg = Geometry.Covariant3Vector(FT(z))
        weight = 2
        dss_t = dss_transform(arg, lg, weight)
        dss_ut = dss_untransform(Geometry.Covariant3Vector{FT}, dss_t, lg)
        @test typeof(arg) == typeof(dss_ut)
        @test dss_t isa Geometry.Covariant3Vector
        @test dss_t === arg * weight
        @test arg == dss_ut / weight
        FT(1)
    end
end
