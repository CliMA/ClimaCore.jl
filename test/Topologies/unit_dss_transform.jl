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
    # Horizontal space
    domain = Domains.SphereDomain{FT}(R)
    horizontal_mesh = Meshes.EquiangularCubedSphere(domain, h_elem)
    horizontal_topology = Topologies.Topology2D(
        context,
        horizontal_mesh,
        Topologies.spacefillingcurve(horizontal_mesh),
    )
    quad = Quadratures.GLL{npoly + 1}()
    h_space = Spaces.SpectralElementSpace2D(horizontal_topology, quad)
    # Vertical space
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
    result = map(local_geometry) do lg
        FT = Geometry.undertype(typeof(lg))
        (; lat, long, z) = lg.coordinates
        # Test that vertical component is treated as a scalar:

        arg1 = Geometry.Covariant123Vector(FT(lat), FT(long), FT(z))
        weight1 = 2
        dss_t1 = dss_transform(arg1, lg, weight1)
        dss_ut1 = dss_untransform(Geometry.Covariant123Vector{FT}, dss_t1, lg)
        pass1 = typeof(arg1) == typeof(dss_ut1) && arg1 ≈ dss_ut1 / weight1

        arg2 = Geometry.Covariant12Vector(FT(lat), FT(long))
        weight2 = 2
        dss_t2 = dss_transform(arg2, lg, weight2)
        dss_ut2 = dss_untransform(Geometry.Covariant12Vector{FT}, dss_t2, lg)
        pass2 = typeof(arg2) == typeof(dss_ut2) && arg2 ≈ dss_ut2 / weight2

        arg3 = Geometry.Covariant3Vector(FT(z))
        weight3 = 2
        dss_t3 = dss_transform(arg3, lg, weight3)
        dss_ut3 = dss_untransform(Geometry.Covariant3Vector{FT}, dss_t3, lg)
        pass3 =
            typeof(arg3) == typeof(dss_ut3) && dss_t3 === arg3 * weight3 &&
            arg3 == dss_ut3 / weight3

        pass1 && pass2 && pass3
    end
    @test all(Array(parent(result)))
end
