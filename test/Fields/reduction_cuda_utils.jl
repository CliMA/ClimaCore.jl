
# set initial condition for steady-state test
function set_initial_condition(space)
    Y = map(Fields.local_geometry_field(space)) do local_geometry
        h = 1.0
        return h
    end
    return Y
end

# set simple field
function set_simple_field(space)
    α0 = 45.0
    h0 = 1.0
    Y = map(Fields.local_geometry_field(space)) do local_geometry
        coord = local_geometry.coordinates
        ϕ = coord.lat
        λ = coord.long
        z = coord.z
        h = h0 * z * (cosd(α0) * cosd(ϕ) + sind(α0) * cosd(λ) * sind(ϕ))
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

function extruded_sphere_spaces(::Type{FT}, device) where {FT}
    context = ClimaComms.SingletonCommsContext(device)
    R = FT(6.371229e6)
    npoly = 4
    z_max = FT(30e3)
    z_elem = 10
    h_elem = 4
    @info(
        "running reduction test on",
        context.device,
        h_elem,
        npoly,
        R,
        z_max,
        FT,
    )
    # horizontal space
    domain = Domains.SphereDomain(R)
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
        Geometry.ZPoint(zero(z_max)),
        Geometry.ZPoint(z_max);
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain, nelems = z_elem)
    z_topology = Topologies.IntervalTopology(context, z_mesh)
    z_center_space = Spaces.CenterFiniteDifferenceSpace(z_topology)
    z_face_space = Spaces.FaceFiniteDifferenceSpace(z_topology)
    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(h_space, z_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (;cspace = hv_center_space, fspace = hv_face_space)
end

function sem_2d_sphere_spaces(::Type{FT}, device) where {FT}
    context = ClimaComms.SingletonCommsContext(device)
    # Set up discretization
    ne = 72
    Nq = 4
    ndof = ne * ne * 6 * Nq * Nq
    @info(
        "Running reduction test on",
        context.device,
        ne,
        Nq,
        ndof,
        FT,
    )
    R = FT(6.37122e6) # radius of earth
    domain = Domains.SphereDomain(R)
    mesh = Meshes.EquiangularCubedSphere(domain, ne)
    quad = Quadratures.GLL{Nq}()
    grid_topology = Topologies.Topology2D(context, mesh)
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    return (;space)
end

