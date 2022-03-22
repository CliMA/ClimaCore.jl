using ClimaCore: Geometry, Domains, Meshes, Topologies, Spaces

abstract type Space{FT} end

Base.@kwdef struct PeriodicLine{FT} <: Space{FT}
    xmax::FT
    xelem::Int
    npoly::Int
end

Base.@kwdef struct PeriodicRectangle{FT} <: Space{FT}
    xmax::FT
    ymax::FT
    xelem::Int
    yelem::Int
    npoly::Int
end

Base.@kwdef struct CubedSphere{FT} <: Space{FT}
    radius::FT
    helem::Int # number of elements along side of each panel (6 panels in total)
    npoly::Int
end

Base.@kwdef struct ExtrudedSpace{FT, S <: Space{FT}} <: Space{FT}
    zmax::FT
    zelem::Int
    hspace::S
end

function make_space((; xmax, xelem, npoly)::PeriodicLine)
    domain = Domains.IntervalDomain(
        Geometry.XPoint(zero(xmax)),
        Geometry.XPoint(xmax);
        periodic = true,
    )
    mesh = Meshes.IntervalMesh(domain; nelems = xelem)
    topology = Topologies.IntervalTopology(mesh)
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    return Spaces.SpectralElementSpace1D(topology, quad)
end

function make_space((; xmax, ymax, xelem, yelem, npoly)::PeriodicRectangle)
    xdomain = Domains.IntervalDomain(
        Geometry.XPoint(zero(xmax)),
        Geometry.XPoint(xmax);
        periodic = true,
    )
    ydomain = Domains.IntervalDomain(
        Geometry.YPoint(zero(ymax)),
        Geometry.YPoint(ymax);
        periodic = true,
    )
    domain = Domains.RectangleDomain(xdomain, ydomain)
    mesh = Meshes.RectilinearMesh(domain, xelem, yelem)
    topology = Topologies.Topology2D(mesh)
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    return Spaces.SpectralElementSpace2D(topology, quad)
end

function make_space((; radius, helem, npoly)::CubedSphere)
    domain = Domains.SphereDomain(radius)
    mesh = Meshes.EquiangularCubedSphere(domain, helem)
    topology = Topologies.Topology2D(mesh)
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    return Spaces.SpectralElementSpace2D(topology, quad)
end

function make_space((; zmax, zelem, hspace)::ExtrudedSpace)
    vdomain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(zmax)),
        Geometry.ZPoint(zmax);
        boundary_tags = (:bottom, :top),
    )
    vmesh = Meshes.IntervalMesh(vdomain, nelems = zelem)
    vspace = Spaces.CenterFiniteDifferenceSpace(vmesh)
    return Spaces.ExtrudedFiniteDifferenceSpace(make_space(hspace), vspace)
end

function local_geometry_fields(space::ExtrudedSpace)
    center_space = make_space(space)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)
    return (
        Fields.local_geometry_field(center_space),
        Fields.local_geometry_field(face_space),
    )
end

local_geometry_fields(
    cspace::Spaces.CenterExtrudedFiniteDifferenceSpace,
    fspace::Spaces.FaceExtrudedFiniteDifferenceSpace,
) = (Fields.local_geometry_field(cspace), Fields.local_geometry_field(fspace))
