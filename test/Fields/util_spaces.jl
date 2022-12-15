import ClimaCore as CC
using ClimaComms

#=
Return a vector of toy spaces for testing
=#
function all_spaces(::Type{FT}; zelem = 10) where {FT}

    # 1d domain space
    domain = CC.Domains.IntervalDomain(
        CC.Geometry.XPoint{FT}(-3) .. CC.Geometry.XPoint{FT}(5),
        periodic = true,
    )
    mesh = CC.Meshes.IntervalMesh(domain; nelems = 1)
    topology = CC.Topologies.IntervalTopology(mesh)

    quad = CC.Spaces.Quadratures.GLL{4}()
    points, weights = CC.Spaces.Quadratures.quadrature_points(FT, quad)

    space1 = CC.Spaces.SpectralElementSpace1D(topology, quad)

    # finite difference spaces
    domain = CC.Domains.IntervalDomain(
        CC.Geometry.ZPoint{FT}(0) .. CC.Geometry.ZPoint{FT}(5),
        boundary_names = (:bottom, :top),
    )
    mesh = CC.Meshes.IntervalMesh(domain; nelems = 1)
    topology = CC.Topologies.IntervalTopology(mesh)

    space2 = CC.Spaces.CenterFiniteDifferenceSpace(topology)
    space3 = CC.Spaces.FaceFiniteDifferenceSpace(topology)

    # 1×1 domain space
    domain = CC.Domains.RectangleDomain(
        CC.Geometry.XPoint{FT}(-3) .. CC.Geometry.XPoint{FT}(5),
        CC.Geometry.YPoint{FT}(-2) .. CC.Geometry.YPoint{FT}(8),
        x1periodic = true,
        x2periodic = false,
        x2boundary = (:south, :north),
    )
    mesh = CC.Meshes.RectilinearMesh(domain, 1, 1)
    grid_topology =
        CC.Topologies.Topology2D(ClimaComms.SingletonCommsContext(), mesh)

    quad = CC.Spaces.Quadratures.GLL{4}()
    points, weights = CC.Spaces.Quadratures.quadrature_points(FT, quad)

    space4 = CC.Spaces.SpectralElementSpace2D(grid_topology, quad)

    # sphere space
    radius = FT(3)
    ne = 4
    Nq = 4
    domain = CC.Domains.SphereDomain(radius)
    mesh = CC.Meshes.EquiangularCubedSphere(domain, ne)
    topology =
        CC.Topologies.Topology2D(ClimaComms.SingletonCommsContext(), mesh)
    quad = CC.Spaces.Quadratures.GLL{Nq}()
    space5 = CC.Spaces.SpectralElementSpace2D(topology, quad)

    radius = FT(128)
    zlim = (0, 1)
    helem = 4
    Nq = 4

    vertdomain = CC.Domains.IntervalDomain(
        CC.Geometry.ZPoint{FT}(zlim[1]),
        CC.Geometry.ZPoint{FT}(zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = CC.Meshes.IntervalMesh(vertdomain, nelems = zelem)
    space6 = CC.Spaces.CenterFiniteDifferenceSpace(vertmesh)

    horzdomain = CC.Domains.SphereDomain(radius)
    horzmesh = CC.Meshes.EquiangularCubedSphere(horzdomain, helem)
    horztopology =
        CC.Topologies.Topology2D(ClimaComms.SingletonCommsContext(), horzmesh)
    quad = CC.Spaces.Quadratures.GLL{Nq}()
    space7 = CC.Spaces.SpectralElementSpace2D(horztopology, quad)

    space8 = CC.Spaces.ExtrudedFiniteDifferenceSpace(space7, space6)

    return [space1, space2, space3, space4, space5, space6, space7, space8]
end

bycolumnable(space) = (
    space isa Spaces.ExtrudedFiniteDifferenceSpace ||
    space isa Spaces.SpectralElementSpace1D ||
    space isa Spaces.SpectralElementSpace2D
)
