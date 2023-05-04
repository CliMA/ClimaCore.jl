"""
    TestUtilities

TestUtilities is designed to:
 - Reduce the test boilerplate
 - Provide testers with iterators over
   instances of types to ease testing
   a diverse set of inputs to functions
"""
module TestUtilities

using IntervalSets
import ClimaComms
import ClimaCore.Fields as Fields
import ClimaCore.Utilities as Utilities
import ClimaCore.Geometry as Geometry
import ClimaCore.Meshes as Meshes
import ClimaCore.Spaces as Spaces
import ClimaCore.Topologies as Topologies
import ClimaCore.Domains as Domains

function PointSpace(
    ::Type{FT};
    context = ClimaComms.SingletonCommsContext(),
) where {FT}
    coord = Geometry.XPoint(FT(π))
    space = Spaces.PointSpace(context, coord)
    return space
end

function SpectralElementSpace1D(
    ::Type{FT};
    context = ClimaComms.SingletonCommsContext(),
) where {FT}
    # 1d domain space
    domain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(-3) .. Geometry.XPoint{FT}(5),
        periodic = true,
    )
    mesh = Meshes.IntervalMesh(domain; nelems = 1)
    topology = Topologies.IntervalTopology(context, mesh)
    quad = Spaces.Quadratures.GLL{4}()
    return Spaces.SpectralElementSpace1D(topology, quad)
end

function SpectralElementSpace2D(
    ::Type{FT};
    context = ClimaComms.SingletonCommsContext(),
) where {FT}
    # 1×1 domain space
    domain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(-3) .. Geometry.XPoint{FT}(5),
        Geometry.YPoint{FT}(-2) .. Geometry.YPoint{FT}(8),
        x1periodic = true,
        x2periodic = false,
        x2boundary = (:south, :north),
    )
    mesh = Meshes.RectilinearMesh(domain, 1, 1)
    topology = Topologies.Topology2D(context, mesh)
    quad = Spaces.Quadratures.GLL{4}()
    return Spaces.SpectralElementSpace2D(topology, quad)
end

#= (single column) =#
function ColumnCenterFiniteDifferenceSpace(
    ::Type{FT};
    zelem = 10,
    context = ClimaComms.SingletonCommsContext(),
) where {FT}
    zlim = (0, 1)
    domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    mesh = Meshes.IntervalMesh(domain, nelems = zelem)
    topology = Topologies.IntervalTopology(context, mesh)
    return Spaces.CenterFiniteDifferenceSpace(topology)
end

#= (single column) =#
function ColumnFaceFiniteDifferenceSpace(
    ::Type{FT};
    zelem = 10,
    context = ClimaComms.SingletonCommsContext(),
) where {FT}
    cspace = ColumnCenterFiniteDifferenceSpace(FT; zelem, context)
    return Spaces.FaceFiniteDifferenceSpace(cspace)
end

function SphereSpectralElementSpace(
    ::Type{FT};
    context = ClimaComms.SingletonCommsContext(),
) where {FT}
    radius = FT(3)
    ne = 4
    Nq = 4
    domain = Domains.SphereDomain(radius)
    mesh = Meshes.EquiangularCubedSphere(domain, ne)
    topology = Topologies.Topology2D(context, mesh)
    quad = Spaces.Quadratures.GLL{Nq}()
    return Spaces.SpectralElementSpace2D(topology, quad)
end

function ExtrudedCenterFiniteDifferenceSpace(
    ::Type{FT};
    zelem = 10,
    context = ClimaComms.SingletonCommsContext(),
) where {FT}
    radius = FT(128)
    zlim = (0, 1)
    helem = 4
    Nq = 4
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
    vtopology = Topologies.IntervalTopology(context, vertmesh)
    vspace = Spaces.CenterFiniteDifferenceSpace(vtopology)

    hdomain = Domains.SphereDomain(radius)
    hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
    htopology = Topologies.Topology2D(context, hmesh)
    quad = Spaces.Quadratures.GLL{Nq}()
    hspace = Spaces.SpectralElementSpace2D(htopology, quad)
    return Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
end

function ExtrudedFaceFiniteDifferenceSpace(
    ::Type{FT};
    zelem = 10,
    context = ClimaComms.SingletonCommsContext(),
) where {FT}
    cspace = ExtrudedCenterFiniteDifferenceSpace(FT; zelem, context)
    return Spaces.ExtrudedFaceFiniteDifferenceSpace(cspace)
end

function all_spaces(
    ::Type{FT};
    zelem = 10,
    context = ClimaComms.SingletonCommsContext(),
) where {FT}
    return [
        PointSpace(FT; context),
        SpectralElementSpace1D(FT; context),
        SpectralElementSpace2D(FT; context),
        # TODO: add these
        # SpectralElementRectilinearSpace2D(FT; context),
        # SpectralElementFiniteDifferenceRectilinearSpace2D(FT; context),
        ColumnCenterFiniteDifferenceSpace(FT; zelem, context),
        ColumnFaceFiniteDifferenceSpace(FT; zelem, context),
        SphereSpectralElementSpace(FT; context),
        ExtrudedCenterFiniteDifferenceSpace(FT; zelem, context),
        # ExtrudedFaceFiniteDifferenceSpace(FT; zelem, context), # errors on sum
        # TODO: incorporate this list of spaces somehow:
        #     space_vf = Spaces.CenterFiniteDifferenceSpace(topology_z)
        #     space_ifh = Spaces.SpectralElementSpace1D(topology_x, quad)
        #     space_ijfh = Spaces.SpectralElementSpace2D(topology_xy, quad)
        #     space_vifh = Spaces.ExtrudedFiniteDifferenceSpace(space_ifh, space_vf)
        #     space_vijfh = Spaces.ExtrudedFiniteDifferenceSpace(space_ijfh, space_vf)
    ]
end

bycolumnable(space) = (
    space isa Spaces.ExtrudedFiniteDifferenceSpace ||
    space isa Spaces.SpectralElementSpace1D ||
    space isa Spaces.SpectralElementSpace2D
)

levelable(space) = (
    space isa Spaces.ExtrudedFiniteDifferenceSpace ||
    space isa Spaces.FiniteDifferenceSpace
)

fc_index(
    i,
    ::Union{
        Spaces.FaceExtrudedFiniteDifferenceSpace,
        Spaces.FaceFiniteDifferenceSpace,
    },
) = Utilities.PlusHalf(i)

fc_index(
    i,
    ::Union{
        Spaces.CenterExtrudedFiniteDifferenceSpace,
        Spaces.CenterFiniteDifferenceSpace,
    },
) = i

has_z_coordinates(space) = :z in propertynames(Spaces.coordinates_data(space))

function FieldFromNamedTuple(space, nt::NamedTuple)
    cmv(z) = nt
    return cmv.(Fields.coordinate_field(space))
end

end
