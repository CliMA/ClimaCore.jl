using Test
using StaticArrays
using ClimaComms, ClimaCore
ClimaComms.@import_required_backends
import ClimaCore:
    Geometry,
    Fields,
    Domains,
    Topologies,
    Meshes,
    Spaces,
    Operators,
    Quadratures
using LinearAlgebra, IntervalSets

function hvspace_3D_box(
    context,
    xlim = (-π, π),
    ylim = (-π, π),
    zlim = (0, 4π),
    xelem = 4,
    yelem = 4,
    zelem = 16,
    npoly = 7,
)
    FT = Float64

    # Define vert domain and mesh
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)

    # Define vert topology and space
    verttopology = Topologies.IntervalTopology(context, vertmesh)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(verttopology)

    horzdomain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(xlim[1]) .. Geometry.XPoint{FT}(xlim[2]),
        Geometry.YPoint{FT}(ylim[1]) .. Geometry.YPoint{FT}(ylim[2]),
        x1periodic = true,
        x2periodic = true,
    )
    horzmesh = Meshes.RectilinearMesh(horzdomain, xelem, yelem)

    quad = Quadratures.GLL{npoly + 1}()

    # Define horz topology and space
    horztopology = Topologies.Topology2D(context, horzmesh)
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space)
end

function hvspace_3D_sphere(context)
    FT = Float64

    # Define vert domain and mesh
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0.0),
        Geometry.ZPoint{FT}(1.0);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = 10)

    # Define vert topology and space
    verttopology = Topologies.IntervalTopology(context, vertmesh)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(verttopology)

    # Define horz domain and mesh
    horzdomain = Domains.SphereDomain(FT(30.0))
    horzmesh = Meshes.EquiangularCubedSphere(horzdomain, 4)

    quad = Quadratures.GLL{3 + 1}()

    # Define horz topology and space
    horztopology = Topologies.Topology2D(context, horzmesh)
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    # Define hv spaces
    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return hv_center_space, hv_face_space
end
