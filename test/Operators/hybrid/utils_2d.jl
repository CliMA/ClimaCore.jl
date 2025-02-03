# V_face = Covariant3Vector
# V_center = Covariant12Vector
# V = V_center + V_face

# divergence(V) = horz_div(V) + vert_div(V)
# divergence(V) = horz_div(V_center) + horz_div(V_face) + vert_div(V_center) + vert_div(V_face)


# 1) horz_div(V_center): project to Contravariant1 + Contravariant2, take spectral derivative
# 2) horz_div(V_face): project to Contravariant1 + Contravariant2, interpolate to center, take spectral derivative
#   - will be zero if orthogional geom
# 3) vert_div(V_center): project to Contravariant3, interpolate to face, take FD deriv
#   - will be zero if orthogional geom
# 4) vert_div(V_face): project to Contravariant3, take FD deriv

using Test
using StaticArrays, IntervalSets, LinearAlgebra
import Adapt
import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore:
    ClimaCore,
    slab,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    Spaces,
    Quadratures,
    Fields,
    Operators,
    level
import ClimaCore.Geometry: WVector
import ClimaCore.Domains.Geometry: ⊗
import ClimaCore.Utilities: half

convergence_rate(err, Δh) =
    [log(err[i] / err[i - 1]) / log(Δh[i] / Δh[i - 1]) for i in 2:length(Δh)]

function hvspace_2D(;
    xlim = (-π, π),
    zlim = (0, 4π),
    helem = 10,
    velem = 64,
    npoly = 7,
    stretch = Meshes.Uniform(),
)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, stretch, nelems = velem)
    device = ClimaComms.device()
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(device, vertmesh)

    horzdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1]) .. Geometry.XPoint{FT}(xlim[2]),
        periodic = true,
    )
    horzmesh = Meshes.IntervalMesh(horzdomain, nelems = helem)
    horztopology = Topologies.IntervalTopology(device, horzmesh)

    quad = Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space)
end
