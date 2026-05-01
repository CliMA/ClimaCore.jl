using ClimaCore:
    DataLayouts,
    Fields,
    Domains,
    Geometry,
    Topologies,
    Meshes,
    Spaces,
    Limiters,
    Quadratures
using ClimaCore: slab
using Test

using ClimaComms
ClimaComms.@import_required_backends
const context = ClimaComms.MPICommsContext()
ClimaComms.init(context)

FT = Float64
xlim = (0, 2π)
ylim = (0, 1)
zlim = (0, 2)
xelems = 3
yelems = 3
zelems = 3
Nij = 5

xdomain = Domains.IntervalDomain(
    Geometry.XPoint{FT}(xlim[1]),
    Geometry.XPoint{FT}(xlim[2]),
    periodic = true,
)
ydomain = Domains.IntervalDomain(
    Geometry.YPoint{FT}(ylim[1]),
    Geometry.YPoint{FT}(ylim[2]),
    periodic = true,
)

horzdomain = Domains.RectangleDomain(xdomain, ydomain)
horzmesh = Meshes.RectilinearMesh(horzdomain, xelems, yelems)
horztopology = Topologies.Topology2D(context, horzmesh)

zdomain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(zlim[1]),
    Geometry.ZPoint{FT}(zlim[2]);
    boundary_names = (:bottom, :top),
)
vertmesh = Meshes.IntervalMesh(zdomain, nelems = zelems)
device = ClimaComms.device(context)
vert_center_space = Spaces.CenterFiniteDifferenceSpace(device, vertmesh)

quad = Quadratures.GLL{Nij}()
horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

hv_center_space =
    Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)


# Initialize fields
ρ = map(
    coord -> exp(-coord.x - coord.z),
    Fields.coordinate_field(hv_center_space),
)
q = map(
    coord -> (x = 1.2 * coord.x, y = 1.5 * coord.y),
    Fields.coordinate_field(hv_center_space),
)
ρq = ρ .* q
q_ref = map(
    coord -> (x = coord.x, y = coord.y),
    Fields.coordinate_field(hv_center_space),
)
ρq_ref = ρ .* q_ref

total_ρq = sum(ρq)

limiter = Limiters.QuasiMonotoneLimiter(ρq)

Limiters.compute_bounds!(limiter, ρq_ref, ρ)
Limiters.apply_limiter!(ρq, ρ, limiter)
q = ρq ./ ρ

@test sum(ρq.x) ≈ total_ρq.x
@test sum(ρq.y) ≈ total_ρq.y
@test all(0 .<= parent(ρq) .<= 1)
