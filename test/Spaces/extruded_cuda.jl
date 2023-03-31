using Revise
using LinearAlgebra, IntervalSets, UnPack
using ClimaComms
import ClimaCore: Domains, Topologies, Meshes, Spaces, Geometry, column

using Test

FT = Float64
context = ClimaComms.SingletonCommsContext(ClimaComms.CUDA())
radius = FT(128)
zlim = (0, 1)
helem = 4
zelem = 10
Nq = 4

vertdomain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(zlim[1]),
    Geometry.ZPoint{FT}(zlim[2]);
    boundary_tags = (:bottom, :top),
)
vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
verttopology = Topologies.IntervalTopology(context, vertmesh)
vert_center_space = Spaces.CenterFiniteDifferenceSpace(verttopology)

horzdomain = Domains.SphereDomain(radius)
horzmesh = Meshes.EquiangularCubedSphere(horzdomain, helem)
horztopology = Topologies.Topology2D(context, horzmesh)
quad = Spaces.Quadratures.GLL{Nq}()
horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

hv_center_space =
    Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
