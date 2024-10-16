using Test
using StaticArrays
using ClimaComms
import ClimaCore.DataLayouts: IJHF, VF
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

FT = Float64
hdomain = Domains.IntervalDomain(
    Geometry.XPoint{FT}(-pi) .. Geometry.XPoint{FT}(pi),
    periodic = true,
)

Nq = 5
quad = Quadratures.GLL{Nq}()
device = ClimaComms.CPUSingleThreaded()
hmesh = Meshes.IntervalMesh(hdomain, nelems = 16)
htopology =
    Topologies.IntervalTopology(ClimaComms.SingletonCommsContext(device), hmesh)
hspace = Spaces.SpectralElementSpace1D(htopology, quad)

vdomain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(-pi),
    Geometry.ZPoint{FT}(pi);
    boundary_names = (:bottom, :top),
)
vmesh = Meshes.IntervalMesh(vdomain, nelems = 16)
vtopology =
    Topologies.IntervalTopology(ClimaComms.SingletonCommsContext(device), vmesh)
vspace = Spaces.CenterFiniteDifferenceSpace(vtopology)

cspace = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)

curl = Operators.Curl()

w = map(
    coord -> Geometry.WVector(coord.x + coord.z),
    Fields.coordinate_field(cspace),
)
@test curl.(Geometry.Covariant3Vector.(w)) ≈ map(
    coord -> Geometry.Contravariant2Vector(-1.0),
    Fields.coordinate_field(cspace),
)

v = map(
    coord -> Geometry.Covariant2Vector(coord.x + coord.z),
    Fields.coordinate_field(cspace),
)
@test Geometry.WVector.(curl.(Geometry.Covariant2Vector.(v))) ≈
      map(coord -> Geometry.WVector(1.0), Fields.coordinate_field(cspace))
