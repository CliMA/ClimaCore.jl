using Test
using StaticArrays
using ClimaComms, ClimaCore
import ClimaCore:
    Geometry, Fields, Domains, Topologies, Meshes, Spaces, Operators
using LinearAlgebra, IntervalSets
using CUDA

FT = Float64
vertdomain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(0.0),
    Geometry.ZPoint{FT}(1.0);
    boundary_tags = (:bottom, :top),
)
vertmesh = Meshes.IntervalMesh(vertdomain, nelems = 10)
verttopology = Topologies.IntervalTopology(
    ClimaComms.SingletonCommsContext(ClimaComms.CUDA()),
    vertmesh,
)
vert_center_space = Spaces.CenterFiniteDifferenceSpace(verttopology)

horzdomain = Domains.SphereDomain(30.0)
horzmesh = Meshes.EquiangularCubedSphere(horzdomain, 4)
horztopology = Topologies.Topology2D(
    ClimaComms.SingletonCommsContext(ClimaComms.CUDA()),
    horzmesh,
)
quad = Spaces.Quadratures.GLL{3 + 1}()
horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

hv_center_space =
    Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
z = Fields.coordinate_field(hv_face_space).z

gradc = Operators.GradientF2C()

@test parent(Geometry.WVector.(gradc.(z))) ≈
      parent(Geometry.WVector.(ones(hv_center_space)))
