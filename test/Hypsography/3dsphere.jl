using Test
using ClimaComms
import ClimaCore
import ClimaCore:
    ClimaCore,
    slab,
    Spaces,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    Spaces,
    Fields,
    Operators,
    InputOutput,
    Hypsography
using ClimaCore.Geometry

FT = Float64
vertdomain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(0),
    Geometry.ZPoint{FT}(4);
    boundary_names = (:bottom, :top),
)
vertmesh = Meshes.IntervalMesh(vertdomain, nelems = 40)
vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)



horzdomain = Domains.SphereDomain(6e6)
horzmesh = Meshes.EquiangularCubedSphere(horzdomain, 12)
horztopology =
    Topologies.Topology2D(ClimaComms.SingletonCommsContext(), horzmesh)
quad = Spaces.Quadratures.GLL{4 + 1}()
horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

z_surface =
    cosd.(Fields.coordinate_field(horzspace).lat) .+
    cosd.(Fields.coordinate_field(horzspace).long) .+ 1

hv_center_space = Spaces.ExtrudedFiniteDifferenceSpace(
    horzspace,
    vert_center_space,
    Hypsography.LinearAdaption(z_surface),
)
hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)

grad_h = Operators.Gradient()
grad_v = Operators.GradientF2C()

center_z = Fields.coordinate_field(hv_center_space).z
face_z = Fields.coordinate_field(hv_face_space).z
∇z_h = Geometry.UVWVector.(grad_h.(center_z))
∇z_v = Geometry.UVWVector.(grad_v.(face_z))
∇z = ∇z_h .+ ∇z_v

@test maximum(u -> abs(u.u), ∇z) < 1e-4
@test maximum(u -> abs(u.v), ∇z) < 1e-4
@test map(u -> u.w, ∇z) ≈ ones(hv_center_space)

center_x = cosd.(Fields.coordinate_field(hv_center_space).long)
face_x = cosd.(Fields.coordinate_field(hv_face_space).long)
∇x_h = Geometry.UVWVector.(grad_h.(center_x))
∇x_v = Geometry.UVWVector.(grad_v.(face_x))
∇x = ∇x_h .+ ∇x_v

@test maximum(u -> abs(u.w), ∇x) < 1e-10
