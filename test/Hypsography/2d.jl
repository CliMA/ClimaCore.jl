using Test
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



horzdomain = Domains.IntervalDomain(
    Geometry.XPoint{FT}(0),
    Geometry.XPoint{FT}(4pi);
    periodic = true,
)
horzmesh = Meshes.IntervalMesh(horzdomain, nelems = 20)
horztopology = Topologies.IntervalTopology(horzmesh)
quad = Spaces.Quadratures.GLL{4 + 1}()
horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)

z_surface = sin.(Fields.coordinate_field(horzspace).x) .+ 1

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
∇z_h = Geometry.UWVector.(grad_h.(center_z))
∇z_v = Geometry.UWVector.(grad_v.(face_z))
∇z = ∇z_h .+ ∇z_v

@test maximum(u -> abs(u.u), ∇z) < 1e-4
@test map(u -> u.w, ∇z) ≈ ones(hv_center_space)

center_x = Fields.coordinate_field(hv_center_space).x
face_x = Fields.coordinate_field(hv_face_space).x
∇x_h = Geometry.UWVector.(grad_h.(center_x))
∇x_v = Geometry.UWVector.(grad_v.(face_x))
∇x = ∇x_h .+ ∇x_v

@test map(u -> u.u, ∇x) ≈ ones(hv_center_space)
@test maximum(u -> abs(u.w), ∇x) < 1e-10
