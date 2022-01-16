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
    Operators

include("ref_state.jl")

const n_vert = 10
const n_horz = 4
const p_horz = 4

const R = 6.4e6 # radius
const z_top = 3.0e4 # height position of the model top
const T_0 = 300 # isothermal atmospheric temperature
# set up function space
function sphere_3D(
    R = 6.4e6,
    zlim = (0, 30.0e3),
    helem = 4,
    zelem = 15,
    npoly = 5,
)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

    horzdomain = Domains.SphereDomain(R)
    horzmesh = Meshes.EquiangularCubedSphere(horzdomain, helem)
    horztopology = Topologies.Topology2D(horzmesh)
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space)
end

hv_center_space, hv_face_space =
    sphere_3D(R, (0, z_top), n_horz, n_vert, p_horz)
c_coords = Fields.coordinate_field(hv_center_space)
f_coords = Fields.coordinate_field(hv_face_space)

zc_vec = parent(c_coords.z) |> unique


ref_profile = calc_ref_state(c_coords, isothermal_profile)
ref_profile = calc_ref_state(c_coords, decaying_temperature_profile)

# Plots

ρ = unique(parent(ref_profile.ρ))
ρe = unique(parent(ref_profile.ρe))
p = unique(parent(ref_profile.p))

ana_profile = decaying_temperature_profile.(c_coords.z) 
ρ_ana = unique(parent(ana_profile.ρ))
T_ana = unique(parent(ana_profile.Tv))
p_ana = unique(parent(ana_profile.p))

# ρ_ana = []
# T_ana = []
# p_ana = []
# for i in 1:length(zc_vec)
#     var = isothermal_profile(zc_vec[i])
#     append!(ρ_ana, var.ρ)
#     append!(p_ana, var.p)
#     append!(T_ana, var.Tv)
# end

T = @. (ρe/ρ - Φ(zc_vec)) / cv_d + T_tri
# T = @. p/R_d/ρ
ρe_ana = @. ρ_ana * cv_d * (T_ana - T_tri) + ρ_ana * Φ(zc_vec) 

using Plots

Plots.png( plot( [ρ, ρ_ana], zc_vec, labels = ["ρ-disc", "ρ-ana"] ), "./rho.png")
Plots.png( plot( [ρe, ρe_ana], zc_vec, labels = ["ρe-disc", "ρe-ana"] ), "./rhoe.png")
Plots.png( plot( [T, T_ana], zc_vec, labels = ["T-disc", "T-ana"] ), "./T.png")
Plots.png( plot( [p, p_ana], zc_vec, labels = ["p-disc", "p-ana"] ), "./p.png")