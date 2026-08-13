using Test
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore
import ClimaCore:
    Domains, Meshes, Geometry, Topologies, Spaces, Quadratures, Fields, Operators

# Smoke test for the 3D hydrostatic dynamical core: a hydrostatically-balanced,
# motionless atmosphere on the cubed sphere must remain at rest and conserve
# mass/energy, integrated with a hand-rolled SSP RK33 loop.
const FT = Float64

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU
import .TestUtilities: ssp33!  # shared hand-rolled SSP RK33 time integrator

const context = ClimaComms.SingletonCommsContext()
const R = FT(6.4e6)
const Ω = FT(7.2921e-5)
const z_top = FT(3.0e4)
const grav = FT(9.8)
const p_0 = FT(1e5)
const R_d = FT(287.058)
const T_tri = FT(273.16)
const γ = FT(1.4)
const cv_d = R_d / (γ - 1)
const cp_d = R_d + cv_d
const T_0 = FT(300)
const H = R_d * T_0 / grav

# Trimmed resolution/duration (source: helem=4, zelem=10, npoly=4, T=3600,
# dt=5). The at-rest and conservation properties hold from the first steps; a
# short run keeps this in the smoke budget.
const n_vert = 10
const n_horz = 4
const p_horz = 4
const dt = FT(5)
const nsteps = 10

function sphere_3D(helem, zelem, npoly)
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(FT(0)),
        Geometry.ZPoint{FT}(z_top);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
    device = ClimaComms.device(context)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(device, vertmesh)
    horzdomain = Domains.SphereDomain(R)
    horzmesh = Meshes.EquiangularCubedSphere(horzdomain, helem)
    horztopology = Topologies.Topology2D(context, horzmesh)
    quad = Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)
    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space)
end

Φ(z) = grav * z
function pressure(ρ, e, normuvw, z)
    I = e - Φ(z) - normuvw^2 / 2
    T = I / cv_d + T_tri
    return ρ * R_d * T
end
function init_sbr_thermo(z)
    p = p_0 * exp(-z / H)
    ρ = 1 / R_d / T_0 * p
    e = cv_d * (T_0 - T_tri) + Φ(z)
    return (ρ = ρ, ρe = ρ * e)
end

function rhs!(dY, Y, parameters, t)
    (; f, c_coords, cuvw, cw, cω³, fω¹², fu¹², fu³, cp, cE) = parameters
    cρ = Y.Yc.ρ
    fw = Y.w
    cuₕ = Y.uₕ
    cρe = Y.Yc.ρe
    dρ = dY.Yc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρe = dY.Yc.ρe
    z = c_coords.z

    hdiv = Operators.Divergence()
    hgrad = Operators.Gradient()
    hcurl = Operators.Curl()

    @. dρ = 0 * cρ
    @. dw = 0 * fw
    @. duₕ = 0 * cuₕ
    @. dρe = 0 * cρe

    If2c = Operators.InterpolateF2C()
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    @. cw = If2c(fw)
    @. cuvw =
        Geometry.Covariant123Vector.(cuₕ) .+ Geometry.Covariant123Vector.(cw)
    @. dρ -= hdiv(cρ * (cuvw))
    vdivf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
    )
    @. dρ -= vdivf2c(Ic2f(cρ * cuₕ))
    @. dρ -= vdivf2c(Ic2f(cρ) * fw)

    vcurlc2f = Operators.CurlC2F(
        bottom = Operators.SetCurl(Geometry.Contravariant12Vector(FT(0), FT(0))),
        top = Operators.SetCurl(Geometry.Contravariant12Vector(FT(0), FT(0))),
    )
    @. cω³ = hcurl(cuₕ)
    @. fω¹² = hcurl(fw)
    @. fω¹² += vcurlc2f(cuₕ)

    @. fu¹² =
        Geometry.Contravariant12Vector(Geometry.Covariant123Vector(Ic2f(cuₕ)))
    @. fu³ = Geometry.Contravariant3Vector(Geometry.Covariant123Vector(fw))
    @. dw -= fω¹² × fu¹²
    @. duₕ -= If2c(fω¹² × fu³)
    @. duₕ -=
        (f + cω³) ×
        Geometry.Contravariant12Vector(Geometry.Covariant123Vector(cuₕ))

    @. cp = pressure(cρ, cρe / cρ, norm(cuvw), z)
    @. duₕ -= hgrad(cp) / cρ
    vgradc2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
        top = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
    )
    @. dw -= vgradc2f(cp) / Ic2f(cρ)

    @. cE = (norm(cuvw)^2) / 2 + Φ(z)
    @. duₕ -= hgrad(cE)
    @. dw -= vgradc2f(cE)

    @. dρe -= hdiv(cuvw * (cρe + cp))
    @. dρe -= vdivf2c(fw * Ic2f(cρe + cp))
    @. dρe -= vdivf2c(Ic2f(cuₕ * (cρe + cp)))

    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.uₕ)
    Spaces.weighted_dss!(dY.w)
    return dY
end

function discrete_hydrostatic_balance!(ρ, p, dz, grav)
    for i in 1:(length(ρ) - 1)
        ρ[i + 1] = -ρ[i] - 2 * (p[i + 1] - p[i]) / dz / grav
    end
end

function build_solid_body_rotation()
    hv_center_space, hv_face_space = sphere_3D(n_horz, n_vert, p_horz)
    c_coords = Fields.coordinate_field(hv_center_space)
    f_coords = Fields.coordinate_field(hv_face_space)
    f = @. Geometry.Contravariant3Vector(
        Geometry.WVector(2 * Ω * sind(c_coords.lat)),
    )

    zc_vec = parent(c_coords.z) |> unique
    N = length(zc_vec)
    ρ = zeros(FT, N)
    p = zeros(FT, N)
    ρe = zeros(FT, N)
    for i in 1:N
        var = init_sbr_thermo(zc_vec[i])
        ρ[i], ρe[i] = var.ρ, var.ρe
        p[i] = pressure(ρ[i], ρe[i] / ρ[i], FT(0), zc_vec[i])
    end
    ρ_ana = copy(ρ)
    discrete_hydrostatic_balance!(ρ, p, z_top / n_vert, grav)
    ρe = @. ρe + (ρ - ρ_ana) * Φ(zc_vec) - (ρ - ρ_ana) * cv_d * T_tri

    Yc = map(coord -> init_sbr_thermo(coord.z), c_coords)
    parent(Yc.ρ) .= ρ
    parent(Yc.ρe) .= ρe
    uₕ = map(_ -> Geometry.Covariant12Vector(FT(0), FT(0)), c_coords)
    w = map(_ -> Geometry.Covariant3Vector(FT(0)), f_coords)
    Y = Fields.FieldVector(Yc = Yc, uₕ = uₕ, w = w)

    cuvw = Geometry.Covariant123Vector.(Y.uₕ)
    If2c = Operators.InterpolateF2C()
    hcurl = Operators.Curl()
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    cw = If2c.(Y.w)
    cω³ = hcurl.(Y.uₕ)
    fω¹² = hcurl.(Y.w)
    fu¹² = @. Geometry.Contravariant12Vector(
        Geometry.Covariant123Vector(Ic2f(Y.uₕ)),
    )
    fu³ = @. Geometry.Contravariant3Vector(Geometry.Covariant123Vector(Y.w))
    cp = @. pressure(Y.Yc.ρ, Y.Yc.ρe / Y.Yc.ρ, norm(cuvw), c_coords.z)
    cE = @. (norm(cuvw)^2) / 2 + Φ(c_coords.z)
    params = (; f, c_coords, cuvw, cw, cω³, fω¹², fu¹², fu³, cp, cE)
    return Y, params
end

@testset "Solid-body rotation (3D sphere): well-balanced at rest" begin
    Y, params = build_solid_body_rotation()
    Y0 = copy(Y)
    dY, Y1, Y2 = similar(Y), similar(Y), similar(Y)

    rhs!(dY, Y, params, FT(0))                 # warm up / sanity
    @test all(!isnan, parent(dY.Yc.ρ))

    ssp33!(rhs!, Y, dY, Y1, Y2, params, dt, nsteps)

    uₕ_phy = Geometry.transform.(Ref(Geometry.UVAxis()), Y.uₕ)
    w_phy = Geometry.transform.(Ref(Geometry.WAxis()), Y.w)

    # The balanced state stays at rest: velocities remain near machine zero
    # (observed ~1e-13 after these steps; bound left generous for portability).
    @test maximum(abs, parent(uₕ_phy.components.data.:1)) ≤ 1e-10
    @test maximum(abs, parent(uₕ_phy.components.data.:2)) ≤ 1e-10
    @test maximum(abs, parent(w_phy)) ≤ 1e-10
    # Mass and energy norms are preserved.
    @test norm(Y.Yc.ρ) ≈ norm(Y0.Yc.ρ) rtol = 1e-2
    @test norm(Y.Yc.ρe) ≈ norm(Y0.Yc.ρe) rtol = 1e-2
end
