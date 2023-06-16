using Test
using StaticArrays, IntervalSets, LinearAlgebra, UnPack

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
    Hypsography
using ClimaCore.Geometry

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

function warp_surface(coord)
    x = Geometry.component(coord, 1)
    FT = eltype(x)
    ac = 5000
    hc = 4000.0
    h = hc / (1 + (x / ac)^2)
    return h
end

function no_warp(coord)
    x = Geometry.component(coord, 1)
    FT = eltype(x)
    return FT(0) * x
end

function hvspace_2D(
    xlim = (-π, π),
    zlim = (0, 4π),
    xelem = 30,
    zelem = 30,
    npoly = 4,
    warp_fn = warp_surface,
)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
    vert_face_space = Spaces.FaceFiniteDifferenceSpace(vertmesh)

    horzdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1]),
        Geometry.XPoint{FT}(xlim[2]);
        periodic = true,
    )
    horzmesh = Meshes.IntervalMesh(horzdomain, nelems = xelem)
    horztopology = Topologies.IntervalTopology(horzmesh)
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)
    z_surface = warp_fn.(Fields.coordinate_field(horzspace))
    hv_face_space = Spaces.ExtrudedFiniteDifferenceSpace(
        horzspace,
        vert_face_space,
        Hypsography.LinearAdaption(z_surface),
    )
    hv_center_space = Spaces.CenterExtrudedFiniteDifferenceSpace(hv_face_space)
    return (hv_center_space, hv_face_space)
end

# set up 2D domain - doubly periodic box
hv_center_space, hv_face_space = hvspace_2D((-25600, 25600), (0, 25000))

const MSLP = 1e5 # mean sea level pressure
const grav = 9.8 # gravitational constant
const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const γ = 1.4 # heat capacity ratio
const C_p = R_d * γ / (γ - 1) # heat capacity at constant pressure
const C_v = R_d / (γ - 1) # heat capacity at constant volume
const T_0 = 273.16 # triple point temperature

Φ(z) = grav * z

# Reference: https://journals.ametsoc.org/view/journals/mwre/140/4/mwr-d-10-05073.1.xml, Section 5a
# Prognostic thermodynamic variable: Total Energy
function init_dry_density_current_2d(x, z)
    x_c = 0.0
    z_c = 7000.0
    r_c = 1.0
    x_r = 4000.0
    z_r = 2000.0
    θ_b = 300.0
    θ_c = -15.0
    cp_d = C_p
    cv_d = C_v
    p_0 = MSLP
    g = grav

    # auxiliary quantities
    r = sqrt((x - x_c)^2 / x_r^2 + (z - z_c)^2 / z_r^2)
    θ_p = r < r_c ? 0.5 * θ_c * (1.0 + cospi(r / r_c)) : 0.0 # potential temperature perturbation

    θ = θ_b + θ_p # potential temperature
    π_exn = 1.0 - Φ(z) / cp_d / θ # exner function
    T = π_exn * θ # temperature
    p = p_0 * π_exn^(cp_d / R_d) # pressure
    ρ = p / R_d / T # density
    e = cv_d * (T - T_0) + Φ(z)
    ρe = ρ * e # total energy

    return (ρ = ρ, ρe = ρe)
end

# initial conditions
coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

Yc = map(coord -> init_dry_density_current_2d(coord.x, coord.z), coords)
uₕ = map(_ -> Geometry.Covariant1Vector(0.0), coords)
w = map(_ -> Geometry.Covariant3Vector(0.0), face_coords)
Ic2f = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
# ==========
u₁_bc = Fields.level(Ic2f.(uₕ), ClimaCore.Utilities.half)
gⁱʲ =
    Fields.level(
        Fields.local_geometry_field(hv_face_space),
        ClimaCore.Utilities.half,
    ).gⁱʲ
g13 = gⁱʲ.components.data.:3
g11 = gⁱʲ.components.data.:1
g33 = gⁱʲ.components.data.:4
u₃_bc = Geometry.Covariant3Vector.(-1 .* g13 .* u₁_bc.components.data.:1 ./ g33)
apply_boundary_w =
    Operators.SetBoundaryOperator(bottom = Operators.SetValue(u₃_bc))
@. w = apply_boundary_w(w)
# ==========

Y = Fields.FieldVector(Yc = Yc, uₕ = uₕ, w = w)
energy_0 = sum(Y.Yc.ρe)
mass_0 = sum(Y.Yc.ρ)

function rhs_invariant!(dY, Y, p, t)

    (; fρ, ᶠ∇ᵥuₕ, ᶜ∇ᵥw, ᶠ∇ᵥh_tot, ᶜ∇ₕuₕ, ᶠ∇ₕw, ᶜ∇ₕh_tot) = p
    (; hκ₂∇²uₕ, vκ₂∇²uₕ, hκ₂∇²w, vκ₂∇²w, hκ₂∇²h_tot) = p
    (; vκ₂∇²h_tot, cE, fu¹, fu³, fu, cω², fω², h_tot) = p
    (; ce, cI, cT, cp, cuw, cw, u₃_bc, fuₕ, coords) = p

    cρ = Y.Yc.ρ # scalar on centers
    fw = Y.w # Covariant3Vector on faces
    cuₕ = Y.uₕ # Covariant1Vector on centers
    cρe = Y.Yc.ρe

    dρ = dY.Yc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρe = dY.Yc.ρe
    z = coords.z

    # 0) update w at the bottom
    # fw = -g^31 cuₕ/ g^33

    hdiv = Operators.Divergence()
    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()
    hwgrad = Operators.WeakGradient()
    hcurl = Operators.Curl()

    If2c = Operators.InterpolateF2C()
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )

    @. dρ = 0 * cρ

    @. cw = If2c(fw)
    @. fuₕ = Ic2f(cuₕ)
    # ==========
    u₁_bc = Fields.level(fuₕ, ClimaCore.Utilities.half)
    gⁱʲ =
        Fields.level(
            Fields.local_geometry_field(hv_face_space),
            ClimaCore.Utilities.half,
        ).gⁱʲ
    g13 = gⁱʲ.components.data.:3
    g11 = gⁱʲ.components.data.:1
    g33 = gⁱʲ.components.data.:4
    u₁_bc₁ = u₁_bc.components.data.:1
    @. u₃_bc = Geometry.Covariant3Vector(-1 * g13 / g33 * u₁_bc₁)
    apply_boundary_w =
        Operators.SetBoundaryOperator(bottom = Operators.SetValue(u₃_bc))
    @. fw = apply_boundary_w(fw)
    Spaces.weighted_dss!(fw)
    # ==========
    @. cw = If2c(fw)
    @. cuw = Geometry.Covariant13Vector(cuₕ) + Geometry.Covariant13Vector(cw)

    @. ce = cρe / cρ
    @. cI = ce - Φ(z) - (norm(cuw)^2) / 2
    @. cT = cI / C_v + T_0
    @. cp = cρ * R_d * cT

    @. h_tot = ce + cp / cρ # Total enthalpy at cell centers

    ### HYPERVISCOSITY
    # 1) compute hyperviscosity coefficients
    # Move to cache if necessary
    χe = @. dρe = hwdiv(hgrad(h_tot)) # we store χe in dρe
    χuₕ = @. duₕ = hwgrad(hdiv(cuₕ))

    Spaces.weighted_dss!(dρe)
    Spaces.weighted_dss!(duₕ)

    κ₄ = 1e8 # m^4/s
    @. dρe = -κ₄ * hwdiv(cρ * hgrad(χe))
    @. duₕ = -κ₄ * (hwgrad(hdiv(χuₕ)))

    # 1) Mass conservation
    @. dw = fw * 0

    # 1.a) horizontal divergence
    @. dρ -= hdiv(cρ * (cuw))

    # 1.b) vertical divergence
    vdivf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    )
    vdivc2f = Operators.DivergenceC2F(
        top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    )
    # we want the total u³ at the boundary to be zero: we can either constrain
    # both to be zero, or allow one to be non-zero and set the other to be its
    # negation

    # explicit part
    @. dρ -= vdivf2c(Ic2f(cρ * cuₕ))
    # implicit part
    @. dρ -= vdivf2c(Ic2f(cρ) * fw)

    # 2) Momentum equation

    # curl term
    hcurl = Operators.Curl()
    # effectively a homogeneous Dirichlet condition on u₁ at the boundary
    vcurlc2f = Operators.CurlC2F(
        bottom = Operators.SetCurl(Geometry.Contravariant2Vector(0.0)),
        top = Operators.SetCurl(Geometry.Contravariant2Vector(0.0)),
    )

    @. fω² = hcurl(fw)
    @. fω² += vcurlc2f(cuₕ)

    @. cω² = hcurl(cw) # Compute new center curl
    @. cω² += If2c(vcurlc2f(cuₕ)) # Compute new centerl curl

    # Linearly interpolate the horizontal velocities multiplying the curl terms from centers to faces 
    # (i.e., u_f^i = (u_c^{i+1} + u_c^{i})/2)
    # Leave the horizontal curl terms (living on faces) untouched.

    # cross product
    # convert to contravariant
    # these will need to be modified with topography
    @. fu =
        Geometry.Covariant13Vector(Ic2f(cuₕ)) + Geometry.Covariant13Vector(fw)
    contra1 = (Geometry.Contravariant1Axis(),)
    contra3 = (Geometry.Contravariant3Axis(),)
    @. fu¹ = Geometry.project(contra1, fu)
    @. fu³ = Geometry.project(contra3, fu)

    @. duₕ -= If2c(fω² × fu³)
    @. dw -= fω² × fu¹ # Covariant3Vector on faces

    @. duₕ -= hgrad(cp) / cρ
    vgradc2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
        top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
    )
    @. dw -= vgradc2f(cp) / Ic2f(cρ)

    @. cE = (norm(cuw)^2) / 2 + Φ(z)
    @. duₕ -= hgrad(cE)
    @. dw -= vgradc2f(cE)

    # 3) total energy
    @. dρe -= hdiv(cuw * (cρe + cp))
    @. dρe -= vdivf2c(fw * Ic2f(cρe + cp))
    @. dρe -= vdivf2c(Ic2f(cuₕ * (cρe + cp)))

    # Uniform 2nd order diffusion
    ∂c = Operators.GradientF2C()
    @. fρ = Ic2f(cρ)
    κ₂ = 0.0 # m^2/s

    @. ᶠ∇ᵥuₕ = vgradc2f(cuₕ.components.data.:1)
    @. ᶜ∇ᵥw = ∂c(fw.components.data.:1)
    @. ᶠ∇ᵥh_tot = vgradc2f(h_tot)

    @. ᶜ∇ₕuₕ = hgrad(cuₕ.components.data.:1)
    @. ᶠ∇ₕw = hgrad(fw.components.data.:1)
    @. ᶜ∇ₕh_tot = hgrad(h_tot)

    @. hκ₂∇²uₕ = hwdiv(κ₂ * ᶜ∇ₕuₕ)
    @. vκ₂∇²uₕ = vdivf2c(κ₂ * ᶠ∇ᵥuₕ)
    @. hκ₂∇²w = hwdiv(κ₂ * ᶠ∇ₕw)
    @. vκ₂∇²w = vdivc2f(κ₂ * ᶜ∇ᵥw)
    @. hκ₂∇²h_tot = hwdiv(cρ * κ₂ * ᶜ∇ₕh_tot)
    @. vκ₂∇²h_tot = vdivf2c(fρ * κ₂ * ᶠ∇ᵥh_tot)

    dfw = dY.w.components.data.:1
    dcu = dY.uₕ.components.data.:1

    # Laplacian Diffusion (Uniform)
    @. dcu += hκ₂∇²uₕ
    @. dcu += vκ₂∇²uₕ
    @. dfw += hκ₂∇²w
    @. dfw += vκ₂∇²w
    @. dρe += hκ₂∇²h_tot
    @. dρe += vκ₂∇²h_tot

    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.uₕ)
    Spaces.weighted_dss!(dY.w)

    return dY
end

function get_cache(Y)
    cρ = Y.Yc.ρ # scalar on centers
    fw = Y.w # Covariant3Vector on faces
    cuₕ = Y.uₕ # Covariant1Vector on centers
    cρe = Y.Yc.ρe

    z = coords.z

    # 0) update w at the bottom
    # fw = -g^31 cuₕ/ g^33

    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()
    hcurl = Operators.Curl()

    If2c = Operators.InterpolateF2C()
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )

    fuₕ = Ic2f.(cuₕ)
    # ==========
    u₁_bc = Fields.level(fuₕ, ClimaCore.Utilities.half)
    gⁱʲ =
        Fields.level(
            Fields.local_geometry_field(hv_face_space),
            ClimaCore.Utilities.half,
        ).gⁱʲ
    g13 = gⁱʲ.components.data.:3
    g11 = gⁱʲ.components.data.:1
    g33 = gⁱʲ.components.data.:4

    u₁_bc₁ = u₁_bc.components.data.:1
    u₃_bc = @. Geometry.Covariant3Vector(-1 * g13 / g33 * u₁_bc₁)
    # ==========
    cw = If2c.(fw)
    cuw = Geometry.Covariant13Vector.(cuₕ) .+ Geometry.Covariant13Vector.(cw)

    ce = @. cρe / cρ
    cI = @. ce - Φ(z) - (norm(cuw)^2) / 2
    cT = @. cI / C_v + T_0
    cp = @. cρ * R_d * cT
    h_tot = @. ce + cp / cρ # Total enthalpy at cell centers
    # 1.b) vertical divergence
    vdivf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    )
    vdivc2f = Operators.DivergenceC2F(
        top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    )

    fω² = hcurl.(fw)
    cω² = hcurl.(cw) # Compute new center curl

    # Linearly interpolate the horizontal velocities multiplying the curl terms from centers to faces 
    # (i.e., u_f^i = (u_c^{i+1} + u_c^{i})/2)
    # Leave the horizontal curl terms (living on faces) untouched.

    # cross product
    # convert to contravariant
    # these will need to be modified with topography
    fu =
        Geometry.Covariant13Vector.(Ic2f.(cuₕ)) .+
        Geometry.Covariant13Vector.(fw)
    fu¹ = Geometry.project.(Ref(Geometry.Contravariant1Axis()), fu)
    fu³ = Geometry.project.(Ref(Geometry.Contravariant3Axis()), fu)

    vgradc2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
        top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
    )

    cE = @. (norm(cuw)^2) / 2 + Φ(z)

    # Uniform 2nd order diffusion
    ∂c = Operators.GradientF2C()
    fρ = @. Ic2f(cρ)

    ᶠ∇ᵥuₕ = @. vgradc2f(cuₕ.components.data.:1)
    ᶜ∇ᵥw = @. ∂c(fw.components.data.:1)
    ᶠ∇ᵥh_tot = @. vgradc2f(h_tot)

    ᶜ∇ₕuₕ = @. hgrad(cuₕ.components.data.:1)
    ᶠ∇ₕw = @. hgrad(fw.components.data.:1)
    ᶜ∇ₕh_tot = @. hgrad(h_tot)

    hκ₂∇²uₕ = @. hwdiv(ᶜ∇ₕuₕ)
    vκ₂∇²uₕ = @. vdivf2c(ᶠ∇ᵥuₕ)
    hκ₂∇²w = @. hwdiv(ᶠ∇ₕw)
    vκ₂∇²w = @. vdivc2f(ᶜ∇ᵥw)
    hκ₂∇²h_tot = @. hwdiv(cρ * ᶜ∇ₕh_tot)
    vκ₂∇²h_tot = @. vdivf2c(fρ * ᶠ∇ᵥh_tot)

    p1 = (; fρ, ᶠ∇ᵥuₕ, ᶜ∇ᵥw, ᶠ∇ᵥh_tot, ᶜ∇ₕuₕ, ᶠ∇ₕw, ᶜ∇ₕh_tot)
    p2 = (; hκ₂∇²uₕ, vκ₂∇²uₕ, hκ₂∇²w, vκ₂∇²w, hκ₂∇²h_tot)
    p3 = (; vκ₂∇²h_tot, cE, fu¹, fu³, fu, cω², fω², h_tot)
    p4 = (; ce, cI, cT, cp, cuw, cw, u₃_bc, fuₕ, coords)

    return merge(p1, p2, p3, p4)
end

p = get_cache(Y);

dYdt = similar(Y);
rhs_invariant!(dYdt, Y, p, 0.0);

# run!
using OrdinaryDiffEq
timeend = 3600.0 * 3
Δt = 0.05
prob = ODEProblem(rhs_invariant!, Y, (0.0, timeend), p)
integrator = OrdinaryDiffEq.init(
    prob,
    SSPRK33(),
    dt = Δt,
    saveat = 10.0,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

if haskey(ENV, "CI_PERF_SKIP_RUN") # for performance analysis
    throw(:exit_profile)
end

sol = @timev OrdinaryDiffEq.solve!(integrator)

ENV["GKSwstype"] = "nul"
import Plots, ClimaCorePlots
Plots.GRBackend()

dir = "dc_invariant_etot_warp"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.Yc.ρe ./ u.Yc.ρ)
end
Plots.mp4(anim, joinpath(path, "total_energy.mp4"), fps = 20)

Ic2f = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
If2c = Operators.InterpolateF2C()
anim = Plots.@animate for u in sol.u
    ᶠuw = @. Geometry.Covariant13Vector(Ic2f.(u.uₕ)) +
       Geometry.Covariant13Vector(u.w)
    w = @. Geometry.project(Geometry.WAxis(), ᶠuw)
    Plots.plot(w)
end
Plots.mp4(anim, joinpath(path, "vel_w.mp4"), fps = 20)

anim = Plots.@animate for u in sol.u
    ᶠuw = @. Geometry.Covariant13Vector(Ic2f.(u.uₕ)) +
       Geometry.Covariant13Vector(u.w)
    w = @. Geometry.project(Geometry.WAxis(), ᶠuw)
    Plots.plot(Fields.level(w, ClimaCore.Utilities.half))
end
Plots.mp4(anim, joinpath(path, "vel_w_level1.mp4"), fps = 20)

anim = Plots.@animate for u in sol.u
    ᶠuw = @. Geometry.Covariant13Vector(Ic2f.(u.uₕ)) +
       Geometry.Covariant13Vector(u.w)
    u = @. Geometry.project(Geometry.UAxis(), ᶠuw)
    Plots.plot(Fields.level(u, ClimaCore.Utilities.half))
end
Plots.mp4(anim, joinpath(path, "vel_fu_level1.mp4"), fps = 20)

anim = Plots.@animate for u in sol.u
    ᶠuw = @. Geometry.Covariant13Vector(Ic2f.(u.uₕ)) +
       Geometry.Covariant13Vector(u.w)
    u = @. Geometry.project(Geometry.Covariant1Axis(), ᶠuw)
    Plots.plot(Fields.level(u, ClimaCore.Utilities.half))
end
Plots.mp4(anim, joinpath(path, "vel_fcovariant1_level1.mp4"), fps = 20)

anim = Plots.@animate for u in sol.u
    ᶠuw = @. Geometry.Covariant13Vector(Ic2f.(u.uₕ)) +
       Geometry.Covariant13Vector(u.w)
    u = @. Geometry.project(Geometry.Covariant3Axis(), ᶠuw)
    Plots.plot(Fields.level(u, ClimaCore.Utilities.half))
end
Plots.mp4(anim, joinpath(path, "vel_fcovariant3_level1.mp4"), fps = 20)

anim = Plots.@animate for u in sol.u
    ᶜuw = @. Geometry.Covariant13Vector(u.uₕ) +
       Geometry.Covariant13Vector(If2c.(u.w))
    w = @. Geometry.project(Geometry.WAxis(), ᶜuw)
    Plots.plot(Fields.level(w, 1))
end
Plots.mp4(anim, joinpath(path, "vel_cw_level1.mp4"), fps = 20)

anim = Plots.@animate for u in sol.u
    ᶜuw = @. Geometry.Covariant13Vector(u.uₕ) +
       Geometry.Covariant13Vector(If2c.(u.w))
    u = @. Geometry.project(Geometry.UAxis(), ᶜuw)
    Plots.plot(Fields.level(u, 1))
end
Plots.mp4(anim, joinpath(path, "vel_cu_level1.mp4"), fps = 20)

anim = Plots.@animate for u in sol.u
    ᶠuw = @. Geometry.Covariant13Vector(Ic2f.(u.uₕ)) +
       Geometry.Covariant13Vector(u.w)
    w = @. Geometry.project(Geometry.WAxis(), ᶠuw)
    Plots.plot(Fields.level(w, ClimaCore.Utilities.half + 1))
end
Plots.mp4(anim, joinpath(path, "vel_w_level2.mp4"), fps = 20)

anim = Plots.@animate for u in sol.u
    ᶜuw = @. Geometry.Covariant13Vector(u.uₕ) +
       Geometry.Covariant13Vector(If2c(u.w))
    u = @. Geometry.project(Geometry.UAxis(), ᶜuw)
    Plots.plot(u)
end
Plots.mp4(anim, joinpath(path, "vel_u.mp4"), fps = 20)

anim = Plots.@animate for u in sol.u
    ᶠu = @. Geometry.Covariant13Vector(Ic2f(u.uₕ))
    ᶠw = @. Geometry.Covariant13Vector(u.w)
    ᶠuw = @. ᶠu + ᶠw
    w = @. Geometry.project(Geometry.Contravariant3Axis(), ᶠu) +
       Geometry.project(Geometry.Contravariant3Axis(), ᶠw)
    Plots.plot(Fields.level(w, ClimaCore.Utilities.half))
end
Plots.mp4(anim, joinpath(path, "contravariant3_level1.mp4"), fps = 20)

# post-processing
Es = [sum(u.Yc.ρe) for u in sol.u]
Mass = [sum(u.Yc.ρ) for u in sol.u]

Plots.png(
    Plots.plot((Es .- energy_0) ./ energy_0),
    joinpath(path, "energy_cons.png"),
)
Plots.png(
    Plots.plot((Mass .- mass_0) ./ mass_0),
    joinpath(path, "mass_cons.png"),
)

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig(
    relpath(joinpath(path, "energy_cons.png"), joinpath(@__DIR__, "../..")),
    "Total Energy",
)
linkfig(
    relpath(joinpath(path, "mass_cons.png"), joinpath(@__DIR__, "../..")),
    "Mass",
)
