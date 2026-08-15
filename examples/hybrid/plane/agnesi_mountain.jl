# Linear mountain waves: uniform stratified flow (N = 0.01 s⁻¹) over a
# witch-of-Agnesi hill, following Ullrich and Guerra [2016, GMD]. The hill is
# only 1 m high, so the response stays in the linear regime and can be compared
# against linear theory. Exercises the terrain-following (`LinearAdaption`) mesh
# and the sponge layers that absorb waves at the lateral and upper boundaries.
using Test
using StaticArrays, IntervalSets, LinearAlgebra

import ClimaComms
ClimaComms.@import_required_backends
include("plane_utils.jl")
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
    Hypsography
using ClimaCore.Geometry
import LazyBroadcast: lazy


using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

const kinematic_viscosity = 75.0 #m²/s
const hyperdiffusivity = 1e7 #m²/s

const u₀ = 10.0 # initial horizontal wind (m/s)

# Unlike `schar_mountain.jl`, this case does not project the surface velocity
# onto the terrain (u₃ = -g³¹u₁/g³³) and does not zero the surface `w`
# tendency. The hill is 1 m over a 1000 m half-width, so |g³¹/g³³| on the
# bottom face reaches 7.2e-4 and the coordinate-normal and surface-normal
# directions agree to that accuracy; the case exists to check the linear
# mountain-wave response, whose peak |w| is 3e-3 m/s. On the 250 m Schar ridge
# the same ratio is 0.059, which is why that case imposes the constraint.
function warp_surface(coord)
    # Parameters from GMD-9-2007-2016
    # Specification for Agnesi Mountain following 
    # Ulrich and Guerra [2016 GMD]
    x = Geometry.component(coord, 1)
    FT = eltype(x)
    ac = 1000
    hc = 1.0
    return hc / (1 + (x / ac)^2)
end


# set up 2D domain - doubly periodic box
const xmin = -72000.0
const xmax = 72000.0
hv_center_space, hv_face_space =
    hvspace_2D(
        (xmin, xmax),
        (0, 25000);
        xelem = 32,
        zelem = 25,
        warp_fn = warp_surface,
    )

geopotential(z) = grav * z

# Prognostic thermodynamic variable: Total Energy 
function init_advection_over_mountain(x, z)
    θ₀ = 280.0
    cp_d = C_p
    cv_d = C_v
    p₀ = MSLP
    g = grav

    N = 0.01
    θ = @. θ₀ * exp(N^2 * z / g)
    π_exner = @. 1 + g^2 / N^2 / cp_d / θ₀ * (exp(-N^2 * z / g) - 1)
    T = @. π_exner * θ # temperature
    # p = p₀ π^(cp/R) and p = ρ R_d T = ρ R_d π θ, so ρ carries π^(cp/R - 1),
    # and cp - R_d = cv.
    ρ = @. p₀ / (R_d * θ) * (π_exner)^(cv_d / R_d)
    # total energy: internal + potential + kinetic energy of the initial wind
    e = @. cv_d * (T - T_0) + geopotential(z) + u₀^2 / 2
    ρe = @. ρ * e
    return (ρ = ρ, ρe = ρe)
end

# initial conditions
coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

# Assign initial conditions to cell center, cell face variables Group scalars
# (ρ, ρe) in Yc Retain uₕ and w as separate components of velocity vector
# (primitive variables)
Yc = map(coord -> init_advection_over_mountain(coord.x, coord.z), coords)
w = map(_ -> Geometry.Covariant3Vector(0.0), face_coords)
uₕ_local = map(_ -> Geometry.UWVector(u₀, 0.0), coords)
uₕ = Geometry.Covariant1Vector.(uₕ_local)

# A snapshot, not an alias: the sponge relaxes toward the *initial* wind.
const u_init = copy(uₕ)

ᶜlg = Fields.local_geometry_field(hv_center_space)
ᶠlg = Fields.local_geometry_field(hv_face_space)

Y = Fields.FieldVector(Yc = Yc, uₕ = uₕ, w = w)

energy_0 = sum(Y.Yc.ρe)
mass_0 = sum(Y.Yc.ρ)

function rayleigh_sponge_z(
    z;
    z_sponge = 15000.0,
    z_max = 25000.0,
    α = 0.1,  # Relaxation timescale
    τ = 0.1,
    γ = 2.0,
)
    if z >= z_sponge
        r = (z - z_sponge) / (z_max - z_sponge)
        β_sponge = α * sinpi(τ * r)^γ
        return β_sponge
    else
        return eltype(z)(0)
    end
end
function rhs_invariant!(dY, Y, _, t)

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

    dρ .= 0 .* cρ

    cw = If2c.(fw)
    fuₕ = Ic2f.(cuₕ)
    cuw = Geometry.Covariant13Vector.(cuₕ) .+ Geometry.Covariant13Vector.(cw)

    ce = @. cρe / cρ
    cI = @. ce - geopotential(z) - (norm(cuw)^2) / 2
    cT = @. cI / C_v + T_0
    cp = @. cρ * R_d * cT

    h_tot = @. ce + cp / cρ # Total enthalpy at cell centers

    ### HYPERVISCOSITY
    # 1) compute hyperviscosity coefficients
    χe = @. dρe = hwdiv(hgrad(h_tot)) # we store χe in dρe
    χuₕ = @. duₕ = hwgrad(hdiv(cuₕ))

    Spaces.weighted_dss!(dρe)
    Spaces.weighted_dss!(duₕ)

    κ₄ = hyperdiffusivity # m^4/s
    @. dρe = -κ₄ * hwdiv(cρ * hgrad(χe))
    @. duₕ = -κ₄ * (hwgrad(hdiv(χuₕ)))

    # 1) Mass conservation
    dw .= fw .* 0

    # 1.a) horizontal divergence
    dρ .-= hwdiv.(cρ .* (cuw))

    # 1.b) vertical divergence
    vdivf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    )
    # we want the total u³ at the boundary to be zero: we can either constrain
    # both to be zero, or allow one to be non-zero and set the other to be its
    # negation

    # explicit part
    dρ .-= vdivf2c.(Ic2f.(cρ .* cuₕ))
    # implicit part
    dρ .-= vdivf2c.(Ic2f.(cρ) .* fw)

    # 2) Momentum equation

    # curl term
    hcurl = Operators.Curl()
    # effectively a homogeneous Dirichlet condition on u₁ at the boundary
    vcurlc2f = Operators.CurlC2F(
        bottom = Operators.SetCurl(Geometry.Contravariant2Vector(0.0)),
        top = Operators.SetCurl(Geometry.Contravariant2Vector(0.0)),
    )

    fω¹ = hcurl.(fw)
    fω¹ .+= vcurlc2f.(cuₕ)

    # cross product
    # convert to contravariant
    fu =
        Geometry.Contravariant13Vector.(Ic2f.(cuₕ)) .+
        Geometry.Contravariant13Vector.(fw)
    fu¹ = Geometry.project.(Ref(Geometry.Contravariant1Axis()), fu)
    fu³ = Geometry.project.(Ref(Geometry.Contravariant3Axis()), fu)
    @. dw -= fω¹ × fu¹ # Covariant3Vector on faces
    @. duₕ -= If2c(fω¹ × fu³)


    @. duₕ -= hgrad(cp) / cρ
    vgradc2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
        top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
    )
    @. dw -= vgradc2f(cp) / Ic2f(cρ)

    cE = @. (norm(cuw)^2) / 2 + geopotential(z)
    @. duₕ -= hgrad(cE)
    @. dw -= vgradc2f(cE)

    # 3) total energy

    @. dρe -= hwdiv(cuw * (cρe + cp))
    @. dρe -= vdivf2c(fw * Ic2f(cρe + cp))
    @. dρe -= vdivf2c(Ic2f(cuₕ * (cρe + cp)))

    # Uniform 2nd order diffusion
    ∂c = Operators.GradientF2C()
    fρ = @. Ic2f(cρ)
    κ₂ = kinematic_viscosity # m^2/s

    ᶠ∇ᵥuₕ = @. vgradc2f(cuₕ.components.data.:1)
    ᶜ∇ᵥw = @. ∂c(fw.components.data.:1)
    ᶠ∇ᵥh_tot = @. vgradc2f(h_tot)

    ᶜ∇ₕuₕ = @. hgrad(cuₕ.components.data.:1)
    ᶠ∇ₕw = @. hgrad(fw.components.data.:1)
    ᶜ∇ₕh_tot = @. hgrad(h_tot)

    hκ₂∇²uₕ = @. hwdiv(κ₂ * ᶜ∇ₕuₕ)
    vκ₂∇²uₕ = @. vdivf2c(κ₂ * ᶠ∇ᵥuₕ)
    hκ₂∇²w = @. hwdiv(κ₂ * ᶠ∇ₕw)
    lg_field_faces = Fields.local_geometry_field(axes(fw))
    lg_field_centers = Fields.local_geometry_field(axes(cρ))
    # Only `J` on the boundary faces is needed below, on the same level space
    # as the center quantities, so the face `J` (a scalar field) is shifted
    # onto centers: `LeftBiasedF2C(x)[i] = x[i-half]`, so its first level is
    # the bottom face, and `RightBiasedF2C(x)[i] = x[i+half]`, so its last
    # level is the top face. The whole `LocalGeometry` field cannot be shifted
    # instead, because a finite difference operator multiplies its argument by
    # an operator matrix row.
    J_bottom_face = Fields.level(Operators.LeftBiasedF2C().(lg_field_faces.J), 1)
    J_top_face = Fields.level(
        Operators.RightBiasedF2C().(lg_field_faces.J),
        Fields.nlevels(lg_field_centers),
    )
    lg_bottom_center = Fields.level(lg_field_centers, 1)
    lg_top_center = Fields.level(lg_field_centers, Fields.nlevels(lg_field_centers))
    ᶜ∇ᵥw_bottom = Fields.level(ᶜ∇ᵥw, 1)
    ᶜ∇ᵥw_top = Fields.level(ᶜ∇ᵥw, Fields.nlevels(ᶜ∇ᵥw))
    bottom_divergence = @. lazy(
        Geometry.Jcontravariant3(κ₂ * ᶜ∇ᵥw_bottom, lg_bottom_center) *
        (2 * inv(J_bottom_face)),
    )
    top_divergence = @. lazy(
        Geometry.Jcontravariant3(κ₂ * ᶜ∇ᵥw_top, lg_top_center) *
        (-2 * inv(J_top_face)),
    )
    vdivc2f_bcs = Operators.DivergenceC2F(
        bottom = Operators.SetDivergence(bottom_divergence),
        top = Operators.SetDivergence(top_divergence),
    )
    vκ₂∇²w = @. vdivc2f_bcs(κ₂ * ᶜ∇ᵥw)
    hκ₂∇²h_tot = @. hwdiv(cρ * κ₂ * ᶜ∇ₕh_tot)
    vκ₂∇²h_tot = @. vdivf2c(fρ * κ₂ * ᶠ∇ᵥh_tot)

    dfw = dY.w.components.data.:1
    dcu = dY.uₕ.components.data.:1

    # Laplacian Diffusion (Uniform)
    @. dcu += hκ₂∇²uₕ
    @. dcu += vκ₂∇²uₕ
    @. dfw += hκ₂∇²w
    @. dfw += vκ₂∇²w
    @. dρe += hκ₂∇²h_tot
    @. dρe += vκ₂∇²h_tot

    # Sponge tendency
    β = @. rayleigh_sponge_z(coords.z)
    @. duₕ -= β * (cuₕ - u_init)
    @. dw -= Ic2f(β) * fw

    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.uₕ)
    Spaces.weighted_dss!(dY.w)

    return dY
end

dYdt = similar(Y);
rhs_invariant!(dYdt, Y, nothing, 0.0);

# run!
import ClimaTimeSteppers as CTS
Δt = 1.0
timeend = 72000.0
# ClimaTimeSteppers calls `dss!` after every stage; a FieldVector may hold
# non-Field entries (e.g. scalars), which need no DSS.
_dss!(x::Fields.Field) = Spaces.weighted_dss!(x)
_dss!(::Any) = nothing
dss!(Y, parameters, t) = foreach(_dss!, Fields._values(Y))
prob = CTS.ODEProblem(
    CTS.ClimaODEFunction(; T_exp! = rhs_invariant!, dss!),
    Y,
    (0.0, timeend),
    nothing,
)
integrator = CTS.init(
    prob,
    CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()),
    dt = Δt,
    saveat = collect(0.0:1800.0:timeend),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

if haskey(ENV, "CI_PERF_SKIP_RUN") # for performance analysis
    throw(:exit_profile)
end

sol = @timev CTS.solve!(integrator)

# Mass and total energy must be conserved (measured drift: 8e-14 and 2e-13).
# The 1 m hill in a 10 m/s flow forces mountain waves of a few mm/s
# (measured max|w| = 3e-3); an O(1) value means the surface condition broke.
@test abs(sum(sol.u[end].Yc.ρ) - sum(sol.u[1].Yc.ρ)) / sum(sol.u[1].Yc.ρ) < 1e-11
@test abs(sum(sol.u[end].Yc.ρe) - sum(sol.u[1].Yc.ρe)) / sum(sol.u[1].Yc.ρe) < 1e-11
@testset "mountain-wave amplitude" begin
    w = maximum(abs, parent(Geometry.WVector.(sol.u[end].w)))
    @info "Peak |w| at the end of the run: $w m/s"
    @test 1e-4 < w < 0.1
end

ENV["GKSwstype"] = "nul"
import Plots, ClimaCorePlots
Plots.GRBackend()

dir = "agnesi_etot_nh"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.Yc.ρe ./ u.Yc.ρ)
end
Plots.mp4(anim, joinpath(path, "total_energy.mp4"), fps = 20)

If2c = Operators.InterpolateF2C()
anim = Plots.@animate for u in sol.u
    Plots.plot(Geometry.WVector.(Geometry.Covariant13Vector.(If2c.(u.w))))
end
Plots.mp4(anim, joinpath(path, "vel_w.mp4"), fps = 20)

anim = Plots.@animate for u in sol.u
    Plots.plot(
        Geometry.UVector.(Geometry.Covariant13Vector.(u.uₕ)) .-
        Geometry.UVector.(10.0),
    )
end
Plots.mp4(anim, joinpath(path, "vel_u.mp4"), fps = 20)

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

include(joinpath(@__DIR__, "../..", "example_utils.jl")) # linkfig

linkfig(
    relpath(joinpath(path, "energy_cons.png"), joinpath(@__DIR__, "../../..")),
    "Total Energy",
)
linkfig(
    relpath(joinpath(path, "mass_cons.png"), joinpath(@__DIR__, "../../..")),
    "Mass",
)
