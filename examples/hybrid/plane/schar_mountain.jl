# Schär mountain waves: stratified flow over a sinusoidally modulated Gaussian
# ridge, from Schär et al. (2002), Section 3(b). The ridge is tall enough and
# narrow enough that the terrain-following coordinate is strongly distorted near
# the surface, which is what makes this a demanding test of the metric terms.
# The steady solution has |w| ≈ 2 m/s, and mass and total energy are conserved
# to roundoff over the 15 h run.
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
using ClimaCore.Utilities: half
import LazyBroadcast: lazy


using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())


### This file follows the test problem described in
# https://doi.org/10.1175/1520-0493(2002)130<2459:ANTFVC>2.0.CO;2
# Section 3(b)

const kinematic_viscosity = 75.0 # m²/s
const hyperdiffusivity = 2e7 #m²/s

const u₀ = 10.0 # initial horizontal wind (m/s)

function warp_schar(coord)
    x = Geometry.component(coord, 1)
    FT = eltype(x)
    a = 5000
    λ = 4000
    h₀ = 250.0
    if abs(x) <= a
        h = h₀ * exp(-(x / a)^2) * (cos(π * x / λ))^2
    else
        h = FT(0)
    end
end

const nx = 32
const nz = 40
const np = 4
const Lx = 120000
const Lz = 25000

# set up 2D domain - doubly periodic box
hv_center_space, hv_face_space = hvspace_2D(
    (-Lx / 2, Lx / 2),
    (0, Lz);
    xelem = nx,
    zelem = nz,
    npoly = np,
    warp_fn = warp_schar,
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
    ρq = @. ρ * 0.0
    return (ρ = ρ, ρe = ρe, ρq = ρq)
end

function initial_velocity(x, z)
    FT = eltype(x)
    return @. Geometry.UWVector(FT(u₀), FT(0))
end

# initial conditions
coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

# Assign initial conditions to cell center, cell face variables Group scalars
# (ρ, ρe) in Yc Retain uₕ and w as separate components of velocity vector
# (primitive variables)
Yc = map(coord -> init_advection_over_mountain(coord.x, coord.z), coords)
uₕ_local = map(coord -> initial_velocity(coord.x, coord.z), coords)
w = map(_ -> Geometry.Covariant3Vector(0.0), face_coords)
uₕ = Geometry.Covariant1Vector.(uₕ_local)

# A snapshot, not an alias: the sponge relaxes toward the *initial* wind.
const u_init = copy(uₕ)

ᶜlg = Fields.local_geometry_field(hv_center_space)
ᶠlg = Fields.local_geometry_field(hv_face_space)

Y = Fields.FieldVector(Yc = Yc, uₕ = uₕ, w = w)

Spaces.weighted_dss!(Y.Yc)
Spaces.weighted_dss!(Y.uₕ)
Spaces.weighted_dss!(Y.w)
Spaces.weighted_dss!(u_init)

energy_0 = sum(Y.Yc.ρe)
mass_0 = sum(Y.Yc.ρ)

function rayleigh_sponge(
    z;
    z_sponge = 12500.0,
    z_max = 25000.0,
    α = 0.5,  # Relaxation timescale
    τ = 0.5,
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
    cρq = Y.Yc.ρq

    dρ = dY.Yc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρe = dY.Yc.ρe
    dρq = dY.Yc.ρq
    z = coords.z
    fz = face_coords.z
    fx = face_coords.x

    # 0) update w at the bottom

    hdiv = Operators.Divergence()
    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()
    hwgrad = Operators.WeakGradient()
    hcurl = Operators.Curl()

    # get u_cov at first interior cell center
    # constant extrapolation to bottom face
    # apply as boundary condition on w for interpolation operator

    If2c = Operators.InterpolateF2C()
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )

    dρ .= 0 .* cρ

    cw = If2c.(fw)

    cuw = @. Geometry.Covariant13Vector(cuₕ) + Geometry.Covariant13Vector(cw)

    fuw = @. Ic2f(cuw)

    ce = @. cρe / cρ
    cq = @. cρq / cρ
    cI = @. ce - geopotential(z) - (norm(cuw)^2) / 2
    cT = @. cI / C_v + T_0
    cp = @. cρ * R_d * cT

    h_tot = @. ce + cp / cρ # Total enthalpy at cell centers

    ### HYPERVISCOSITY
    # 1) compute hyperviscosity coefficients
    χe = @. dρe = hwdiv(hgrad(h_tot)) # we store χq in dρq
    χq = @. dρq = hwdiv(hgrad(cq)) # we store χq in dρq
    χuₕ = @. duₕ = hwgrad(hdiv(cuₕ))

    Spaces.weighted_dss!(dρe)
    Spaces.weighted_dss!(duₕ)
    Spaces.weighted_dss!(dρq)

    κ₄_dynamic = hyperdiffusivity # m^4/s
    κ₄_tracer = hyperdiffusivity * 0
    @. dρe = -κ₄_dynamic * hwdiv(cρ * hgrad(χe))
    @. dρq = -κ₄_tracer * hwdiv(cρ * hgrad(χq))
    @. duₕ = -κ₄_dynamic * (hwgrad(hdiv(χuₕ)))

    # 1) Mass conservation
    dw .= fw .* 0

    # 1.a) horizontal divergence
    dρ .-= hwdiv.(cρ .* (cuw))

    # 1.b) vertical divergence

    # Apply n ⋅ ∇(X) = F
    # n^{i} * ∂X/∂_{x^{i}}
    # Contravariant3Vector(1) ⊗ (Flux Tensor)

    vdivf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    )
    vdivc2f = Operators.DivergenceC2F()
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
    # a homogeneous Neumann condition on u₁ at the boundary
    vcurlc2f = Operators.CurlC2F(
        bottom = Operators.SetCurl(Geometry.Contravariant2Vector(0.0)),
        top = Operators.SetCurl(Geometry.Contravariant2Vector(0.0)),
    )

    fω¹ = hcurl.(fw)
    fω¹ .+= vcurlc2f.(cuₕ)

    # cross product
    # convert to contravariant
    # these will need to be modified with topography

    fu¹ = @. Geometry.project(Geometry.Contravariant1Axis(), fuw)
    fu³ = @. Geometry.project(Geometry.Contravariant3Axis(), fuw)

    @. dw -= fω¹ × fu¹ # Covariant3Vector on faces
    @. duₕ -= If2c(fω¹ × fu³)
    @. duₕ -= hgrad(cp) / cρ

    vgradc2fP = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
        top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
    )
    vgradc2fE = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
        top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
    )

    cE = @. (norm(cuw)^2) / 2 + geopotential(z)
    @. duₕ -= hgrad(cE)
    @. dw -= vgradc2fE(cE)

    @. dw -= vgradc2fP(cp) / Ic2f(cρ)

    # 3) total energy

    @. dρe -= hwdiv(cuw * (cρe + cp))

    @. dρe -= vdivf2c(fw * Ic2f(cρe + cp))

    @. dρe -= vdivf2c(Ic2f(cuₕ * (cρe + cp)))

    # 4) tracer tendencies
    # In extruded grids
    @. dρq -= hwdiv(cuw * (cρq))
    @. dρq -= vdivf2c(fw * Ic2f(cρq))
    @. dρq -= vdivf2c(Ic2f(cuₕ * (cρq)))

    # Uniform 2nd order diffusion
    ∂c = Operators.GradientF2C()
    fρ = @. Ic2f(cρ)
    κ₂ = kinematic_viscosity # m^2/s

    ᶠ∇ᵥuₕ = @. vgradc2fE(cuₕ.components.data.:1)
    ᶜ∇ᵥw = @. ∂c(fw.components.data.:1)
    ᶠ∇ᵥh_tot = @. vgradc2fE(h_tot)

    ᶜ∇ₕuₕ = @. hgrad(cuₕ.components.data.:1)
    ᶠ∇ₕw = @. hgrad(fw.components.data.:1)
    ᶜ∇ₕh_tot = @. hgrad(h_tot)

    # `DivergenceC2F` no longer takes `SetValue` boundary conditions, so
    # evaluate its stencil on the boundary faces here, with the argument set to
    # zero outside of the domain, and impose the result with a
    # `SetBoundaryOperator`.
    lg_field_faces = Fields.local_geometry_field(axes(fw))
    lg_field_centers = Fields.local_geometry_field(axes(cρ))
    # `LeftBiasedF2C(x)[i] = x[i-half]`, so its first level is the bottom face,
    # and `RightBiasedF2C(x)[i] = x[i+half]`, so its last level is the top face.
    lg_bottom_face = Fields.level(Operators.LeftBiasedF2C().(lg_field_faces), 1)
    lg_top_face = Fields.level(
        Operators.RightBiasedF2C().(lg_field_faces),
        Fields.nlevels(lg_field_centers),
    )
    lg_bottom_center = Fields.level(lg_field_centers, 1)
    lg_top_center =
        Fields.level(lg_field_centers, Fields.nlevels(lg_field_centers))
    ᶜ∇ᵥw_bottom = Fields.level(ᶜ∇ᵥw, 1)
    ᶜ∇ᵥw_top = Fields.level(ᶜ∇ᵥw, Fields.nlevels(ᶜ∇ᵥw))
    bottom_divergence = @. lazy(
        Geometry.Jcontravariant3(κ₂ * ᶜ∇ᵥw_bottom, lg_bottom_center) *
        (2 * inv(lg_bottom_face.J)),
    )
    top_divergence = @. lazy(
        Geometry.Jcontravariant3(κ₂ * ᶜ∇ᵥw_top, lg_top_center) *
        (-2 * inv(lg_top_face.J)),
    )
    set_bcs = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(bottom_divergence),
        top = Operators.SetValue(top_divergence),
    )

    dfw = dY.w.components.data.:1
    dcu = dY.uₕ.components.data.:1

    @. dcu += hwdiv(κ₂ * ᶜ∇ₕuₕ)
    @. dcu += vdivf2c(κ₂ * ᶠ∇ᵥuₕ)
    @. dfw += hwdiv(κ₂ * ᶠ∇ₕw)
    @. dfw += set_bcs(vdivc2f(κ₂ * ᶜ∇ᵥw))
    @. dρe += hwdiv(cρ * κ₂ * ᶜ∇ₕh_tot)
    @. dρe += vdivf2c(fρ * κ₂ * ᶠ∇ᵥh_tot)

    # Sponge tendency
    β = @. rayleigh_sponge(z)
    βf = @. rayleigh_sponge(fz)
    @. duₕ -= β * (cuₕ - u_init)
    @. dw -= βf * fw

    # `w` at the surface is fixed by the free-slip terrain constraint applied at
    # the top of this function, so the vertical momentum equation must not drive
    # it off that constraint. Without this the surface value grows without
    # bound.
    apply_boundary_dw = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(Geometry.Covariant3Vector(0.0)),
    )
    @. dw = apply_boundary_dw(dw)

    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.uₕ)
    Spaces.weighted_dss!(dY.w)

    return dY
end
dYdt = similar(Y);
rhs_invariant!(dYdt, Y, nothing, 0.0);

# run!
import ClimaTimeSteppers as CTS
Δt = min(Lx / nx / np / 300, Lz / nz / 300) * 0.50
@info "Timestep Δt[s]: $(Δt)"

timeend = 3600.0 * 15.0
# The terrain follows the lower boundary, so the flow there must be tangent to
# it: contravariant u³ = g³¹u₁ + g³³u₃ = 0, that is u₃ = -g³¹u₁/g³³. That is a
# constraint on the state, not a tendency, so it is applied where the
# timestepper allows the state to be changed — after every stage, next to the
# DSS — rather than inside `rhs_invariant!`, which would mutate the stage value
# the integrator is still combining.
function project_surface_w!(Y)
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    face_geometry =
        Fields.level(
            Fields.local_geometry_field(hv_face_space),
            ClimaCore.Utilities.half,
        )
    u₁_bc =
        Geometry.contravariant3.(
            Fields.level(Ic2f.(Y.uₕ), ClimaCore.Utilities.half),
            face_geometry,
        )
    g33 = face_geometry.gⁱʲ.components.data.:9
    u₃_bc = Geometry.Covariant3Vector.(-1 .* u₁_bc ./ g33)
    apply_boundary_w =
        Operators.SetBoundaryOperator(bottom = Operators.SetValue(u₃_bc))
    @. Y.w = apply_boundary_w(Y.w)
    return Y
end

# A `FieldVector` may hold non-`Field` entries, which need no DSS.
_dss!(x::Fields.Field) = Spaces.weighted_dss!(x)
_dss!(::Any) = nothing
function dss!(Y, parameters, t)
    foreach(_dss!, Fields._values(Y))
    project_surface_w!(Y)
end

# The stage values reaching `rhs_invariant!` satisfy the constraint because
# `dss!` runs after every stage; the initial state has to be projected here.
project_surface_w!(Y)
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
    saveat = collect(0.0:500.0:timeend),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

if haskey(ENV, "CI_PERF_SKIP_RUN") # for performance analysis
    throw(:exit_profile)
end

sol = @timev CTS.solve!(integrator)

# Mass and total energy must be conserved (measured drift: 5e-14 and 1.4e-13).
# The 250 m ridge in a 10 m/s flow forces a steady wave with max|w| ≈ 2.0 m/s:
# the lower bound checks the wave forms, the upper that the surface
# condition holds.
@test abs(sum(sol.u[end].Yc.ρ) - sum(sol.u[1].Yc.ρ)) / sum(sol.u[1].Yc.ρ) < 1e-11
@test abs(sum(sol.u[end].Yc.ρe) - sum(sol.u[1].Yc.ρe)) / sum(sol.u[1].Yc.ρe) < 1e-11
@testset "mountain-wave amplitude" begin
    w = maximum(abs, parent(Geometry.WVector.(sol.u[end].w)))
    @info "Peak |w| at the end of the run: $w m/s"
    @test 1 < w < 3
end

ENV["GKSwstype"] = "nul"
import Plots, ClimaCorePlots
Plots.GRBackend()

dir = "schar_etot_nh"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)


anim = Plots.@animate for u in sol.u
    Plots.plot(u.Yc.ρe ./ u.Yc.ρ)
end
Plots.mp4(anim, joinpath(path, "total_energy.mp4"), fps = 20)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.Yc.ρq ./ u.Yc.ρ)
end
Plots.mp4(anim, joinpath(path, "tracer.mp4"), fps = 20)

If2c = Operators.InterpolateF2C()
anim = Plots.@animate for u in sol.u
    ᶜuw = @. Geometry.Covariant13Vector(u.uₕ) + Geometry.Covariant13Vector(If2c(u.w))
    w = @. Geometry.project(Geometry.WAxis(), ᶜuw)
    Plots.plot(w, ylim = (0, 12000), xlim = (-10000, 10000))
end
Plots.mp4(anim, joinpath(path, "vel_w.mp4"), fps = 20)

anim = Plots.@animate for u in sol.u
    ᶜuw = @. Geometry.Covariant13Vector(u.uₕ) + Geometry.Covariant13Vector(If2c(u.w))
    u = @. Geometry.project(Geometry.UAxis(), ᶜuw)
    Plots.plot(u, ylim = (0, 12000), xlim = (-10000, 10000))
end
Plots.mp4(anim, joinpath(path, "vel_u.mp4"), fps = 20)

anim = Plots.@animate for u in sol.u
    ᶜu = @. Geometry.Covariant13Vector(u.uₕ)
    ᶜw = @. Geometry.Covariant13Vector(If2c(u.w))
    w = @. Geometry.project(Geometry.Contravariant1Axis(), ᶜu) +
       Geometry.project(Geometry.Contravariant1Axis(), ᶜw)
    Plots.plot(w, ylim = (0, 12000), xlim = (-10000, 10000))
end
Plots.mp4(anim, joinpath(path, "ucontravariant1.mp4"), fps = 20)

Ic2f = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
anim = Plots.@animate for u in sol.u
    ᶠu = @. Geometry.Covariant13Vector(Ic2f(u.uₕ))
    ᶠw = @. Geometry.Covariant13Vector(u.w)
    w = @. Geometry.project(Geometry.Contravariant3Axis(), ᶠu) +
       Geometry.project(Geometry.Contravariant3Axis(), ᶠw)
    Plots.plot(w, ylim = (0, 500), clims = (-0.001, 0.001))
end
Plots.mp4(anim, joinpath(path, "contravariant3.mp4"), fps = 20)

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
