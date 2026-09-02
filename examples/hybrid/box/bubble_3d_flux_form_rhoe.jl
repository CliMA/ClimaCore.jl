# Dry rising bubble in a doubly-periodic 3D box, discretized in *flux form*
# with total energy (ρe) as the prognostic thermodynamic variable.
#
# This is the flux-form counterpart of `bubble_3d_invariant_rhoe.jl`: both
# solve the same problem with the same prognostic variables, but this one
# carries momentum as ρuₕ (cell centers) and ρw (cell faces) and advances it
# with divergence-form tendencies, whereas the invariant version uses the
# vector-invariant form. Running both keeps the two momentum formulations
# exercised against a case with a known qualitative answer: a warm bubble that
# rises and deforms while total mass and energy are conserved.
#
# Reference: Section 5a of
# https://journals.ametsoc.org/view/journals/mwre/140/4/mwr-d-10-05073.1.xml

using LinearAlgebra, StaticArrays

import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore:
    ClimaCore,
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Quadratures,
    Spaces,
    Topologies
import ClimaCore.Geometry: ⊗

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

const context = ClimaComms.SingletonCommsContext()

# ============================================================================
# Physical parameters
# ============================================================================

include(joinpath(@__DIR__, "../..", "example_utils.jl")) # linkfig
include(joinpath(@__DIR__, "bubble_parameters.jl")) # PhysicalParameters, geopotential
include(joinpath(@__DIR__, "..", "flux_correction_utils.jl")) # add_flux_correction_{c2c,f2f}

# The invariant counterpart threads these through a `PhysicalParameters` value
# because it also runs on the GPU; this case is CPU-only, so it reads the same
# defaults into constants and keeps the tendencies free of a parameter
# argument.
const params = PhysicalParameters{Float64}()
const MSLP = params.MSLP
const grav = params.grav
const R_d = params.R_d
const γ = params.γ
const C_p = params.C_p
const C_v = params.C_v
const T_0 = params.T_0

# ============================================================================
# Spaces
# ============================================================================

function hvspace_3D(
    xlim = (-500, 500),
    ylim = (-500, 500),
    zlim = (0, 1000);
    xelem = 4,
    yelem = 4,
    zelem = 16,
    npoly = 3, # matches the invariant case, so the two are comparable
)
    FT = Float64
    device = ClimaComms.device(context)

    xdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1]),
        Geometry.XPoint{FT}(xlim[2]),
        periodic = true,
    )
    ydomain = Domains.IntervalDomain(
        Geometry.YPoint{FT}(ylim[1]),
        Geometry.YPoint{FT}(ylim[2]),
        periodic = true,
    )
    horzdomain = Domains.RectangleDomain(xdomain, ydomain)
    horzmesh = Meshes.RectilinearMesh(horzdomain, xelem, yelem)
    horztopology = Topologies.Topology2D(context, horzmesh)
    horzspace =
        Spaces.SpectralElementSpace2D(horztopology, Quadratures.GLL{npoly + 1}())

    zdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(zdomain, nelems = zelem)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(device, vertmesh)

    center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)
    return (center_space, face_space)
end

center_space, face_space = hvspace_3D()

# ============================================================================
# Initial condition
# ============================================================================

"""
    init_dry_rising_bubble_3d(x, y, z)

Warm bubble of radius `r_c` centered at `(0, 0, z_c)` in a neutrally stratified
atmosphere, returning the prognostic state: density, total energy density, and
horizontal momentum (zero at rest).
"""
function init_dry_rising_bubble_3d(x, y, z)
    # Bubble geometry and amplitude
    x_c = 0.0
    y_c = 0.0
    z_c = 350.0
    r_c = 250.0
    θ_b = 300.0   # background potential temperature (K)
    θ_c = 0.5     # bubble potential temperature perturbation (K)

    r = sqrt((x - x_c)^2 + (y - y_c)^2 + (z - z_c)^2)
    θ_p = r < r_c ? 0.5 * θ_c * (1.0 + cospi(r / r_c)) : 0.0

    θ = θ_b + θ_p
    π_exn = 1.0 - grav * z / C_p / θ    # Exner function
    T = π_exn * θ                       # temperature
    p = MSLP * π_exn^(C_p / R_d)        # pressure
    ρ = p / R_d / T                     # density
    e = C_v * (T - T_0) + geopotential(z, grav)   # total energy per unit mass
    ρe = ρ * e

    return (ρ = ρ, ρe = ρe, ρuₕ = ρ * Geometry.UVVector(0.0, 0.0))
end

coords = Fields.coordinate_field(center_space)
face_coords = Fields.coordinate_field(face_space)

Yc = map(coord -> init_dry_rising_bubble_3d(coord.x, coord.y, coord.z), coords)
ρw = map(_ -> Geometry.WVector(0.0), face_coords)
Y = Fields.FieldVector(Yc = Yc, ρw = ρw)

# ============================================================================
# Diagnostics
# ============================================================================

function combine_momentum(ρuₕ, ρw)
    Geometry.transform(Geometry.UVWAxis(), ρuₕ) +
    Geometry.transform(Geometry.UVWAxis(), ρw)
end

function center_momentum(Y)
    If2c = Operators.InterpolateF2C()
    combine_momentum.(Y.Yc.ρuₕ, If2c.(Y.ρw))
end

"""
    pressure(ρ, ρe, ρu, z)

Pressure diagnosed from the prognostic state, by subtracting the kinetic and
potential contributions from the total energy to obtain the internal energy.
"""
function pressure(ρ, ρe, ρu, z)
    u = ρu / ρ
    internal_energy = ρe / ρ - norm(u)^2 / 2 - geopotential(z, grav)
    T = internal_energy / C_v + T_0
    return ρ * R_d * T
end

total_energy(Y) = sum(Y.Yc.ρe)

energy_0 = total_energy(Y)
mass_0 = sum(Yc.ρ)   # ∫ρ dΩ, with quadrature weights accounted for

# ============================================================================
# Tendency
# ============================================================================

function rhs!(dY, Y, _, t)
    Yc = Y.Yc
    ρw = Y.ρw
    dYc = dY.Yc
    dρw = dY.ρw
    ρ = Yc.ρ
    ρuₕ = Yc.ρuₕ
    ρe = Yc.ρe
    dρ = dYc.ρ
    dρuₕ = dYc.ρuₕ
    dρe = dYc.ρe

    # Spectral horizontal operators. The weak divergence is the adjoint of the
    # gradient, so flux divergences built from it conserve the transported
    # quantity to roundoff on this periodic domain.
    hwdiv = Operators.Divergence{Operators.WeakForm}()
    hgrad = Operators.Gradient()

    # Vertical finite difference operators, with no flux through the
    # rigid top and bottom boundaries
    vdivf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(0.0)),
        top = Operators.SetValue(Geometry.WVector(0.0)),
    )
    vdivc2f = Operators.DivergenceC2F(
        bottom = Operators.SetDivergence(Geometry.WVector(0.0)),
        top = Operators.SetDivergence(Geometry.WVector(0.0)),
    )
    uvdivf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(
            Geometry.WVector(0.0) ⊗ Geometry.UVVector(0.0, 0.0),
        ),
        top = Operators.SetValue(
            Geometry.WVector(0.0) ⊗ Geometry.UVVector(0.0, 0.0),
        ),
    )
    If = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    Ic = Operators.InterpolateF2C()
    ∂f = Operators.GradientC2F()
    B = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(Geometry.WVector(0.0)),
        top = Operators.SetValue(Geometry.WVector(0.0)),
    )

    z = coords.z
    uₕ = @. ρuₕ / ρ
    w = @. ρw / If(ρ)
    wc = @. Ic(ρw) / ρ
    ρf = @. If(ρ)
    p = @. pressure(ρ, ρe, combine_momentum(ρuₕ, Ic(ρw)), z)

    # Hyperviscosity: a ∇⁴ operator applied as two successive ∇² passes, with
    # a DSS between them so the intermediate field is continuous.
    e = @. ρe / ρ
    @. dρe = hwdiv(hgrad(e))
    @. dρuₕ = hwdiv(hgrad(uₕ))
    @. dρw = hwdiv(hgrad(w))
    Spaces.weighted_dss!(dYc)
    Spaces.weighted_dss!(dρw)

    κ₄ = 100.0   # hyperviscosity coefficient (m⁴/s)
    @. dρe = -κ₄ * hwdiv(ρ * hgrad(dρe))
    @. dρuₕ = -κ₄ * hwdiv(ρ * hgrad(dρuₕ))
    @. dρw = -κ₄ * hwdiv(ρf * hgrad(dρw))

    # Continuity
    @. dρ = -vdivf2c(ρw)
    @. dρ -= hwdiv(ρuₕ)

    # Total energy: advected by the flow and worked on by pressure, so the
    # transported quantity is the enthalpy density ρe + p
    @. dρe += -vdivf2c(ρw * If((ρe + p) / ρ))
    @. dρe -= hwdiv(uₕ * (ρe + p))

    # Horizontal momentum
    @. dρuₕ += -uvdivf2c(ρw ⊗ If(uₕ))
    Ih = Ref(Geometry.Tensor(LinearAlgebra.I, (Geometry.UVAxis(), Geometry.UVAxis())))
    @. dρuₕ -= hwdiv(ρuₕ ⊗ uₕ + p * Ih)

    # Vertical momentum: pressure gradient and buoyancy, plus advection
    @. dρw += B(
        Geometry.transform(
            Geometry.WAxis(),
            -(∂f(p)) - If(ρ) * ∂f(geopotential(z, grav)),
        ) - vdivc2f(Ic(ρw ⊗ w)),
    )
    uₕf = @. If(ρuₕ / ρ)
    @. dρw -= hwdiv(uₕf ⊗ ρw)

    # Upwind flux correction
    add_flux_correction_c2c!(dρ, w, ρ)
    add_flux_correction_c2c!(dρe, w, ρe)
    add_flux_correction_c2c!(dρuₕ, w, ρuₕ)
    add_flux_correction_f2f!(dρw, wc, ρw)

    Spaces.weighted_dss!(dYc)
    Spaces.weighted_dss!(dρw)
    return dY
end

# ============================================================================
# Solve
# ============================================================================

dYdt = similar(Y)
rhs!(dYdt, Y, nothing, 0.0)

import ClimaTimeSteppers as CTS
Δt = 0.05
# 1 s of integration time, just to verify the code does not crash
prob = CTS.ODEProblem(CTS.ClimaODEFunction(; T_exp! = rhs!), Y, (0.0, 1.0), nothing)
integrator = CTS.init(
    prob,
    CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()),
    dt = Δt,
    saveat = [0.0, 1.0],
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)

if haskey(ENV, "CI_PERF_SKIP_RUN")   # for performance analysis
    throw(:exit_profile)
end

sol = @timev CTS.solve!(integrator)

# ============================================================================
# Post-processing
# ============================================================================

ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

path = joinpath(@__DIR__, "output", "bubble_3d_flux_form_rhoe")
mkpath(path)

energies = [total_energy(u) for u in sol.u]
masses = [sum(u.Yc.ρ) for u in sol.u]

using Test
# The domain is closed: mass and total energy must be conserved.
@test abs(masses[end] - mass_0) / mass_0 < 1e-12
@test abs(energies[end] - energy_0) / energy_0 < 1e-8
# At t = 1 s the bubble can only just have started to move, and how fast is
# set by its buoyancy: a θ_c = 0.5 K perturbation on a θ_b = 300 K background
# accelerates it at g θ_c / θ_b = 0.016 m/s², so after one second ρw cannot
# exceed ρ × 0.016 ≈ 0.018, and reaches a good fraction of it once the
# compensating pressure gradient is subtracted (measured: 0.012). The return
# flow around the bubble is an order of magnitude weaker (measured: -0.0011).
@testset "updraft and return-flow brackets" begin
    ρw = vec(parent(sol.u[end].ρw))
    @test 0.005 < maximum(ρw) < 0.018
    @test -0.005 < minimum(ρw) < 0
end

Plots.png(
    Plots.plot((energies .- energy_0) ./ energy_0),
    joinpath(path, "energy.png"),
)
Plots.png(Plots.plot((masses .- mass_0) ./ mass_0), joinpath(path, "mass.png"))


linkfig(
    relpath(joinpath(path, "energy.png"), joinpath(@__DIR__, "../..")),
    "Total Energy",
)
linkfig(
    relpath(joinpath(path, "mass.png"), joinpath(@__DIR__, "../..")),
    "Mass",
)
