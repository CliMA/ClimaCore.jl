using Test
using LinearAlgebra

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
import ClimaCore.Utilities: half

using OrdinaryDiffEq: ODEProblem, solve
using DiffEqBase
using ClimaTimeSteppers

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

# 3D deformation flow (DCMIP 2012 Test 1-1)
# Reference: http://www-personal.umich.edu/~cjablono/DCMIP-2012_TestCaseDocument_v1.7.pdf, Section 1.1

const R = 6.37122e6 # radius
const grav = 9.8 # gravitational constant
const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const z_top = 1.2e4 # height position of the model top
const p_top = 25494.4 # pressure at the model top
const T_0 = 300 # isothermal atmospheric temperature
const H = R_d * T_0 / grav # scale height
const p_0 = 1.0e5 # reference pressure
const τ = 1036800.0 # period of motion
const ω_0 = 23000 * pi / τ # maxium of the vertical pressure velocity
const b = 0.2 # normalized pressure depth of divergent layer
const λ_c1 = 150.0 # initial longitude of first tracer
const λ_c2 = 210.0 # initial longitude of second tracer
const ϕ_c = 0.0 # initial latitude of tracers
const z_c = 5.0e3 # initial altitude of tracers
const R_t = R / 2 # horizontal half-width of tracers
const Z_t = 1000.0 # vertical half-width of tracers
const κ₄ = 1.0e16 # hyperviscosity

# time constants
T = 86400.0 * 12.0
dt = 60.0 * 60.0

# set up function space
function sphere_3D(
    R = 6.37122e6,
    zlim = (0, 12.0e3),
    helem = 4,
    zelem = 36,
    npoly = 4,
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

# set up 3D domain
hv_center_space, hv_face_space = sphere_3D()

# initial conditions
coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

r1(λ, ϕ) = R * acos(sind(ϕ_c) * sind(ϕ) + cosd(ϕ_c) * cosd(ϕ) * cosd(λ - λ_c1))
r2(λ, ϕ) = R * acos(sind(ϕ_c) * sind(ϕ) + cosd(ϕ_c) * cosd(ϕ) * cosd(λ - λ_c2))

p(z) = p_0 * exp(-z / H)
ρ_ref(z) = p(z) / R_d / T_0

y0 = map(coords) do coord
    z = coord.z
    zd = z - z_c
    λ = coord.long
    ϕ = coord.lat
    rd1 = r1(λ, ϕ)
    rd2 = r2(λ, ϕ)

    d1 = min(1, (rd1 / R_t)^2 + (zd / Z_t)^2)
    d2 = min(1, (rd2 / R_t)^2 + (zd / Z_t)^2)

    q1 = 1 / 2.0 * (1 + cos(pi * d1)) + 1 / 2.0 * (1 + cos(pi * d2))
    q2 = 0.9 - 0.8 * q1^2
    q3 = 0.0
    if d1 < 0.5 || d2 < 0.5
        q3 = 1.0
    else
        q3 = 0.1
    end

    if (z > z_c) && (ϕ > ϕ_c - rad2deg(1 / 8)) && (ϕ < ϕ_c + rad2deg(1 / 8))
        q3 = 0.1
    end
    q4 = 1 - 3 / 10 * (q1 + q2 + q3)

    ρq1 = ρ_ref(z) * q1
    ρq2 = ρ_ref(z) * q2
    ρq3 = ρ_ref(z) * q3
    ρq4 = ρ_ref(z) * q4

    return (ρ = ρ_ref(z), ρq1 = ρq1, ρq2 = ρq2, ρq3 = ρq3, ρq4 = ρq4)
end

y0 = Fields.FieldVector(
    ρ = y0.ρ,
    ρq1 = y0.ρq1,
    ρq2 = y0.ρq2,
    ρq3 = y0.ρq3,
    ρq4 = y0.ρq4,
)

function rhs!(dydt, y, parameters, t, alpha, beta)

    τ = parameters.τ
    coords = parameters.coords
    face_coords = parameters.face_coords

    ϕ = coords.lat
    λ = coords.long
    zc = coords.z
    zf = face_coords.z
    λp = λ .- 360 * t / τ
    k = 10 * R / τ

    ϕf = face_coords.lat
    λf = face_coords.long
    λpf = λf .- 360 * t / τ

    sp =
        @. 1 + exp((p_top - p_0) / b / p_top) - exp((p(zf) - p_0) / b / p_top) -
           exp((p_top - p(zf)) / b / p_top)
    ua = @. k * sind(λp)^2 * sind(2 * ϕ) * cos(pi * t / τ) +
       2 * pi * R / τ * cosd(ϕ)
    ud = @. ω_0 * R / b / p_top *
       cosd(λp) *
       cosd(ϕ)^2 *
       cos(2 * pi * t / τ) *
       (-exp((p(zc) - p_0) / b / p_top) + exp((p_top - p(zc)) / b / p_top))
    uu = @. ua + ud
    uv = @. k * sind(2 * λp) * cosd(ϕ) * cos(pi * t / τ)
    ω = @. ω_0 * sind(λpf) * cosd(ϕf) * cos(2 * pi * t / τ) * sp
    uw = @. -ω / ρ_ref(zf) / grav

    uₕ = Geometry.Covariant12Vector.(Geometry.UVVector.(uu, uv))
    w = Geometry.Covariant3Vector.(Geometry.WVector.(uw))

    ρ = y.ρ
    ρq1 = y.ρq1
    ρq2 = y.ρq2
    ρq3 = y.ρq3
    ρq4 = y.ρq4

    dρ = dydt.ρ
    dρq1 = dydt.ρq1
    dρq2 = dydt.ρq2
    dρq3 = dydt.ρq3
    dρq4 = dydt.ρq4

    # No change in density: divergence-free flow
    @. dydt.ρ = beta * dydt.ρ + 0 .* y.ρ

    If2c = Operators.InterpolateF2C()
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    vdivf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    )
    third_order_upwind_c2f = Operators.Upwind3rdOrderBiasedProductC2F(
        bottom = Operators.ThirdOrderOneSided(),
        top = Operators.ThirdOrderOneSided(),
    )
    hdiv = Operators.Divergence()
    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()

    ### HYPERVISCOSITY
    ystar = similar(y)
    @. ystar.ρq1 = hwdiv(hgrad(ρq1 / ρ))
    Spaces.weighted_dss!(ystar.ρq1)
    @. ystar.ρq1 = -κ₄ * hwdiv(ρ * hgrad(ystar.ρq1))

    @. ystar.ρq2 = hwdiv(hgrad(ρq2 / ρ))
    Spaces.weighted_dss!(ystar.ρq2)
    @. ystar.ρq2 = -κ₄ * hwdiv(ρ * hgrad(ystar.ρq2))

    @. ystar.ρq3 = hwdiv(hgrad(ρq3 / ρ))
    Spaces.weighted_dss!(ystar.ρq3)
    @. ystar.ρq3 = -κ₄ * hwdiv(ρ * hgrad(ystar.ρq3))

    @. ystar.ρq4 = hwdiv(hgrad(ρq4 / ρ))
    Spaces.weighted_dss!(ystar.ρq4)
    @. ystar.ρq4 = -κ₄ * hwdiv(ρ * hgrad(ystar.ρq4))

    cw = If2c.(w)
    cuvw = Geometry.Covariant123Vector.(uₕ) .+ Geometry.Covariant123Vector.(cw)

    @. dρq1 = beta * dρq1 - alpha * hdiv(cuvw * ρq1) + alpha * ystar.ρq1
    @. dρq1 -= alpha * vdivf2c(Ic2f(ρ) * third_order_upwind_c2f.(w, ρq1 ./ ρ))
    @. dρq1 -= alpha * vdivf2c(Ic2f(uₕ * ρq1))

    @. dρq2 = beta * dρq2 - alpha * hdiv(cuvw * ρq2) + alpha * ystar.ρq2
    @. dρq2 -= alpha * vdivf2c(Ic2f(ρ) * third_order_upwind_c2f.(w, ρq2 ./ ρ))
    @. dρq2 -= alpha * vdivf2c(Ic2f(uₕ * ρq2))

    @. dρq3 = beta * dρq3 - alpha * hdiv(cuvw * ρq3) + alpha * ystar.ρq3
    @. dρq3 -= alpha * vdivf2c(Ic2f(ρ) * third_order_upwind_c2f.(w, ρq3 ./ ρ))
    @. dρq3 -= alpha * vdivf2c(Ic2f(uₕ * ρq3))

    @. dρq4 = beta * dρq4 - alpha * hdiv(cuvw * ρq4) + alpha * ystar.ρq4
    @. dρq4 -= alpha * vdivf2c(Ic2f(ρ) * third_order_upwind_c2f.(w, ρq4 ./ ρ))
    @. dρq4 -= alpha * vdivf2c(Ic2f(uₕ * ρq4))

    Spaces.weighted_dss!(dydt.ρ)
    Spaces.weighted_dss!(dydt.ρq1)
    Spaces.weighted_dss!(dydt.ρq2)
    Spaces.weighted_dss!(dydt.ρq3)
    Spaces.weighted_dss!(dydt.ρq4)

    return dydt
end

# Set up RHS function
ystar = copy(y0)
parameters = (τ = τ, coords = coords, face_coords = face_coords)

rhs!(ystar, y0, parameters, 0.0, dt, 1)

# run!
prob = ODEProblem(IncrementingODEFunction(rhs!), copy(y0), (0.0, T), parameters)
sol = solve(
    prob,
    SSPRK33ShuOsher(),
    dt = dt,
    saveat = dt,
    progress = true,
    adaptive = false,
    progress_message = (dt, u, p, t) -> t,
)

q1_error =
    norm(sol.u[end].ρq1 ./ ρ_ref.(coords.z) .- y0.ρq1 ./ ρ_ref.(coords.z)) /
    norm(y0.ρq1 ./ ρ_ref.(coords.z))
@test q1_error ≈ 0.0 atol = 0.7

q2_error =
    norm(sol.u[end].ρq2 ./ ρ_ref.(coords.z) .- y0.ρq2 ./ ρ_ref.(coords.z)) /
    norm(y0.ρq2 ./ ρ_ref.(coords.z))
@test q2_error ≈ 0.0 atol = 0.03

q3_error =
    norm(sol.u[end].ρq3 ./ ρ_ref.(coords.z) .- y0.ρq3 ./ ρ_ref.(coords.z)) /
    norm(y0.ρq3 ./ ρ_ref.(coords.z))
@test q3_error ≈ 0.0 atol = 0.4

q4_error =
    norm(sol.u[end].ρq4 ./ ρ_ref.(coords.z) .- y0.ρq4 ./ ρ_ref.(coords.z)) /
    norm(y0.ρq4 ./ ρ_ref.(coords.z))
@test q4_error ≈ 0.0 atol = 0.03

# visualization artifacts
ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()
dir = "deformation_flow"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

Plots.png(
    Plots.plot(
        sol.u[trunc(Int, end / 2)].ρq3 ./ ρ_ref.(coords.z),
        level = 15,
        clim = (-1, 1),
    ),
    joinpath(path, "q3_6day.png"),
)

Plots.png(
    Plots.plot(sol.u[end].ρq3 ./ ρ_ref.(coords.z), level = 15, clim = (-1, 1)),
    joinpath(path, "q3_12day.png"),
)
