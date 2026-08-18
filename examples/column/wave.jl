# 1D wave equation written as a first-order system in a center-valued `u` and a
# face-valued `p`, so the two unknowns sit on the staggered grid that
# `GradientC2F` and `DivergenceF2C` pair across. Homogeneous Dirichlet
# conditions on `u` hold the ends fixed; the run animates the standing wave.
#
# From u = sin(z) at rest the exact solution is the standing wave
#
#     u = sin(z) cos(t),   p = -cos(z) sin(t),
#
# which the run asserts against. Because the two operators are adjoint on this
# grid, the semi-discrete system is skew-symmetric and conserves a discrete
# energy exactly; that is asserted too.
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore:
    Fields,
    Domains,
    Topologies,
    Meshes,
    DataLayouts,
    Operators,
    Geometry,
    Spaces

import ClimaTimeSteppers as CTS

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())
const FT = Float64

const z_length = FT(4pi)   # two wavelengths of the sin(z) initial condition
const nelems = 30
const t_end = FT(4pi)      # two periods, so the wave returns to where it began

domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(0.0),
    Geometry.ZPoint{FT}(z_length),
    boundary_names = (:left, :right),
)
mesh = Meshes.IntervalMesh(domain; nelems = nelems)
device = ClimaComms.device()
cspace = Spaces.CenterFiniteDifferenceSpace(device, mesh)
fspace = Spaces.FaceFiniteDifferenceSpace(cspace)

zc = Fields.coordinate_field(cspace)
u = sin.(zc.z)
p = Geometry.WVector.(zeros(Float64, fspace))

Y = Fields.FieldVector(u = u, p = p)

function tendency!(dY, Y, _, t)
    u = Y.u
    p = Y.p

    du = dY.u
    dp = dY.p

    ∂f = Operators.GradientC2F(
        left = Operators.SetValue(0.0),
        right = Operators.SetValue(0.0),
    )
    ∂c = Operators.DivergenceF2C()

    @. dp = -Geometry.WVector(∂f(u))
    @. du = -∂c(p)

    return dY
end

tendency!(similar(Y), Y, nothing, 0.0)

# Solve the ODE operator
Δt = 0.01
prob =
    CTS.ODEProblem(CTS.ClimaODEFunction(; T_exp! = tendency!), Y, (0.0, t_end), nothing)
sol = CTS.solve(
    prob,
    CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()),
    dt = Δt,
    saveat = [0.0:(10 * Δt):t_end..., t_end],
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

using Test
using LinearAlgebra: norm

z_centers = vec(parent(Fields.coordinate_field(cspace).z))
z_faces = vec(parent(Fields.coordinate_field(fspace).z))
exact_u(z, t) = sin(z) * cos(t)
exact_p(z, t) = -cos(z) * sin(t)

# The scheme conserves the discrete energy (Δz/2) (Σ u² + Σ w p²) exactly, with
# the boundary faces carrying half weight — that weighting is what makes the
# boundary terms in `GradientC2F(SetValue)` and `DivergenceF2C` cancel.
face_weights = [j == 1 || j == nelems + 1 ? 0.5 : 1.0 for j in 1:(nelems + 1)]
Δz = z_length / nelems
energy(Y) =
    Δz / 2 * (
        sum(abs2, vec(parent(Y.u))) +
        sum(face_weights .* vec(parent(Y.p)) .^ 2)
    )

@testset "wave energy" begin
    energies = energy.(sol.u)
    # Only the SSP33 time integrator can change it, and only by damping
    # (measured drift over the run: -1e-6, monotone).
    @test energies[end] ≤ energies[1]
    @test abs(energies[end] - energies[1]) / energies[1] < 1e-5
end

@testset "computed wave vs the exact standing wave" begin
    # The computed wave tracks the exact standing wave over the whole run
    # (measured worst case: 0.080 in u, 0.092 in p). The error is dispersion:
    # the discrete frequency is 2 sin(kΔz/2) / Δz = 0.9925 rather than 1, so
    # after two periods the wave lags by 0.095 rad. A wave that failed to
    # oscillate at all would miss the exact solution by 1.
    u_errors = map(zip(sol.t, sol.u)) do (t, Y)
        maximum(abs, vec(parent(Y.u)) .- exact_u.(z_centers, t))
    end
    p_errors = map(zip(sol.t, sol.u)) do (t, Y)
        maximum(abs, vec(parent(Y.p)) .- exact_p.(z_faces, t))
    end
    @test maximum(u_errors) < 0.15
    @test maximum(p_errors) < 0.15
    # Two periods return the wave to its initial state (measured: 4e-3).
    @test norm(parent(sol.u[end].u) .- parent(sol.u[1].u)) /
          norm(parent(sol.u[1].u)) < 2e-2
end

ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = "wave"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

# Plot the computed wave against the exact standing wave at the same time.
function wave_plot(t, Y; kwargs...)
    plt = Plots.plot(
        exact_u.(z_centers, t),
        z_centers,
        marker = :circle,
        xlim = (-1.1, 1.1),
        xlabel = "u",
        ylabel = "z",
        label = "Exact";
        kwargs...,
    )
    return Plots.plot!(plt, vec(parent(Y.u)), z_centers, label = "Computed")
end

anim = Plots.@animate for (t, Y) in zip(sol.t, sol.u)
    wave_plot(t, Y, title = "t = $(round(t, digits = 2))")
end
Plots.mp4(anim, joinpath(path, "wave.mp4"), fps = 10)

Plots.png(wave_plot(sol.t[end], sol.u[end]), joinpath(path, "wave_end.png"))

include(joinpath(@__DIR__, "..", "example_utils.jl")) # linkfig

linkfig(
    relpath(joinpath(path, "wave_end.png"), joinpath(@__DIR__, "../..")),
    "Wave End",
)
