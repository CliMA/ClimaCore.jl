# 1D vertical advection of a sine wave, solved four ways: the first-order upwind
# flux form (`UpwindBiasedProductC2F`), the centered advective form
# (`AdvectionC2C`), and each of those with `FluxCorrectionC2C` — an upwind-style
# diffusive stabilizer — added. The exact solution is sin(z - t), which supplies
# the boundary values, so the errors measure what each formulation does to the
# wave: the centered form is nearly exact but relies on the problem's
# smoothness, the upwind form damps, and the flux correction adds upwind damping
# to whatever it is applied to.
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

const z_left = FT(0)
const z_right = FT(4pi)
const nelems = 128

domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(z_left),
    Geometry.ZPoint{FT}(z_right),
    boundary_names = (:left, :right),
)
mesh = Meshes.IntervalMesh(domain, nelems = nelems)
device = ClimaComms.device()
cspace = Spaces.CenterFiniteDifferenceSpace(device, mesh)
fspace = Spaces.FaceFiniteDifferenceSpace(cspace)

# Unit upward velocity, so the exact solution of ∂θ/∂t = -∂(wθ)/∂z is the
# initial sine wave translated at unit speed. It also supplies the inflow
# boundary values.
w = Geometry.WVector.(ones(FT, fspace))
exact_θ(z, t) = sin(z - t)
θ_init = exact_θ.(Fields.coordinate_field(cspace).z, 0)

# `FluxCorrectionC2C` adds an upwind-style diffusive flux to whichever
# formulation it is applied to; it carries no boundary values of its own.
flux_correction() = Operators.FluxCorrectionC2C(
    left = Operators.Extrapolate(),
    right = Operators.Extrapolate(),
)

# Flux form: the face flux is reconstructed by first-order upwinding, and the
# inflow face takes the exact value.
upwind_flux(t) = Operators.UpwindBiasedProductC2F(
    left = Operators.SetValue(exact_θ(z_left, t)),
    right = Operators.SetValue(exact_θ(z_right, t)),
)

# Advective form: centered differences, with the exact value at the inflow
# boundary and extrapolation at the outflow one.
centered_advection(t) = Operators.AdvectionC2C(
    left = Operators.SetValue(exact_θ(z_left, t)),
    right = Operators.Extrapolate(),
)

function tendency_upwind!(dθ, θ, _, t)
    upwind = upwind_flux(t)
    divf2c = Operators.DivergenceF2C()
    return @. dθ = -divf2c(upwind(w, θ))
end
function tendency_upwind_corrected!(dθ, θ, _, t)
    upwind = upwind_flux(t)
    divf2c = Operators.DivergenceF2C()
    correction = flux_correction()
    return @. dθ = -divf2c(upwind(w, θ)) + correction(w, θ)
end
function tendency_centered!(dθ, θ, _, t)
    advect = centered_advection(t)
    return @. dθ = -advect(w, θ)
end
function tendency_centered_corrected!(dθ, θ, _, t)
    advect = centered_advection(t)
    correction = flux_correction()
    return @. dθ = -advect(w, θ) + correction(w, θ)
end

tendency_upwind!(similar(θ_init), θ_init, nothing, 0.0)

# Solve the ODE operator
Δt = 0.001
t_end = 10.0
tendencies = (;
    upwind = tendency_upwind!,
    upwind_corrected = tendency_upwind_corrected!,
    centered = tendency_centered!,
    centered_corrected = tendency_centered_corrected!,
)
solutions = map(tendencies) do tendency!
    prob = CTS.ODEProblem(
        CTS.ClimaODEFunction(; T_exp! = tendency!),
        copy(θ_init),
        (0.0, t_end),
        nothing,
    )
    CTS.solve(
        prob,
        CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()),
        dt = Δt,
        saveat = collect(0.0:(10 * Δt):t_end),
        progress = true,
        progress_message = (dt, u, p, t) -> t,
    )
end

using Test
@testset "advected profiles vs the analytic solution" begin
    z = vec(parent(Fields.coordinate_field(cspace).z))
    exact = exact_θ.(z, t_end)
    final(sol) = vec(parent(sol.u[end]))
    Linf(sol) = maximum(abs, final(sol) .- exact)
    # Root-mean-square amplitude: the wave carries variance, and only the
    # dissipative formulations may lose it.
    rms(values) = sqrt(sum(abs2, values) / length(values))
    # Measured errors: upwind 0.39, upwind+correction 0.77, centered 0.018,
    # centered+correction 0.62. The centered scheme's bound is what pins the
    # physics: at this resolution its phase error is ~2e-2, so anything larger
    # means the discretization (not the formulation) has broken.
    @test Linf(solutions.upwind) < 0.5
    @test Linf(solutions.upwind_corrected) < 1.0
    @test Linf(solutions.centered) < 0.05
    @test Linf(solutions.centered_corrected) < 1.0
    # None of the schemes may create new extrema (exact solution is in [-1, 1]).
    for sol in solutions
        @test maximum(abs, final(sol)) < 1.05
    end
    # The centered form is non-dissipative: it carries the wave at full
    # amplitude (measured rms 0.7076 against the exact 0.7071). Upwinding and
    # the flux correction are both diffusive, so every other formulation loses
    # variance, and applying the correction on top of either form loses more
    # (measured rms: upwind 0.53, upwind+correction 0.35, centered+correction
    # 0.42). This ordering is the point of the example.
    @test rms(final(solutions.centered)) ≈ rms(exact) rtol = 0.01
    @test rms(final(solutions.upwind)) < 0.9 * rms(exact)
    @test rms(final(solutions.upwind_corrected)) < rms(final(solutions.upwind))
    @test rms(final(solutions.centered_corrected)) <
          rms(final(solutions.centered))
end

ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = "advect"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

for (name, sol) in pairs(solutions)
    anim = Plots.@animate for θ in sol.u
        Plots.plot(θ, xlim = (-1, 1))
    end
    Plots.mp4(anim, joinpath(path, "advect_$name.mp4"), fps = 10)
    Plots.png(
        Plots.plot(sol.u[end], xlim = (-1, 1)),
        joinpath(path, "advect_$(name)_end.png"),
    )
end

z_centers = vec(parent(Fields.coordinate_field(cspace).z))
line_styles = (:dash, :dot, :solid, :dashdot)
comparison = Plots.plot(
    exact_θ.(z_centers, t_end),
    z_centers,
    xlim = (-1.1, 1.1),
    color = :black,
    xlabel = "θ",
    ylabel = "z",
    label = "Exact",
)
for ((name, sol), style) in zip(pairs(solutions), line_styles)
    Plots.plot!(
        comparison,
        vec(parent(sol.u[end])),
        z_centers,
        ls = style,
        label = "$name",
    )
end
Plots.png(comparison, joinpath(path, "advect_all_end.png"))

include(joinpath(@__DIR__, "..", "example_utils.jl")) # linkfig

linkfig(
    relpath(joinpath(path, "advect_all_end.png"), joinpath(@__DIR__, "../..")),
    "Advect End Simulation",
)
