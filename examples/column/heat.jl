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

using OrdinaryDiffEqSSPRK: ODEProblem, solve, SSPRK33

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())
const CI = !isnothing(get(ENV, "CI", nothing))

const FT = Float64

a = FT(0.0)
b = FT(1.0)
n = 10
α = FT(0.1)

domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(a),
    Geometry.ZPoint{FT}(b),
    boundary_names = (:bottom, :top),
)
mesh = Meshes.IntervalMesh(domain, nelems = n)
device = ClimaComms.device()
cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
T = Fields.zeros(FT, cs)

# Solve Heat Equation: ∂_t T = α ∇²T
function ∑tendencies!(dT, T, _, t)

    bcs_bottom = Operators.SetValue(FT(0.0))
    bcs_top = Operators.SetGradient(Geometry.WVector(FT(1.0)))

    gradc2f = Operators.GradientC2F(bottom = bcs_bottom, top = bcs_top)
    divf2c = Operators.DivergenceF2C()

    return @. dT = α * divf2c(gradc2f(T))
end

@show ∑tendencies!(similar(T), T, nothing, 0.0);

# Solve the ODE operator
Δt = 0.02

prob = ODEProblem(∑tendencies!, T, (0.0, 10.0))
sol = solve(
    prob,
    SSPRK33(),
    dt = Δt,
    saveat = collect(0.0:(10 * Δt):10.0),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = "heat"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u, xlim = (0, 1))
end
Plots.mp4(anim, joinpath(path, "heat.mp4"), fps = 10)
Plots.png(Plots.plot(sol.u[end], xlim = (0, 1)), joinpath(path, "heat_end.png"))

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig(
    relpath(joinpath(path, "heat_end.png"), joinpath(@__DIR__, "../..")),
    "Heat End Simulation",
)
