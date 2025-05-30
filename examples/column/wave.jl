import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore.Geometry, LinearAlgebra
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
const FT = Float64

domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(0.0),
    Geometry.ZPoint{FT}(4pi),
    boundary_names = (:left, :right),
)
mesh = Meshes.IntervalMesh(domain; nelems = 30)
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

@show tendency!(similar(Y), Y, nothing, 0.0)

# Solve the ODE operator
Δt = 0.01
prob = ODEProblem(tendency!, Y, (0.0, 4 * pi))
sol = solve(
    prob,
    SSPRK33(),
    dt = Δt,
    saveat = [0.0:(10 * Δt):(4 * pi)..., 4 * pi],
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);


ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = "wave"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.u, xlim = (-1, 1), label = "u", legend = true)
end
Plots.mp4(anim, joinpath(path, "wave.mp4"), fps = 10)

Plots.png(
    Plots.plot(sol.u[end].u, label = "u", legend = true),
    joinpath(path, "wave_end.png"),
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
    relpath(joinpath(path, "wave_end.png"), joinpath(@__DIR__, "../..")),
    "Wave End",
)
