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

using OrdinaryDiffEq: OrdinaryDiffEq, ODEProblem, solve, SSPRK33

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

const FT = Float64

domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(0.0),
    Geometry.ZPoint{FT}(4pi),
    boundary_tags = (:left, :right),
)
mesh = Meshes.IntervalMesh(domain; nelems = 30)

cspace = Spaces.CenterFiniteDifferenceSpace(mesh)
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
    saveat = 10 * Δt,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);


ENV["GKSwstype"] = "nul"
import Plots, ClimaCorePlots
Plots.GRBackend()

dirname = "wave"
path = joinpath(@__DIR__, "output", dirname)
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
