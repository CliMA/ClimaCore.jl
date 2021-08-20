push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

import ClimaCore.Geometry, LinearAlgebra, UnPack
import ClimaCore:
    Fields,
    Domains,
    Topologies,
    Meshes,
    DataLayouts,
    Operators,
    Geometry,
    Spaces

using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

const FT = Float64

a = FT(0.0)
b = FT(4pi)
n = 128
α = FT(0.1)

domain = Domains.IntervalDomain(a, b, x3boundary = (:left, :right))
mesh = Meshes.IntervalMesh(domain, nelems = n)

cs = Spaces.CenterFiniteDifferenceSpace(mesh)
fs = Spaces.FaceFiniteDifferenceSpace(cs)

V = Geometry.Cartesian3Vector.(ones(FT, fs))
θ = sin.(Fields.coordinate_field(cs))

# Solve advection Equation: ∂θ/dt = -∂(vθ)

# upwinding
function tendency1!(dθ, θ, _, t)
    UB = Operators.UpwindBiasedProductC2F(
        left = Operators.SetValue(sin(a - t)),
        right = Operators.SetValue(sin(b - t)),
    )
    ∂ = Operators.DivergenceF2C()

    return @. dθ = -∂(UB(V, θ))
end

# use the advection operator
function tendency2!(dθ, θ, _, t)
    A = Operators.AdvectionC2C(
        left = Operators.SetValue(sin(-t)),
        right = Operators.Extrapolate(),
    )
    return @. dθ = -A(V, θ)
end

@show tendency2!(similar(θ), θ, nothing, 0.0)

# Solve the ODE operator
Δt = 0.01
prob = ODEProblem(tendency2!, θ, (0.0, 10.0))
sol = solve(
    prob,
    SSPRK33(),
    dt = Δt,
    saveat = 10 * Δt,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dirname = "advect"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u, xlim = (-1, 1))
end
Plots.mp4(anim, joinpath(path, "advect.mp4"), fps = 10)
Plots.png(
    Plots.plot(sol.u[end], xlim = (-1, 1)),
    joinpath(path, "advect_end.png"),
)

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig("output/$(dirname)/advect_end.png", "Advect End Simulation")
