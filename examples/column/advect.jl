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
const FT = Float64

a = FT(0.0)
b = FT(4pi)
n = 128
α = FT(0.1)

domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(a),
    Geometry.ZPoint{FT}(b),
    boundary_names = (:left, :right),
)
mesh = Meshes.IntervalMesh(domain, nelems = n)
device = ClimaComms.device()
cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
fs = Spaces.FaceFiniteDifferenceSpace(cs)

V = Geometry.WVector.(ones(FT, fs))
θ = sin.(Fields.coordinate_field(cs).z)

# Solve advection Equation: ∂θ/dt = -∂(vθ)

# upwinding
function tendency1!(dθ, θ, _, t)
    fcc = Operators.FluxCorrectionC2C(
        left = Operators.Extrapolate(),
        right = Operators.Extrapolate(),
    )
    fcf = Operators.FluxCorrectionF2F(
        left = Operators.Extrapolate(),
        right = Operators.Extrapolate(),
    )
    UB = Operators.UpwindBiasedProductC2F(
        left = Operators.SetValue(sin(a - t)),
        right = Operators.SetValue(sin(b - t)),
    )
    ∂ = Operators.DivergenceF2C()

    return @. dθ = -∂(UB(V, θ))
end
function tendency2!(dθ, θ, _, t)
    fcc = Operators.FluxCorrectionC2C(
        left = Operators.Extrapolate(),
        right = Operators.Extrapolate(),
    )
    fcf = Operators.FluxCorrectionF2F(
        left = Operators.Extrapolate(),
        right = Operators.Extrapolate(),
    )
    UB = Operators.UpwindBiasedProductC2F(
        left = Operators.SetValue(sin(a - t)),
        right = Operators.SetValue(sin(b - t)),
    )
    ∂ = Operators.DivergenceF2C()
    return @. dθ = -∂(UB(V, θ)) + fcc(V, θ)
end
# use the advection operator
function tendency3!(dθ, θ, _, t)

    fcc = Operators.FluxCorrectionC2C(
        left = Operators.Extrapolate(),
        right = Operators.Extrapolate(),
    )
    fcf = Operators.FluxCorrectionF2F(
        left = Operators.Extrapolate(),
        right = Operators.Extrapolate(),
    )
    A = Operators.AdvectionC2C(
        left = Operators.SetValue(sin(-t)),
        right = Operators.Extrapolate(),
    )
    return @. dθ = -A(V, θ)
end
# use the advection operator
function tendency4!(dθ, θ, _, t)

    fcc = Operators.FluxCorrectionC2C(
        left = Operators.Extrapolate(),
        right = Operators.Extrapolate(),
    )
    fcf = Operators.FluxCorrectionF2F(
        left = Operators.Extrapolate(),
        right = Operators.Extrapolate(),
    )
    A = Operators.AdvectionC2C(
        left = Operators.SetValue(sin(-t)),
        right = Operators.Extrapolate(),
    )
    return @. dθ = -A(V, θ) + fcc(V, θ)
end

# use the advection operator

@show tendency1!(similar(θ), θ, nothing, 0.0)
# Solve the ODE operator
Δt = 0.001
prob1 = ODEProblem(tendency1!, θ, (0.0, 10.0))
prob2 = ODEProblem(tendency2!, θ, (0.0, 10.0))
prob3 = ODEProblem(tendency3!, θ, (0.0, 10.0))
prob4 = ODEProblem(tendency4!, θ, (0.0, 10.0))
sol1 = solve(
    prob1,
    SSPRK33(),
    dt = Δt,
    saveat = collect(0.0:(10 * Δt):10.0),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);
sol2 = solve(
    prob2,
    SSPRK33(),
    dt = Δt,
    saveat = collect(0.0:(10 * Δt):10.0),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);
sol3 = solve(
    prob3,
    SSPRK33(),
    dt = Δt,
    saveat = collect(0.0:(10 * Δt):10.0),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);
sol4 = solve(
    prob4,
    SSPRK33(),
    dt = Δt,
    saveat = collect(0.0:(10 * Δt):10.0),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = "advect"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

anim = Plots.@animate for u in sol1.u
    Plots.plot(u, xlim = (-1, 1))
end
Plots.mp4(anim, joinpath(path, "UBP_advect.mp4"), fps = 10)
Plots.png(
    Plots.plot(sol1.u[end], xlim = (-1, 1)),
    joinpath(path, "sol1_advect_end.png"),
)

anim = Plots.@animate for u in sol2.u
    Plots.plot(u, xlim = (-1, 1))
end
Plots.mp4(anim, joinpath(path, "UBP_advect_fc.mp4"), fps = 10)
Plots.png(
    Plots.plot(sol2.u[end], xlim = (-1, 1)),
    joinpath(path, "sol2_advect_end.png"),
)

anim = Plots.@animate for u in sol3.u
    Plots.plot(u, xlim = (-1, 1))
end
Plots.mp4(anim, joinpath(path, "C2C_advect.mp4"), fps = 10)
Plots.png(
    Plots.plot(sol3.u[end], xlim = (-1, 1)),
    joinpath(path, "sol3_advect_end.png"),
)

anim = Plots.@animate for u in sol4.u
    Plots.plot(u, xlim = (-1, 1))
end
Plots.mp4(anim, joinpath(path, "C2C_advect_fc.mp4"), fps = 10)
Plots.png(
    Plots.plot(sol4.u[end], xlim = (-1, 1)),
    joinpath(path, "sol4_advect_end.png"),
)

p = Plots.plot(sol1.u[end], xlim = (-1, 1), ls = :dash, label = "UBP")
p = Plots.plot!(sol2.u[end], xlim = (-1, 1), ls = :dot, label = "UBP_FC")
p = Plots.plot!(sol3.u[end], xlim = (-1, 1), ls = :solid, label = "C2C")
p = Plots.plot!(sol4.u[end], xlim = (-1, 1), ls = :dashdot, label = "C2C_FC")
Plots.png(p, joinpath(path, "all_advect_end.png"))

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig(
    relpath(joinpath(path, "advect_end.png"), joinpath(@__DIR__, "../..")),
    "Advect End Simulation",
)
