push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

import ClimateMachineCore.Geometry, LinearAlgebra, UnPack
import ClimateMachineCore:
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
b = FT(10.0)
n = 10
α = FT(0.1)

cs = Spaces.FaceFiniteDifferenceSpace(a, b, n)

T = Fields.CentField(cs)
∇²T = Fields.CentField(cs)
∇T = Fields.FaceField(cs)

# Solve Heat Equation: ∂_t T = α ∇²T
function ∑tendencies!(dT, T, _, t)

    # apply boundry conditions
    Operators.apply_dirichlet!(T, 0, cs, Spaces.ColumnMin())
    Operators.apply_neumann!(T, 1, cs, Spaces.ColumnMax())

    # compute laplacian
    Operators.vertical_gradient!(∇T, T, cs)
    Operators.vertical_gradient!(∇²T, ∇T, cs)

    # update
    dT .= α .* ∇²T
end

# ∑tendencies!(similar(T), T, nothing, 0.0);

# Solve the ODE operator
prob = ODEProblem(∑tendencies!, T, (0.0, 10.0))
sol = solve(
    prob,
    SSPRK33(),
    dt = 0.02,
    saveat = 1.0,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)
nothing # suppress solution output

using Plots
ENV["GKSwstype"] = "nul"

path = joinpath(@__DIR__, "output")
mkpath(path)

anim = @animate for u in sol.u
    plot(
        parent(u),
        Spaces.coordinates(cs, Spaces.CellCent()),
        ylabel = "z",
        xlabel = "T(z)",
    )
end
mp4(anim, joinpath(path, "temperature.mp4"), fps = 10)
