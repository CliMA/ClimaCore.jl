push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

import ClimateMachineCore.Geometry, LinearAlgebra, UnPack
import ClimateMachineCore:
    Fields, Domains, Topologies, Meshes, DataLayouts, Operators, Geometry

using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

const FT = Float64

a = FT(0.0)
b = FT(1.0)
n = 10
α = FT(0.1)

cs = Spaces.FaceFiniteDifferenceSpace(a, b, n)

T = Fields.Field(DataLayouts.VF{FT}(zeros(FT, Spaces.n_cells(cs), 1)), cs)
∇²T = Fields.Field(DataLayouts.VF{FT}(zeros(FT, Spaces.n_cells(cs), 1)), cs)
∇T = Fields.Field(DataLayouts.VF{FT}(zeros(FT, Spaces.n_faces(cs), 1)), cs)

# Solve Heat Equation: ∂_t T = α ∇²T
function ∑tendencies!(dT, T, _, t)

    # apply boundry conditions 
    Operators.apply_dirichlet!(T, 0, cs, Meshes.ColumnMin())
    Operators.apply_neumann!(T, 1, cs, Meshes.ColumnMax())

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
