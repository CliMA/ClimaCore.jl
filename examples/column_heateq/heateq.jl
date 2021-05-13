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

cm = Meshes.FaceColumnMesh(a, b, n)

T = Fields.Field(DataLayouts.VF{FT}(zeros(FT, Meshes.n_cells(cm), 1)), cm)
∇²T = Fields.Field(DataLayouts.VF{FT}(zeros(FT, Meshes.n_cells(cm), 1)), cm)
∇T = Fields.Field(DataLayouts.VF{FT}(zeros(FT, Meshes.n_faces(cm), 1)), cm)

# Solve Heat Equation: ∂_t T = α ∇²T
function ∑tendencies!(dT, T, _, t)

    # apply boundry conditions 
    Operators.apply_dirichlet!(T, 0, cm, Meshes.ColumnMin())
    Operators.apply_neumann!(T, 1, cm, Meshes.ColumnMax())

    # compute laplacian
    Operators.vertical_gradient!(∇T, T, cm)
    Operators.vertical_gradient!(∇²T, ∇T, cm)

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
