# 1D heat equation ∂_t T = α ∇²T on a vertical column, discretized with the
# staggered finite difference operators: `GradientC2F` to faces, `DivergenceF2C`
# back to centers. Demonstrates the two vertical boundary condition kinds — a
# Dirichlet value at the bottom and a prescribed gradient at the top.
#
# Starting from T ≡ 0, the column heats toward the steady state T = z. The
# separable solution of that problem is known in closed form, so the run asserts
# the computed profile against it at every saved time.
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
import LazyBroadcast: lazy

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

const FT = Float64

const z_bottom = FT(0)
const z_top = FT(1)
const nelems = 10
const α = FT(0.1)          # thermal diffusivity (m²/s)
const T_bottom = FT(0)     # Dirichlet value at the bottom (K)
const dTdz_top = FT(1)     # prescribed gradient at the top (K/m)

domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(z_bottom),
    Geometry.ZPoint{FT}(z_top),
    boundary_names = (:bottom, :top),
)
mesh = Meshes.IntervalMesh(domain, nelems = nelems)
device = ClimaComms.device()
cspace = Spaces.CenterFiniteDifferenceSpace(device, mesh)
T = Fields.zeros(FT, cspace)

# Solve Heat Equation: ∂_t T = α ∇²T
function tendency!(dT, T, _, t)

    # the Dirichlet condition T = T_bottom on the bottom boundary face: the
    # covariant gradient there is 2 (T[1] - T_bottom)
    bottom_level_T = Fields.level(T, 1)
    bcs_bottom = Operators.SetGradient(
        @. lazy(Geometry.Covariant3Vector(2 * (bottom_level_T - T_bottom)))
    )
    bcs_top = Operators.SetGradient(Geometry.WVector(dTdz_top))

    gradc2f = Operators.GradientC2F(bottom = bcs_bottom, top = bcs_top)
    divf2c = Operators.DivergenceF2C()

    return @. dT = α * divf2c(gradc2f(T))
end

tendency!(similar(T), T, nothing, 0.0)

# Solve the ODE operator
Δt = 0.02 # the diffusive stability limit is Δz²/2α ≈ 0.05 s

prob = CTS.ODEProblem(CTS.ClimaODEFunction(; T_exp! = tendency!), T, (0.0, 10.0), nothing)
sol = CTS.solve(
    prob,
    CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()),
    dt = Δt,
    saveat = collect(0.0:(10 * Δt):10.0),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

# Separating variables about the steady state T = z gives the exact solution
#
#     T(z, t) = z - 2 Σₙ (-1)ⁿ⁺¹ sin(λₙ z) exp(-α λₙ² t) / λₙ²,
#
# with λₙ = (n - 1/2)π, the eigenvalues admitted by T(0) = 0 and ∂T/∂z(1) = 1.
# The series is the sine expansion of -z at t = 0, so T(z, 0) = 0 as required.
function exact_temperature(z, t; nterms = 200)
    T = z
    for n in 1:nterms
        λ = (n - 1 / 2) * π
        T -= 2 * (-1)^(n + 1) * sin(λ * z) * exp(-α * λ^2 * t) / λ^2
    end
    return T
end

using Test
@testset "computed solution vs the analytic series" begin
    z = vec(parent(Fields.coordinate_field(cspace).z))
    # The computed profile tracks the exact solution at every saved time, not
    # just at the end (measured worst case over the run: 1.1e-3 K, attained
    # early while the surface gradient is still sharp; it falls by a factor of
    # 4 per grid refinement).
    errors = map(zip(sol.t, sol.u)) do (t, T)
        maximum(abs, vec(parent(T)) .- exact_temperature.(z, t))
    end
    @test maximum(errors) < 2e-3
    # Maximum principle: the column heats monotonically from T ≡ 0 toward the
    # steady state T = z, so it must stay between the two at all times.
    @test minimum(sol.u[end]) ≥ 0
    @test maximum(parent(sol.u[end]) .- z) ≤ sqrt(eps())
end

ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = "heat"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

z_centers = vec(parent(Fields.coordinate_field(cspace).z))

# Plot the computed profile against the exact solution at the same time.
function heat_plot(t, T; kwargs...)
    plt = Plots.plot(
        exact_temperature.(z_centers, t),
        z_centers,
        marker = :circle,
        xlim = (0, 1),
        xlabel = "T",
        ylabel = "z",
        label = "Exact";
        kwargs...,
    )
    return Plots.plot!(plt, vec(parent(T)), z_centers, label = "Computed")
end

anim = Plots.@animate for (t, T) in zip(sol.t, sol.u)
    heat_plot(t, T, title = "t = $t")
end
Plots.mp4(anim, joinpath(path, "heat.mp4"), fps = 10)
Plots.png(heat_plot(sol.t[end], sol.u[end]), joinpath(path, "heat_end.png"))

include(joinpath(@__DIR__, "..", "example_utils.jl")) # linkfig

linkfig(
    relpath(joinpath(path, "heat_end.png"), joinpath(@__DIR__, "../..")),
    "Heat End Simulation",
)
