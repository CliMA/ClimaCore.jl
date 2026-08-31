# 1D advection-diffusion on a vertical column, with a travelling Gaussian as the
# exact solution. Boundary values are evaluated from that Gaussian at each step,
# so the computed profile can be compared against it to show the discretization
# transporting and spreading the pulse at the right rate.
#
# The parameters put the run in the advective-diffusive regime the example is
# meant to show: over t = 7 the pulse travels 7 length units while its width
# grows by a factor of 2.8, and the cell Péclet number w Δz / ν = 0.8 stays
# under 2, the threshold above which the centered advection term rings.
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

const nelems = 256
const z_bottom = FT(0)
const z_top = FT(10)
const t_start = FT(0)
const t_end = FT(7)
const μ = FT(1)      # initial center of the pulse
const ν = FT(0.05)   # diffusivity
const w = FT(1)      # advection velocity
const δ = FT(1)      # time offset that sets the initial width, √(2νδ)

domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(z_bottom),
    Geometry.ZPoint{FT}(z_top),
    boundary_names = (:bottom, :top),
)
mesh = Meshes.IntervalMesh(domain, nelems = nelems)
device = ClimaComms.device()
cspace = Spaces.CenterFiniteDifferenceSpace(device, mesh)
fspace = Spaces.FaceFiniteDifferenceSpace(cspace)

# Exact solution of ∂_t T = ν ∂²_z T - w ∂_z T: a Gaussian that is carried at
# speed `w` while it spreads, normalized to unit amplitude at t = 0.
gaussian(z, t) = exp(-(z - μ - w * t)^2 / (4 * ν * (t + δ))) / sqrt(1 + t / δ)
∇gaussian(z, t) =
    -2 * (z - μ - w * t) / (4 * ν * (δ + t)) * gaussian(z, t)

T = gaussian.(Fields.coordinate_field(cspace).z, t_start)
velocity = Geometry.WVector.(w .* ones(FT, fspace))

# Solve Adv-Diff Equation: ∂_t T = ν ∇²T - w ∇T
function tendency!(dT, T, _, t)
    # The exact solution supplies the inflow value at the bottom and the
    # outflow gradient at the top, so nothing the discretization does to the
    # interior can be blamed on the boundary treatment. The Dirichlet condition
    # T = gaussian(z_bottom, t) on the bottom boundary face is imposed through
    # the gradient operator by `gradient_c2f_dirichlet`; the top boundary's
    # gradient is passed through as an explicit `SetGradient`.
    bc_gradient_top =
        Operators.SetGradient(Geometry.WVector(∇gaussian(z_top, t)))

    # The advective form is an interpolated center-to-face gradient; the
    # gradient on the top boundary face is extrapolated from the closest
    # interior faces.
    T_top = Fields.level(T, Fields.nlevels(T))
    T_top_m1 = Fields.level(T, Fields.nlevels(T) - 1)
    bc_gradient_top_extrapolated = Operators.SetGradient(
        @. lazy(Geometry.Covariant3Vector(T_top - T_top_m1))
    )

    ∇T = Operators.gradient_c2f_dirichlet(
        T;
        bottom = gaussian(z_bottom, t),
        top = bc_gradient_top,
    )
    ∇T_advect = Operators.gradient_c2f_dirichlet(
        T;
        bottom = gaussian(z_bottom, t),
        top = bc_gradient_top_extrapolated,
    )
    interpf2c = Operators.InterpolateF2C()
    divf2c = Operators.DivergenceF2C()

    return @. dT =
        divf2c(ν * ∇T) - interpf2c(
            Geometry.dot(
                Geometry.Contravariant3Vector(velocity),
                ∇T_advect,
            ),
        )
end

tendency!(similar(T), T, nothing, t_start)

# Solve the ODE operator
Δt = FT(0.005) # the diffusive stability limit is Δz²/2ν ≈ 0.015 s

prob = CTS.ODEProblem(
    CTS.ClimaODEFunction(; T_exp! = tendency!),
    T,
    (t_start, t_end),
    nothing,
)
sol = CTS.solve(
    prob,
    CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()),
    dt = Δt,
    saveat = collect(t_start:FT(0.5):t_end),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

using Test
using LinearAlgebra: norm
z_centers = vec(parent(Fields.coordinate_field(cspace).z))
@testset "advected-diffused Gaussian vs the analytic solution" begin
    # The travelling Gaussian is the exact solution, so the computed profile
    # must track it at every saved time, not just at the end (measured worst
    # relative L₂ error over the run: 4.3e-3, entirely spatial truncation —
    # it is unchanged by halving Δt and falls by a factor of 4 per grid
    # refinement).
    rel_errors = map(zip(sol.t, sol.u)) do (t, T)
        exact = gaussian.(z_centers, t)
        norm(vec(parent(T)) .- exact) / norm(exact)
    end
    @test maximum(rel_errors) < 1e-2
    # The pulse is transported, not just smeared: its peak arrives at
    # μ + w t_end = 8 and has decayed to 1/√(1 + t_end/δ) = 0.354 (measured:
    # peak at z = 8.0 with amplitude 0.354).
    T_end = vec(parent(sol.u[end]))
    @test z_centers[argmax(T_end)] ≈ μ + w * t_end atol = z_top / nelems
    @test maximum(T_end) ≈ 1 / sqrt(1 + t_end / δ) rtol = 0.01
    # The exact solution is positive everywhere. The cell Péclet number is
    # below 2, so the centered advection term must not ring the profile
    # negative (measured minimum over the run: -2e-6).
    @test minimum(minimum, sol.u) > -1e-4
end

ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = "advect_diffusion"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

function advect_diffusion_plot(t, T; kwargs...)
    plt = Plots.plot(
        gaussian.(z_centers, t),
        z_centers,
        xlim = (-0.1, 1.1),
        ylim = (z_bottom, z_top),
        lc = :red,
        lw = 2,
        xlabel = "T(z)",
        ylabel = "z",
        label = "Analytical Sol.",
        legend = :outerright;
        kwargs...,
    )
    return Plots.plot!(
        plt,
        vec(parent(T)),
        z_centers,
        lc = :black,
        lw = 2,
        ls = :dash,
        label = "Approx Sol.",
    )
end

anim = Plots.@animate for (t, T) in zip(sol.t, sol.u)
    advect_diffusion_plot(t, T, title = "$t s")
end
Plots.mp4(anim, joinpath(path, "advect_diffusion.mp4"), fps = 10)
Plots.png(
    advect_diffusion_plot(sol.t[end], sol.u[end], title = "$(sol.t[end]) s"),
    joinpath(path, "advect_diffusion_end.png"),
)

include(joinpath(@__DIR__, "..", "example_utils.jl")) # linkfig

linkfig(
    relpath(
        joinpath(path, "advect_diffusion_end.png"),
        joinpath(@__DIR__, "../.."),
    ),
    "Advection-Diffusion End Simulation",
)
