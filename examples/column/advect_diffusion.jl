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

const CI = !isnothing(get(ENV, "CI", nothing))

const FT = Float64

n = 32
z₀ = FT(0)
z₁ = FT(10)
t₀ = FT(0)
t₁ = FT(10)
μ = FT(-1 / 2)
ν = FT(5)
𝓌 = FT(1)
δ = FT(1)

domain = Domains.IntervalDomain(z₀, z₁, x3boundary = (:bottom, :top))
mesh = Meshes.IntervalMesh(domain, nelems = n)

cs = Spaces.CenterFiniteDifferenceSpace(mesh)
fs = Spaces.FaceFiniteDifferenceSpace(cs)
zc = Fields.coordinate_field(cs)
zp = (z₀ + z₁ / n / 2):(z₁ / n):(z₁ - z₁ / n / 2)


function gaussian(z, t; μ = -1 // 2, ν = 1, 𝓌 = 1, δ = 1)
    return exp(-(z - μ - 𝓌 * t)^2 / (4 * ν * (t + δ))) / sqrt(1 + t / δ)
end
function ∇gaussian(z, t; μ = -1 // 2, ν = 1, 𝓌 = 1, δ = 1)
    return -2 * (z - μ - 𝓌 * t) / (4 * ν * (δ + t)) *
           exp(-(z - μ - 𝓌 * t)^2 / (4 * ν * (δ + t))) / sqrt(1 + t / δ)
end

T = gaussian.(zc, -0; μ = μ, δ = δ, ν = ν, 𝓌 = 𝓌)
V = Geometry.Cartesian3Vector.(ones(FT, fs))

# Solve Adv-Diff Equation: ∂_t T = α ∇²T
z₀ = FT(0)
z₁ = FT(10)

function ∑tendencies!(dT, T, z, t)

    ic2f = Operators.InterpolateC2F()
    bc_vb = Operators.SetValue(FT(gaussian(z₀, t; ν = ν, δ = δ, 𝓌 = 𝓌, μ = μ)))
    bc_vt = Operators.SetValue(FT(gaussian(z₁, t; ν = ν, δ = δ, 𝓌 = 𝓌, μ = μ)))
    bc_gb = Operators.SetGradient(
        Geometry.Cartesian3Vector(
            FT(∇gaussian(z₀, t; ν = ν, δ = δ, 𝓌 = 𝓌, μ = μ)),
        ),
    )
    bc_gt = Operators.SetGradient(
        Geometry.Cartesian3Vector(
            FT(∇gaussian(z₁, t; ν = ν, δ = δ, 𝓌 = 𝓌, μ = μ)),
        ),
    )

    #   Upwind Biased Product
    #   UB = Operators.UpwindBiasedProductC2F(
    #       bottom = Operators.Extrapolate(),
    #       top = bc_vt,
    #   )
    #   ∂ = Operators.GradientF2C()
    #   return @. dT = -∂(UB(V, ic2f(T)))

    A = Operators.AdvectionC2C(bottom = bc_vb, top = Operators.Extrapolate())


    gradc2f = Operators.GradientC2F(bottom = bc_vb, top = bc_gt)
    divf2c = Operators.DivergenceF2C()

    return @. dT = divf2c(ν * gradc2f(T)) - A(V, T)
end

@show ∑tendencies!(similar(T), T, nothing, 0.0);

# Solve the ODE operator
Δt = 0.0001

prob = ODEProblem(∑tendencies!, T, (t₀, t₁))
sol = solve(
    prob,
    SSPRK33(),
    dt = Δt,
    saveat = 10000 * Δt,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dirname = "advect_diffusion"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

anim = Plots.@animate for (nt, u) in enumerate(sol.u)
    Plots.plot(
        u,
        xlim = (0, 1),
        ylim = (-1, 10),
        title = "$(nt-1) s",
        lc = :black,
        lw = 2,
        ls = :dash,
        label = "Approx Sol.",
        legend = :outerright,
        m = :o,
        xlabel = "T(z)",
        ylabel = "z",
    )
    Plots.plot!(
        gaussian.(zp, nt - 1; μ = μ, δ = δ, ν = ν, 𝓌 = 𝓌),
        zp,
        xlim = (0, 1),
        ylim = (-1, 10),
        title = "$(nt) s",
        lc = :red,
        lw = 2,
        label = "Analytical Sol.",
        m = :x,
    )
end
Plots.mp4(anim, joinpath(path, "advect_diffusion.mp4"), fps = 10)
Plots.png(
    Plots.plot(sol.u[end], xlim = (0, 1)),
    joinpath(path, "advect_diffusion_end.png"),
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
    "output/$(dirname)/advect_diffusion_end.png",
    "Advection-Diffusion End Simulation",
)
