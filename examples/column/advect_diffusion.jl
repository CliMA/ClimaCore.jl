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

n = 32
z₀ = FT(0)
z₁ = FT(10)
t₀ = FT(0)
t₁ = FT(10)
μ = FT(-1 / 2)
ν = FT(5)
𝓌 = FT(1)
δ = FT(1)

domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(z₀),
    Geometry.ZPoint{FT}(z₁),
    boundary_names = (:bottom, :top),
)
mesh = Meshes.IntervalMesh(domain, nelems = n)
device = ClimaComms.device()
cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
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

T = gaussian.(zc.z, -0; μ = μ, δ = δ, ν = ν, 𝓌 = 𝓌)
V = Geometry.WVector.(ones(FT, fs))

# Solve Adv-Diff Equation: ∂_t T = α ∇²T - v⋅∇T
z₀ = FT(0)
z₁ = FT(10)

function ∑tendencies!(dT, T, z, t)
    bc_gb = Operators.SetGradient(
        Geometry.WVector(FT(∇gaussian(z₀, t; ν = ν, δ = δ, 𝓌 = 𝓌, μ = μ))),
    )
    bc_gt = Operators.SetGradient(
        Geometry.WVector(FT(∇gaussian(z₁, t; ν = ν, δ = δ, 𝓌 = 𝓌, μ = μ))),
    )
    top_center_left_biased_grad =
        Geometry.Covariant3Vector.(
            Fields.level(T, Fields.nlevels(T)) .- Fields.level(T, Fields.nlevels(T) - 1)
        )

    bc_gt_lb = Operators.SetGradient(top_center_left_biased_grad)
    gradc2f = Operators.GradientC2F(bottom = bc_gb, top = bc_gt)
    gradc2f_advect = Operators.GradientC2F(bottom = bc_gb, top = bc_gt_lb)
    interpf2c = Operators.InterpolateF2C()
    divf2c = Operators.DivergenceF2C()

    return @. dT =
        divf2c(ν * gradc2f(T)) -
        interpf2c(Geometry.dot(Geometry.Contravariant3Vector(V), gradc2f_advect(T)))
end

@show ∑tendencies!(similar(T), T, nothing, 0.0);

# Solve the ODE operator
Δt = 0.0001

prob = ODEProblem(∑tendencies!, T, (t₀, t₁))
sol = solve(
    prob,
    SSPRK33(),
    dt = Δt,
    saveat = collect(t₀:(10000 * Δt):t₁),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = "advect_diffusion"
path = joinpath(@__DIR__, "output", dir)
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
    relpath(
        joinpath(path, "advect_diffusion_end.png"),
        joinpath(@__DIR__, "../.."),
    ),
    "Advection-Diffusion End Simulation",
)
