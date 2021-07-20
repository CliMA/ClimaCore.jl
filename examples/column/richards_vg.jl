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

using OrdinaryDiffEq: ODEProblem, solve, SSPRK33,Rosenbrock23, Tsit5

const FT = Float64

n = 15
z₀ = FT(-1.5)
z₁ = FT(0)
ksat = FT(34 / (3600 * 100))
vgn = FT(3.96)
vgα = FT(2.7)
vgm = FT(1)- FT(1)/vgn
θr = FT(0.075)
ν = FT(0.287)
θl_0 = FT(0.1)
θl_surf = FT(0.267)
Δt = FT(0.05)
time_end = FT(60 * 60 * 0.8)
t0 = FT(0)


domain = Domains.IntervalDomain(z₀, z₁, x3boundary = (:bottom, :top))
mesh = Meshes.IntervalMesh(domain, nelems = n)

cs = Spaces.CenterFiniteDifferenceSpace(mesh)
fs = Spaces.FaceFiniteDifferenceSpace(cs)
zc = Fields.coordinate_field(cs)
#zp = (z₀ + z₁ / n / 2):(z₁ / n):(z₁ - z₁ / n / 2)

###doesnt work without kwargs?
###ERROR: cannot infer concrete eltype of effective_saturation on (Float64,)

function effective_saturation(θ; ν = ν, θr = θr)

    if θ < θr
        println("Effective saturation is negative")
        println(θ)
    end

    if θ > ν
        println("Effective saturation is positive")
        println(θ)
    end
    return (θ-θr)/(ν-θr)
end

function matric_potential(S; vgn = vgn, vgα = vgα, vgm = vgm)
    if S < FT(0)
        println("Effective saturation is negative")
        println(S)
    end

    if S > FT(1)
        println("Effective saturation is positive")
        println(S)
    end
    
    ψ_m = -((S^(-FT(1) / vgm) - FT(1)) * vgα^(-vgn))^(FT(1) / vgn)
    return ψ_m
end

function hydraulic_conductivity(S; vgm = vgm, ksat = ksat)
    if S < FT(1)
        K = sqrt(S) * (FT(1) - (FT(1) - S^(FT(1) / vgm))^vgm)^FT(2)
    else
        K = FT(1)
    end
    return K*ksat
end


θl = Fields.zeros(FT, cs) .+ θl_0

# Solve Richard's Equation: ∂_t θl = ∂_z [K(θl)(∂_z ψ(θl) +1)]


function ∑tendencies!(dθl, θl, z, t)
    #the BC needs to be on the thing we take the gradient of.
    # in climatemachine.jl, i think it was on the flux or the prognostic variable,
    # and the flux could be a grad of a function of the prognostic variable. 

    S_top = effective_saturation.(FT(0.1); ν = ν, θr = θr)
    h_top = matric_potential(S_top; vgn = vgn, vgα = vgα, vgm = vgm)#ztop = 0
    bc_t = Operators.SetValue(h_top)
    bc_b = Operators.SetGradient(FT(1))

    gradc2f = Operators.GradientC2F(bottom = bc_b, top = bc_t)
    gradf2c = Operators.GradientF2C()
    
    S = effective_saturation.(θl; ν = ν, θr = θr)
    K = hydraulic_conductivity.(S; vgm = vgm, ksat = ksat)
    ψ = matric_potential.(S; vgn = vgn, vgα = vgα, vgm = vgm)
    h = ψ .+ zc
    #mismatched spaces if K * grad*h
    Kface = Operators.InterpolateC2F(K)
    return @. dθl = gradf2c( Kface .* gradc2f(h))
    
end
#@show ∑tendencies!(similar(θl), θl, nothing, 0.0);

# Solve the ODE operator

prob = ODEProblem(∑tendencies!, θl, (t0, time_end))
sol = solve(
    prob,
    Tsit5(),
    dt = Δt,
    saveat = 100 * Δt,
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
