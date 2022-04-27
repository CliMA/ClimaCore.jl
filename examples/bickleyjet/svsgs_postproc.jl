ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = "cg_nosgs$(n1)_difftracer"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.ρθ, clim = (-1, 1))
end
Plots.mp4(anim, joinpath(path, "tracer.mp4"), fps = 10)

anim = Plots.@animate for u in sol.u
    grad = Operators.Gradient()
    𝒰 = @. u.ρu / u.ρ
    ∇𝒰 = @. grad(𝒰)
    𝒮 = strainrate(∇𝒰)
    X = @. 𝒮.components.data.:1
    Plots.plot(X)
end
Plots.mp4(anim, joinpath(path, "strainrate11.mp4"), fps = 10)

anim = Plots.@animate for u in sol.u
    grad = Operators.Gradient()
    𝒰 = @. u.ρu / u.ρ
    ∇𝒰 = @. grad(𝒰)
    𝒮 = strainrate(∇𝒰)
    X = @. 𝒮.components.data.:2
    Plots.plot(X)
end
Plots.mp4(anim, joinpath(path, "strainrate12.mp4"), fps = 10)

anim = Plots.@animate for u in sol.u
    grad = Operators.Gradient()
    𝒰 = @. u.ρu / u.ρ
    ∇𝒰 = @. grad(𝒰)
    𝒮 = strainrate(∇𝒰)
    X = @. 𝒮.components.data.:3
    Plots.plot(X)
end
Plots.mp4(anim, joinpath(path, "strainrate21.mp4"), fps = 10)

anim = Plots.@animate for u in sol.u
    grad = Operators.Gradient()
    𝒰 = @. u.ρu / u.ρ
    ∇𝒰 = @. grad(𝒰)
    𝒮 = strainrate(∇𝒰)
    X = @. 𝒮.components.data.:4
    Plots.plot(X)
end
Plots.mp4(anim, joinpath(path, "strainrate22.mp4"), fps = 10)

anim = Plots.@animate for u in sol.u
    grad = Operators.Gradient()
    𝒰 = @. u.ρu / u.ρ
    ∇𝒰 = @. grad(𝒰)
    X = @. ∇𝒰.components.data.:1
    Plots.plot(X)
end
Plots.mp4(anim, joinpath(path, "velgrad11.mp4"), fps = 10)

anim = Plots.@animate for u in sol.u
    grad = Operators.Gradient()
    𝒰 = @. u.ρu / u.ρ
    ∇𝒰 = @. grad(𝒰)
    X = @. ∇𝒰.components.data.:2
    Plots.plot(X)
end
Plots.mp4(anim, joinpath(path, "velgrad12.mp4"), fps = 10)

anim = Plots.@animate for u in sol.u
    grad = Operators.Gradient()
    𝒰 = @. u.ρu / u.ρ
    ∇𝒰 = @. grad(𝒰)
    X = @. ∇𝒰.components.data.:3
    Plots.plot(X)
end
Plots.mp4(anim, joinpath(path, "velgrad21.mp4"), fps = 10)

anim = Plots.@animate for u in sol.u
    grad = Operators.Gradient()
    𝒰 = @. u.ρu / u.ρ
    ∇𝒰 = @. grad(𝒰)
    X = @. ∇𝒰.components.data.:4
    Plots.plot(X)
end
Plots.mp4(anim, joinpath(path, "velgrad22.mp4"), fps = 10)

anim = Plots.@animate for u in sol.u
    # Define Operators
    I = Operators.Interpolate(Ispace)
    div = Operators.WeakDivergence()
    grad = Operators.Gradient()
    R = Operators.Restrict(space)
    𝒰 = @. u.ρu / u.ρ
    ∇𝒰 = @. R(grad(I(𝒰)))
    𝒮 = strainrate(∇𝒰)
    E = compute_ℯᵥ(𝒮)
    ℯᵥ¹ = @. E.components.data.:1
    ℯᵥ² = @. E.components.data.:2
    𝒮₁₁ = @. 𝒮.components.data.:1
    𝒮₁₂ = @. 𝒮.components.data.:2
    𝒮₂₁ = @. 𝒮.components.data.:3
    𝒮₂₂ = @. 𝒮.components.data.:4
    ã₁ = @. ℯᵥ¹*ℯᵥ¹*𝒮₁₁ 
    ã₂ = @. ℯᵥ¹*ℯᵥ²*𝒮₁₂
    ã₃ = @. ℯᵥ²*ℯᵥ¹*𝒮₂₁
    ã₄ = @. ℯᵥ²*ℯᵥ²*𝒮₂₂
    ã = @. abs(ã₁ + ã₂ + ã₃ + ã₄) 
    # Compute Subgrid Tendency Based on Vortex Model
    k₁ = parameters.k₁
    kc = π / Δx
    F₂x = structure_function(𝒰.components.data.:1) # 4.5b
    F₂y = structure_function(𝒰.components.data.:2) # 4.5b
    K₀εx = @. kolmogorov_prefactor(F₂x)
    K₀εy = @. kolmogorov_prefactor(F₂y)
    Q = @. 2*parameters.ν*kc^2/3/(ã + 1e-14)
    Γ = @. gamma(-k₁, Q)
    Kₑx = @. 1/2 * K₀εx * (2*parameters.ν/3/(ã + 1e-14))^(k₁) * Γ # (4.4)
    Kₑy = @. 1/2 * K₀εy * (2*parameters.ν/3/(ã + 1e-14))^(k₁) * Γ # (4.4)
    # Get SGS Flux
    τ = compute_subgrid_stress(Kₑx, Kₑy, E, ∇𝒰)
    flux_sgs = @. u.ρ * τ
    divflux_sgs = @. R(div(I(flux_sgs)))
    #Plots.plot(flux_sgs.components.data.:1)
    Plots.plot(divflux_sgs.components.data.:1)
end
Plots.mp4(anim, joinpath(path, "divsgsflux.mp4"), fps = 10)

anim = Plots.@animate for u in sol.u
    # Define Operators
    I = Operators.Interpolate(Ispace)
    div = Operators.WeakDivergence()
    grad = Operators.Gradient()
    R = Operators.Restrict(space)
    rparameters = Ref(parameters)
    @. dydt = -R(div(flux(I(u), rparameters)))
    Plots.plot(dydt.ρθ)
end
Plots.mp4(anim, joinpath(path, "x_momentum_tendency.mp4"), fps = 10)


Es = [total_energy(u, parameters) for u in sol.u]
Plots.png(Plots.plot(Es), joinpath(path, "energy.png"))
jldsave("energy_nosgs_$(n1).jld2"; E=Es)

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig(
    relpath(joinpath(path, "energy.png"), joinpath(@__DIR__, "../..")),
    "Total Energy",
)
