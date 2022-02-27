using Test
using ClimaCorePlots, Plots
using DiffEqCallbacks

include("baroclinic_wave_utilities.jl")

jld2_callback = PeriodicCallback(output_writer, 864000; initial_affect = true)

driver_values(FT) = (;
    zmax = FT(30.0e3),
    velem = 10,
    helem = 4,
    npoly = 4,
    tmax = FT(60 * 60 * 24 * 10),
    dt = FT(400.0),
    ode_algorithm = OrdinaryDiffEq.Rosenbrock23,
    jacobian_flags = (; ∂𝔼ₜ∂𝕄_mode = :exact, ∂𝕄ₜ∂ρ_mode = :exact),
    max_newton_iters = 2,
    save_every_n_steps = 216,
    additional_solver_kwargs = (; callback = jld2_callback),
)

initial_condition(local_geometry) = initial_condition_ρθ(local_geometry)
initial_condition_velocity(local_geometry) =
    initial_condition_velocity(local_geometry; is_balanced_flow = true)

remaining_cache_values(Y, dt) = merge(
    baroclinic_wave_cache_values(Y, dt),
    final_adjustments_cache_values(Y, dt; use_rayleigh_sponge = false),
)

function remaining_tendency!(dY, Y, p, t)
    dY .= zero(eltype(dY))
    baroclinic_wave_ρθ_remaining_tendency!(dY, Y, p, t; κ₄ = 2.0e17)
    final_adjustments!(
        dY,
        Y,
        p,
        t;
        use_flux_correction = false,
        use_rayleigh_sponge = false,
    )
    return dY
end

function postprocessing(sol, path)
    @info "L₂ norm of ρe at t = $(sol.t[1]): $(norm(sol.u[1].Yc.ρθ))"
    @info "L₂ norm of ρe at t = $(sol.t[end]): $(norm(sol.u[end].Yc.ρθ))"

    anim = Plots.@animate for Y in sol.u
        v = Geometry.UVVector.(Y.uₕ).components.data.:2
        Plots.plot(v, level = 3, clim = (-3, 3))
    end
    Plots.mp4(anim, joinpath(path, "v.mp4"), fps = 5)

    anim = Plots.@animate for Y in sol.u
        v = Geometry.UVVector.(Y.uₕ).components.data.:1
        Plots.plot(v, level = 3, clim = (-25, 25))
    end
    Plots.mp4(anim, joinpath(path, "u.mp4"), fps = 5)

    anim = Plots.@animate for Y in sol.u

        ρ = Y.Yc.ρ
        θ = @. Y.Yc.ρθ / Y.Yc.ρ

        Plots.plot(θ, level = 3, clim = (225, 255))
    end
    Plots.mp4(anim, joinpath(path, "theta.mp4"), fps = 5)
end
