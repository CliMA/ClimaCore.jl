using ClimaCorePlots, Plots
using DiffEqCallbacks

include("baroclinic_wave_utilities.jl")

function cb(Y, t, integrator)
    saved_Ys = integrator.p.saved_Ys

    N_saved = length(saved_Ys)
    for i in 1:(N_saved - 1)
        saved_Ys[i] .= saved_Ys[i + 1]
    end
    saved_Ys[N_saved] .= Y
end

driver_values(FT) = (;
    zmax = FT(30.0e3),
    velem = 10,
    helem = 4,
    npoly = 4,
    tmax = FT(60 * 60 * 24 * 870),
    dt = FT(500.0),
    ode_algorithm = OrdinaryDiffEq.Rosenbrock23,
    jacobian_flags = (; ∂𝔼ₜ∂𝕄_mode = :constant_P, ∂𝕄ₜ∂ρ_mode = :exact),
    max_newton_iters = 2,
    save_every_n_steps = 1728,
    additional_solver_kwargs = (; callback = FunctionCallingCallback(cb)), # e.g., reltol, abstol, etc.
)

initial_condition(local_geometry) =
    initial_condition_ρe(local_geometry; is_balanced_flow = false)
initial_condition_velocity(local_geometry) =
    initial_condition_velocity(local_geometry; is_balanced_flow = false)

remaining_cache_values(Y, dt) = merge(
    baroclinic_wave_cache_values(Y, dt),
    final_adjustments_cache_values(Y, dt; use_rayleigh_sponge = false),
    (; saved_Ys = [copy(Y) for _ in 1:100]),
)

function remaining_tendency!(dY, Y, p, t)
    dY .= zero(eltype(dY))
    baroclinic_wave_ρe_remaining_tendency!(dY, Y, p, t; κ₄ = 2.0e17)
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
    @info "L₂ norm of ρe at t = $(sol.t[1]): $(norm(sol.u[1].Yc.ρe))"
    @info "L₂ norm of ρe at t = $(sol.t[end]): $(norm(sol.u[end].Yc.ρe))"

    anim = Plots.@animate for Y in sol.u
        v = Geometry.UVVector.(Y.uₕ).components.data.:2
        Plots.plot(v, level = 3, clim = (-3, 3))
    end
    Plots.mp4(anim, joinpath(path, "v.mp4"), fps = 5)

    anim = Plots.@animate for Y in sol.u
        cuₕ = Y.uₕ
        fw = Y.w
        cw = If2c.(fw)
        cuvw = Geometry.Covariant123Vector.(cuₕ) .+ Geometry.Covariant123Vector.(cw)
        normuvw = norm(cuvw)
        ρ = Y.Yc.ρ
        e_tot = @. Y.Yc.ρe / Y.Yc.ρ
        Φ = p.Φ
        I = @. e_tot - Φ - normuvw^2 / 2
        T = @. I / cv_d + T_tri
        Plots.plot(T, level = 3, clim = (225, 255))
    end
    Plots.mp4(anim, joinpath(path, "T.mp4"), fps = 5)
end

