include("inertial_gravity_wave_utils.jl")

ode_algorithm = IMEXEuler

prob = inertial_gravity_wave_prob(;
    𝔼_var = :ρθ,
    𝕄_var = :w,
    helem = 75,
    velem = 10,
    npoly = 4,
    is_large_domain = true,
    ode_algorithm = ode_algorithm,
    is_imex = true,
    tspan = (0., 40000.),
    J_𝕄ρ_overwrite = :grav,
)

sol = solve(
    prob,
    ode_algorithm(linsolve = linsolve!, nlsolve = NLNewton(; max_iter = 10));
    dt = 1.,
    reltol = 1e-1,
    abstol = 1e-6,
    adaptive = false,
    saveat = 200.,
    progress = true,
    progress_steps = 1,
    progress_message = (dt, u, p, t) -> t,
)

inertial_gravity_wave_plots(sol, "large_domain_kencarp4_imex_25s_10000s")