include("inertial_gravity_wave_utils.jl")

ode_algorithm = KenCarp4

prob = inertial_gravity_wave_prob(;
    𝔼_var = :ρθ,
    𝕄_var = :ρw,
    helem = 75,
    velem = 10,
    npoly = 4,
    is_large_domain = true,
    ode_algorithm = ode_algorithm,
    is_imex = true,
    tspan = (0., 10000.),
)

sol = solve(
    prob,
    ode_algorithm(linsolve = linsolve!);
    dt = 25.,
    adaptive = false,
    saveat = 10.,
    progress = true,
    progress_steps = 1,
    progress_message = (dt, u, p, t) -> t,
)

inertial_gravity_wave_plots(sol, "large_domain_kencarp4_imex_25s_10000s")