include("inertial_gravity_wave_utils.jl")

ode_algorithm = SSPRK33

prob = inertial_gravity_wave_prob(;
    helem = 75,
    velem = 10,
    npoly = 4,
    is_large_domain = true,
    ode_algorithm = ode_algorithm,
    is_imex = false,
    tspan = (0., 10000.),
)

sol = solve(
    prob,
    ode_algorithm();
    dt = 2.5,
    adaptive = false,
    saveat = 10.,
    progress = true,
    progress_steps = 1,
    progress_message = (dt, u, p, t) -> t,
)

inertial_gravity_wave_plots(sol, "large_domain_ssprk33_2.5s_10000s")