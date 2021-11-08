include("inertial_gravity_wave_utils.jl")

ode_algorithm = Rosenbrock23

prob = inertial_gravity_wave_prob(;
    𝔼_var = :ρθ,
    𝕄_var = :ρw,
    helem = 75,
    velem = 10,
    npoly = 4,
    is_large_domain = true,
    ode_algorithm = ode_algorithm,
    is_imex = false,
    tspan = (0., 10000.),
    J_𝕄ρ_overwrite = :none,
)

sol = solve(
    prob,
    ode_algorithm(linsolve = linsolve!);
    dt = 5.,
    adaptive = false,
    saveat = 10.,
    progress = true,
    progress_steps = 1,
    progress_message = (dt, u, p, t) -> t,
)

inertial_gravity_wave_plots(sol, "large_domain_w_rosenbrock23_5s_10000s")