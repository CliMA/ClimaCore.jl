using ClimaComms
using LinearAlgebra

import ClimaCore:
    Domains, Fields, Geometry, Meshes, Operators, Spaces, Topologies, Limiters

using OrdinaryDiffEq, Test

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

const context = ClimaComms.SingletonCommsContext()

"""
    convergence_rate(err, Δh)

Estimate convergence rate given vectors `err` and `Δh`

    err = C Δh^p + H.O.T
    err_k ≈ C Δh_k^p
    err_k/err_m ≈ Δh_k^p/Δh_m^p
    log(err_k/err_m) ≈ log((Δh_k/Δh_m)^p)
    log(err_k/err_m) ≈ p*log(Δh_k/Δh_m)
    log(err_k/err_m)/log(Δh_k/Δh_m) ≈ p

"""
convergence_rate(err, Δh) =
    [log(err[i] / err[i - 1]) / log(Δh[i] / Δh[i - 1]) for i in 2:length(Δh)]

# Advection problem on a sphere with bounds-preserving quasimonotone limiter.
# The initial condition can be set via a command line argument.
# Possible test cases are: cosine_bells (default), gaussian_bells, and cylinders

const R = 6.37122e6  # sphere radius
const r0 = R / 2     # bells radius
const ρ₀ = 1.0       # air density
const D₄ = 6.6e14    # hyperdiffusion coefficient
const u0 = 2 * pi * R / (86400 * 12)
const T = 86400 * 120 # simulation period in seconds (12 days)
const n_steps = 12000 
const dt = T / n_steps
const centers = [
    Geometry.LatLongPoint(0.0, rad2deg(5 * pi / 6) - 180.0),
    Geometry.LatLongPoint(0.0, rad2deg(7 * pi / 6) - 180.0),
] # center of bells
const test_name = get(ARGS, 1, "cosine_bells") # default test case to run
const cosine_test_name = "cosine_bells"
const gaussian_test_name = "gaussian_bells"
const cylinder_test_name = "cylinders"
const lim_flag = true

# Plot variables and auxiliary function
ENV["GKSwstype"] = "nul"
import ClimaCorePlots, Plots
Plots.GRBackend()
dirname = "cg_sphere_advection_limiter_$(test_name)"

if lim_flag == false
    dirname = "$(dirname)_no_lim"
end
if D₄ == 0
    dirname = "$(dirname)_D0"
end

path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

function conservation_error(sol)
    initial_total_mass = sum(sol.u[1].ρ)
    initial_tracer_mass = sum(sol.u[1].ρq)
    final_total_mass = sum(sol.u[end].ρ)
    final_tracer_mass = sum(sol.u[end].ρq)
    return (
        (final_total_mass - initial_total_mass) / initial_total_mass,
        (final_tracer_mass - initial_tracer_mass) / initial_tracer_mass,
    )
end

# Set up spatial discretization
FT = Float64
ne_seq = (6, 16)
Δh = zeros(FT, length(ne_seq))
L1err, L2err, Linferr, relative_errors = zeros(FT, length(ne_seq)),
zeros(FT, length(ne_seq)),
zeros(FT, length(ne_seq)),
zeros(FT, length(ne_seq))
Nq = 4

# h-refinement study
for (k, ne) in enumerate(ne_seq)
    # Set up space
    domain = Domains.SphereDomain(R)
    mesh = Meshes.EquiangularCubedSphere(domain, ne)
    grid_topology = Topologies.Topology2D(context, mesh)
    quad = Spaces.Quadratures.GLL{Nq}()
    space =
        Spaces.SpectralElementSpace2D(grid_topology, quad; enable_bubble = true)

    # Initialize variables needed for limiters
    n_elems = Topologies.nlocalelems(space.topology)
    min_q = zeros(n_elems)
    max_q = zeros(n_elems)

    coords = Fields.coordinate_field(space)
    Δh[k] = 2 * R / ne
    global_geom = space.global_geometry

    # Initialize state
    y0 = map(coords) do coord

        ϕ = coord.lat
        λ = coord.long
        rd = Vector{Float64}(undef, 2)

        for i in 1:2
            rd[i] = Geometry.great_circle_distance(coord, centers[i], global_geom)
        end

        # Initialize specific tracer concentration
        if test_name == cylinder_test_name
            if rd[1] <= r0 && abs(λ - centers[1].long) * R >= rad2deg(r0 / 6)
                q = 1.0
            elseif rd[2] <= r0 &&
                   abs(λ - centers[2].long) * R >= rad2deg(r0 / 6)
                q = 1.0
            elseif rd[1] <= r0 &&
                   abs(λ - centers[1].long) * R < rad2deg(r0 / 6) &&
                   (ϕ - centers[1].lat) * R < rad2deg(-5 * r0 / 12)
                q = 1.0
            elseif rd[2] <= r0 &&
                   abs(λ - centers[2].long) * R < rad2deg(r0 / 6) &&
                   (ϕ - centers[2].lat) * R > rad2deg(5 * r0 / 12)
                q = 1.0
            else
                q = 0.0
            end
        elseif test_name == gaussian_test_name
            q = 0.95 * (exp(-(rd[1] / r0)^2) + exp(-(rd[2] / r0)^2))
        else # default test case, cosine bells
            if rd[1] < r0
                q = 0.0 + 1.0 * (1 / 2) * (1 + cospi(rd[1] / r0))
            elseif rd[2] < r0
                q = 0.0 + 1.0 * (1 / 2) * (1 + cospi(rd[2] / r0))
            else
                q = 0.0
            end
        end

        # Initialize air density
        ρ = ρ₀

        # Tracer density
        Q = ρ * q
        return (ρ = ρ, ρq = Q)
    end

    function f!(ystar, y, parameters, t)

        grad = Operators.Gradient()
        wdiv = Operators.WeakDivergence()
        T = parameters.T

        coords = Fields.coordinate_field(axes(y.ρq))
        u = map(coords) do coord
            ϕ = coord.lat
            λ = coord.long

            uu =
                u0 * sind(λ)^2 * sind(2 * ϕ) * cospi(t / T) +
                360.0 * cosd(ϕ) / T
            uv = u0 * sind(2 * λ) * cosd(ϕ) * cospi(t / T)
            Geometry.UVVector(uu, uv)
        end

        Limiters.compute_bounds!(parameters.limiter, y.ρq, y.ρ)

        # Compute hyperviscosity for the tracer equation by splitting it in two diffusion calls
        @. ystar.ρq = wdiv(grad(y.ρq / y.ρ))
        Spaces.weighted_dss!(ystar)
        @. ystar.ρq = -D₄ * wdiv(y.ρ * grad(ystar.ρq))

        # Add advective flux divergence
        @. ystar.ρ = -wdiv(y.ρ * u)         # contintuity equation
        @. ystar.ρq += -wdiv(y.ρq * u)      # adevtion of tracers equation
    end

    function stage_callback!(ydoublestar, integrator, parameters, t)
        if lim_flag
            Limiters.apply_limiter!(
                ydoublestar.ρq,
                ydoublestar.ρ,
                parameters.limiter,
            )
        end
        Spaces.weighted_dss!(ydoublestar)
    end

    # Set up RHS function
    ystar = similar(y0)
    parameters =
        (space = space, limiter = Limiters.QuasiMonotoneLimiter(y0.ρq), T = T)
    f!(ystar, y0, parameters, 0.0)

    # Solve the ODE
    end_time = T
    prob = ODEProblem(f!, y0, (0.0, end_time), parameters)
    sol = solve(
        prob,
        SSPRK33(stage_callback!),
        dt = dt,
        saveat = 10 * dt,
        progress = true,
        adaptive = false,
        progress_message = (dt, u, p, t) -> t,
    )
    L1err[k] = norm(sol.u[end].ρq ./ sol.u[end].ρ .- y0.ρq ./ y0.ρ, 1) / norm(y0.ρq ./ y0.ρ, 1)
    L2err[k] = norm(sol.u[end].ρq ./ sol.u[end].ρ .- y0.ρq ./ y0.ρ) / norm(y0.ρq ./ y0.ρ)
    Linferr[k] = norm(sol.u[end].ρq ./ sol.u[end].ρ .- y0.ρq ./ y0.ρ, Inf) / norm(y0.ρq ./ y0.ρ, Inf)

    @info "Test case: $(test_name)"
    @info "With limiter: $(lim_flag)"
    @info "Hyperdiffusion coefficient: D₄ = $(D₄)"
    @info "Number of elements per cube panel: $(ne) x $(ne)"
    @info "Number of quadrature points per element: $(Nq) x $(Nq) (p = $(Nq-1))"
    @info "Time step dt = $(dt) (s)"
    @info "Tracer concentration norm at t = 0 (s): ", norm(y0.ρq ./ y0.ρ)
    @info "Tracer concentration norm at $(n_steps) time steps, t = $(end_time) (s): ",
    norm(sol.u[end].ρq ./ sol.u[end].ρ)
    @info "Tracer concentration extrema at at $(n_steps) time steps, t = $(end_time) (s): ",
    extrema(sol.u[end].ρq ./ sol.u[end].ρ)
    @info "L₁ error at $(n_steps) time steps, t = $(end_time) (s): ", L1err[k]
    @info "L₂ error at $(n_steps) time steps, t = $(end_time) (s): ", L2err[k]
    @info "L∞ error at $(n_steps) time steps, t = $(end_time) (s): ", Linferr[k]

    Plots.png(
        Plots.plot(sol.u[end].ρq ./ sol.u[end].ρ),
        joinpath(path, "final_q.png"),
    )

    # Conservation errors
    lim_ρ_err, lim_ρq_err = conservation_error(sol)
    relative_errors[k] = abs(lim_ρ_err - lim_ρq_err)
end

# Check conservation
#atols = [2.5e1eps(FT), 2.5e1eps(FT), 8.5eps(FT)]
atols = [2.5e1eps(FT), 2.5e1eps(FT), 1.0e1eps(FT)]
for k in 1:length(ne_seq)
    @test relative_errors[k] ≈ FT(0) atol = atols[k]
end

# Print convergence rate info
conv = convergence_rate(L2err, Δh)
@info "Converge rates for this test case are: ", conv

# Plot the errors
# L₁ error Vs number of elements
Plots.png(
    Plots.plot(
        collect(ne_seq),
        L1err,
        yscale = :log10,
        xlabel = "Nₑ",
        ylabel = "log₁₀(L₁ err)",
        label = "",
    ),
    joinpath(path, "L1error.png"),
)
linkfig(
    relpath(joinpath(path, "L1error.png"), joinpath(@__DIR__, "../..")),
    "L₁ error Vs Nₑ",
)

# L₂ error Vs number of elements
Plots.png(
    Plots.plot(
        collect(ne_seq),
        L2err,
        yscale = :log10,
        xlabel = "Nₑ",
        ylabel = "log₁₀(L₂ err)",
        label = "",
    ),
    joinpath(path, "L2error.png"),
)
linkfig(
    relpath(joinpath(path, "L2error.png"), joinpath(@__DIR__, "../..")),
    "L₂ error Vs Nₑ",
)

# L∞ error Vs number of elements
Plots.png(
    Plots.plot(
        collect(ne_seq),
        Linferr,
        yscale = :log10,
        xlabel = "Nₑ",
        ylabel = "log₁₀(L∞ err)",
        label = "",
    ),
    joinpath(path, "Linferror.png"),
)
linkfig(
    relpath(joinpath(path, "Linferror.png"), joinpath(@__DIR__, "../..")),
    "L∞ error Vs Nₑ",
)
