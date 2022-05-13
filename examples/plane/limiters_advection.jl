using LinearAlgebra

import ClimaCore:
    Domains, Fields, Geometry, Meshes, Operators, Spaces, Topologies, Limiters

using OrdinaryDiffEq: ODEProblem, solve
using DiffEqBase
using ClimaTimeSteppers

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

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

# Advection problem on a 2D Cartesian domain with bounds-preserving quasimonotone limiter.
# The initial condition can be set via a command line argument.
# Possible test cases are: cosine_bells (default), gaussian_bells, and cylinders

FT = Float64

# Set up physical parameters
const xmin = -2π              # domain x lower bound
const xmax = 2π               # domain x upper bound
const ymin = -2π              # domain y lower bound
const ymax = 2π               # domain y upper bound
const ρ₀ = 1.0                # air density
const D₄ = 0.0                # hyperdiffusion coefficient
const u0 = π / 2              # angular velocity
const r0 = (xmax - xmin) / 6  # bells radius
const end_time = 2π           # simulation period in seconds
const dt = end_time / 8000
const n_steps = Int(round(end_time / dt))
const flow_center =
    Geometry.XYPoint(xmin + (xmax - xmin) / 2, ymin + (ymax - ymin) / 2)
const bell_centers = [
    Geometry.XYPoint(xmin + (xmax - xmin) / 4, ymin + (ymax - ymin) / 2),
    Geometry.XYPoint(xmin + 3 * (xmax - xmin) / 4, ymin + (ymax - ymin) / 2),
]

# Set up test parameters
const test_name = get(ARGS, 1, "cosine_bells") # default test case to run
const cosine_test_name = "cosine_bells"
const gaussian_test_name = "gaussian_bells"
const cylinder_test_name = "cylinders"
const lim_flag = true
const limiter_tol = 5e-14

# Plot variables and auxiliary function
ENV["GKSwstype"] = "nul"
import ClimaCorePlots, Plots
Plots.GRBackend()
dirname = "plane_advection_limiter_$(test_name)"

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

# Set up spatial domain
domain = Domains.RectangleDomain(
    Domains.IntervalDomain(
        Geometry.XPoint(xmin),
        Geometry.XPoint(xmax),
        periodic = true,
    ),
    Domains.IntervalDomain(
        Geometry.YPoint(ymin),
        Geometry.YPoint(ymax),
        periodic = true,
    ),
)

# Set up spatial discretization
ne_seq = 2 .^ (2, 3, 4)
Δh = zeros(FT, length(ne_seq))
L1err, L2err, Linferr = zeros(FT, length(ne_seq)),
zeros(FT, length(ne_seq)),
zeros(FT, length(ne_seq))
Nq = 4

# h-refinement study loop
for (k, ne) in enumerate(ne_seq)
    mesh = Meshes.RectilinearMesh(domain, ne, ne)
    grid_topology = Topologies.Topology2D(mesh)
    quad = Spaces.Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    # Initialize variables needed for limiters
    n_elems = Topologies.nlocalelems(space.topology)
    min_q = zeros(n_elems)
    max_q = zeros(n_elems)

    coords = Fields.coordinate_field(space)
    Δh[k] = (xmax - xmin) / ne

    # Initialize state
    y0 = map(coords) do coord
        x, y = coord.x, coord.y

        rd = Vector{Float64}(undef, 2)
        for i in 1:2
            rd[i] = Geometry.euclidean_distance(coord, bell_centers[i])
        end

        # Initialize specific tracer concentration
        if test_name == cylinder_test_name
            if rd[1] <= r0 && abs(x - bell_centers[1].x) >= r0 / 6
                q = 1.0
            elseif rd[2] <= r0 && abs(x - bell_centers[2].x) >= r0 / 6
                q = 1.0
            elseif rd[1] <= r0 &&
                   abs(x - bell_centers[1].x) < r0 / 6 &&
                   (y - bell_centers[1].y) < -5 * r0 / 12
                q = 1.0
            elseif rd[2] <= r0 &&
                   abs(x - bell_centers[2].x) < r0 / 6 &&
                   (y - bell_centers[2].y) > 5 * r0 / 12
                q = 1.0
            else
                q = 0.1
            end
        elseif test_name == gaussian_test_name
            q = 0.95 * (exp(-5.0 * (rd[1] / r0)^2) + exp(-5.0 * (rd[2] / r0)^2))
        else # default test case, cosine bells
            if rd[1] < r0
                q = 0.1 + 0.9 * (1 / 2) * (1 + cospi(rd[1] / r0))
            elseif rd[2] < r0
                q = 0.1 + 0.9 * (1 / 2) * (1 + cospi(rd[2] / r0))
            else
                q = 0.1
            end
        end

        # Initialize air density
        ρ = ρ₀

        # Tracer density
        Q = ρ * q
        return (ρ = ρ, ρq = Q)
    end

    y0 = Fields.FieldVector(ρ = y0.ρ, ρq = y0.ρq)

    function f!(dy, y, parameters, t, alpha, beta)

        # Set up operators
        grad = Operators.Gradient()
        wdiv = Operators.WeakDivergence()
        end_time = parameters.end_time

        # Define the flow
        coords = Fields.coordinate_field(axes(y.ρq))
        u = map(coords) do coord
            local y
            x, y = coord.x, coord.y

            uu = -u0 * (y - flow_center.y) * cospi(t / end_time)
            uv = u0 * (x - flow_center.x) * cospi(t / end_time)

            Geometry.UVVector(uu, uv)
        end

        # Compute min_q[] and max_q[] that will be needed later in the stage limiter
        space = parameters.space
        n_elems = Topologies.nlocalelems(space)
        topology = space.topology

        neigh_elems_q_min = Array{Float64}(undef, 8)
        neigh_elems_q_max = Array{Float64}(undef, 8)

        for e in 1:n_elems
            q_e = Fields.slab(y.ρq, e) ./ Fields.slab(y.ρ, e)

            q_e_min = minimum(q_e)
            q_e_max = maximum(q_e)
            neigh_elems = Topologies.local_neighboring_elements(topology, e)
            for i in 1:length(neigh_elems)
                if neigh_elems[i] == 0
                    neigh_elems_q_min[i] = +Inf
                    neigh_elems_q_max[i] = -Inf
                else
                    neigh_elems_q_min[i] = Fields.minimum(
                        Fields.slab(y.ρq, neigh_elems[i]) ./
                        Fields.slab(y.ρ, neigh_elems[i]),
                    )
                    neigh_elems_q_max[i] = Fields.maximum(
                        Fields.slab(y.ρq, neigh_elems[i]) ./
                        Fields.slab(y.ρ, neigh_elems[i]),
                    )
                end
            end
            parameters.min_q[e] = min(minimum(neigh_elems_q_min), q_e_min)
            parameters.max_q[e] = max(maximum(neigh_elems_q_max), q_e_max)
        end

        # Compute hyperviscosity for the tracer equation by splitting it in two diffusion calls
        ystar = similar(y)
        @. ystar.ρq = wdiv(grad(y.ρq / y.ρ))
        Spaces.weighted_dss!(ystar.ρq)
        @. ystar.ρq = -D₄ * wdiv(y.ρ * grad(ystar.ρq))

        # Add advective flux divergence
        @. dy.ρ = beta * dy.ρ - alpha * wdiv(y.ρ * u)                         # contintuity equation
        @. dy.ρq = beta * dy.ρq - alpha * wdiv(y.ρq * u) + alpha * ystar.ρq   # advection of tracers equation

        min_q = parameters.min_q
        max_q = parameters.max_q

        if lim_flag
            # Call quasimonotone limiter, to find optimal ρq (where ρq gets updated in place)
            Limiters.quasimonotone_limiter!(
                dy.ρq,
                dy.ρ,
                min_q,
                max_q,
                rtol = limiter_tol,
            )
        end
        Spaces.weighted_dss!(dy.ρ)
        Spaces.weighted_dss!(dy.ρq)
    end

    # Set up RHS function
    ystar = copy(y0)
    parameters =
        (space = space, min_q = min_q, max_q = max_q, end_time = end_time)
    f!(ystar, y0, parameters, 0.0, dt, 1)

    # Solve the ODE
    prob = ODEProblem(
        IncrementingODEFunction(f!),
        copy(y0),
        (0.0, end_time),
        parameters,
    )
    sol = solve(
        prob,
        SSPRK33ShuOsher(),
        dt = dt,
        saveat = 0.99 * 800 * dt,
        progress = true,
        adaptive = false,
        progress_message = (dt, u, p, t) -> t,
    )

    L1err[k] = norm(
        (sol.u[end].ρq ./ sol.u[end].ρ .- y0.ρq ./ y0.ρ) ./ (y0.ρq ./ y0.ρ),
        1,
    )
    L2err[k] = norm(
        (sol.u[end].ρq ./ sol.u[end].ρ .- y0.ρq ./ y0.ρ) ./ (y0.ρq ./ y0.ρ),
    )
    Linferr[k] = norm(
        (sol.u[end].ρq ./ sol.u[end].ρ .- y0.ρq ./ y0.ρ) ./ (y0.ρq ./ y0.ρ),
        Inf,
    )

    @info "Test case: $(test_name)"
    @info "With limiter: $(lim_flag)"
    @info "Hyperdiffusion coefficient: D₄ = $(D₄)"
    @info "Number of elements in domain: $(ne) x $(ne)"
    @info "Number of quadrature points per element: $(Nq) x $(Nq) (p = $(Nq-1))"
    @info "Time step dt = $(dt) (s)"
    @info "Tracer concentration norm at t = 0 (s): ", norm(y0.ρq ./ y0.ρ)
    @info "Tracer concentration norm at $(n_steps) time steps, t = $(end_time) (s): ",
    norm(sol.u[end].ρq ./ sol.u[end].ρ)
    @info "L₁ error at $(n_steps) time steps, t = $(end_time) (s): ", L1err[k]
    @info "L₂ error at $(n_steps) time steps, t = $(end_time) (s): ", L2err[k]
    @info "L∞ error at $(n_steps) time steps, t = $(end_time) (s): ", Linferr[k]

    Plots.png(
        Plots.plot(sol.u[end].ρq ./ sol.u[end].ρ),
        joinpath(path, "final_q.png"),
    )
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
