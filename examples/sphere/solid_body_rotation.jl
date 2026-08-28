# Solid-body rotation of a tracer bell around the cubed sphere: the simplest
# horizontal transport test, and the one that exposes how the cubed-sphere panel
# edges affect accuracy. Run over a sequence of resolutions to give a
# convergence study. The first command-line argument selects the initial
# condition, `cosine_bell` (default) or `gaussian_bell`; the second selects the
# rotation axis, `alpha0` (along the equator) or `alpha45` (tilted over the
# panel corners).
using ClimaComms
using LinearAlgebra

import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Spaces,
    Topologies,
    Quadratures

import ClimaTimeSteppers as CTS

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

const context = ClimaComms.SingletonCommsContext()

"""
    convergence_rate(err, Δh)

Estimate convergence rate given vectors `err` and `Δh`

    err = C Δh^p+ H.O.T
    err_k ≈ C Δh_k^p
    err_k/err_m ≈ Δh_k^p/Δh_m^p
    log(err_k/err_m) ≈ log((Δh_k/Δh_m)^p)
    log(err_k/err_m) ≈ p*log(Δh_k/Δh_m)
    log(err_k/err_m)/log(Δh_k/Δh_m) ≈ p
"""
convergence_rate(err, Δh) =
    [log(err[i] / err[i - 1]) / log(Δh[i] / Δh[i - 1]) for i in 2:length(Δh)]


const R = 6.37122e6
const h0 = 1000.0
const r0 = R / 3
const u0 = 2 * pi * R / (86400 * 12)
const center = Geometry.LatLongPoint(0.0, 270.0)
const test_name = get(ARGS, 1, "cosine_bell") # default test case to run
const test_angle_name = get(ARGS, 2, "alpha0") # default test case to run
const cosine_test_name = "cosine_bell"
const gaussian_test_name = "gaussian_bell"
const alpha0_test_name = "alpha0"
const alpha45_test_name = "alpha45"

if test_angle_name == alpha45_test_name
    const α0 = 45.0
else # default test case, α0 = 0.0
    const α0 = 0.0
end

# Plot variables and auxiliary function
ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()
dir = "cg_sphere_solid_body_$(test_name)"
dir = "$(dir)_$(test_angle_name)"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

include(joinpath(@__DIR__, "..", "example_utils.jl")) # linkfig

FT = Float64
ne_seq = 2 .^ (2, 3, 4, 5)
Δh = zeros(FT, length(ne_seq))
L1err, L2err, Linferr = zeros(FT, length(ne_seq)),
zeros(FT, length(ne_seq)),
zeros(FT, length(ne_seq))
# Relative drift of the tracer mass, and the deepest undershoot below the
# initial minimum of zero, at each resolution.
mass_drift = zeros(FT, length(ne_seq))
undershoot = zeros(FT, length(ne_seq))
Nq = 4

# h-refinement study
for (k, ne) in enumerate(ne_seq)
    domain = Domains.SphereDomain(R)
    mesh = Meshes.EquiangularCubedSphere(domain, ne)
    grid_topology = Topologies.Topology2D(context, mesh)
    quad = Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    coords = Fields.coordinate_field(space)

    Δh[k] = 2 * R / ne

    global_geom = Spaces.global_geometry(space)

    h_init = map(coords) do coord
        rd = Geometry.great_circle_distance(coord, center, global_geom)

        if test_name == gaussian_test_name
            h0 * exp(-(rd / r0)^2 / 2)
        else # default test case, cosine bell
            if rd < r0
                h0 / 2 * (1 + cospi(rd / r0))
            else
                0.0
            end
        end
    end

    u = map(coords) do coord
        ϕ = coord.lat
        λ = coord.long

        uu = u0 * (cosd(α0) * cosd(ϕ) + sind(α0) * cosd(λ) * sind(ϕ))
        uv = -u0 * sind(α0) * sind(λ)
        Geometry.UVVector(uu, uv)
    end

    function rhs!(dh, h, u, t)
        div = Operators.Divergence()

        @. dh = -div(h * u) # strong form of equation
        Spaces.weighted_dss!(dh)
    end

    # Set the RHS function
    rhs!(similar(h_init), h_init, u, 0.0)

    # Solve the ODE
    T = 86400 * 12
    dt = 20 * 60
    # Integrate a copy: `h_init` is the reference for the error norms below,
    # and ClimaTimeSteppers aliases the state it is given.
    prob = CTS.ODEProblem(
        CTS.ClimaODEFunction(; T_exp! = rhs!),
        copy(h_init),
        (0.0, T),
        u,
    )
    sol = CTS.solve(
        prob,
        CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()),
        dt = dt,
        saveat = collect(0.0:dt:T),
        progress = true,
        adaptive = false,
        progress_message = (dt, u, p, t) -> t,
    )
    mass_drift[k] = abs(sum(sol.u[end]) - sum(h_init)) / sum(h_init)
    undershoot[k] = -minimum(sol.u[end]) / maximum(h_init)
    L1err[k] = norm(sol.u[end] .- h_init, 1)
    L2err[k] = norm(sol.u[end] .- h_init)
    Linferr[k] = norm(sol.u[end] .- h_init, Inf)

    @info "Test case: $(test_name) with α: $(test_angle_name)"
    @info "Number of elements per cube panel: $(ne) x $(ne)"
    @info "Solution norm at t = 0: ", norm(h_init)
    @info "Solution norm at t = $(T): ", norm(sol.u[end])
    @info "L₁ error at t = $(T): ", L1err[k]
    @info "L₂ error at t = $(T): ", L2err[k]
    @info "L∞ error at t = $(T): ", Linferr[k]
end

# Print convergence rate info
conv = convergence_rate(L2err, Δh)
@info "Converge rates for this test case are: ", conv

using Test
# The scheme must converge under refinement (measured L₁: 18.9, 7.1, 0.87,
# 0.11; rates 1.3, 3.0, 2.7) — the bell must arrive back where it started.
@test all(diff(L1err) .< 0)
@test L1err[end] < 0.5
@test all(conv .> 0.8)
# The transport is a flux divergence over a closed surface, so the tracer mass
# is conserved at every resolution (measured drift: 4e-15 and below).
@test all(mass_drift .< 1e-12)
# It carries no limiter, so the bell's edges ring and the tracer goes negative
# where the exact solution is zero. That is a resolution error, not a property
# of the scheme, so it must shrink as the bell is resolved: from 23% of the
# bell's amplitude at ne = 4 down to 0.5% at ne = 32.
@test all(diff(undershoot) .< 0)
@test undershoot[end] < 0.01

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
