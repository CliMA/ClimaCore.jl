push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using ClimaCore.Geometry, LinearAlgebra, UnPack
import ClimaCore: slab, Fields, Domains, Topologies, Meshes, Spaces
import ClimaCore: slab
import ClimaCore.Operators
import ClimaCore.Geometry
using LinearAlgebra, IntervalSets
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33
using QuadGK: quadgk

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

# This example solves the shallow-water equations on a cubed-sphere manifold.
# This file contains five test cases:
# - One, called "steady_state", reproduces Test Case 2 in Williamson et al,
#   "A standard test set for numerical approximations to the shallow water
#   equations in spherical geometry", 1992. This test case gives the steady-state
#   solution to the non-linear shallow water equations. It consists of solid
#   body rotation or zonal flow with the corresponding geostrophic height field.
#   This can be run with an angle α that represents the angle between the north
#   pole and the center of the top cube panel.
# - A second one, called "steady_state_compact", reproduces Test Case 3 in the same
#   reference paper. This test case gives the steady-state solution to the
#   non-linear shallow water equations with nonlinear zonal geostrophic flow
#   with compact support.
# - A third one, called "mountain", reproduces Test Case 5 in the same
#   reference paper. It represents a zonal flow over an isolated mountain,
#   where the governing equations describe a global steady-state nonlinear
#   zonal geostrophic flow, with a corresponding geostrophic height field over
#   a non-uniform reference surface h_s.
# - A fourth one, called "rossby_haurwitz", reproduces Test Case 6 in the same
#   reference paper. It represents the solution of the nonlinear barotropic
#   vorticity equation on the sphere
# - A fifth one, called "barotropic_instability", reproduces the test case in
#   Galewsky et al, "An initial-value problem for testing numerical models of
#   the global shallow-water equations", 2004 (also in Sec. 7.6 of Ullirch et al,
#   "High-order ﬁnite-volume methods for the shallow-water equations on
#   the sphere", 2010). This test case consists of a zonal jet with compact
#   support at a latitude of 45°. A small height disturbance is then added,
#   which causes the jet to become unstable and collapse into a highly vortical
#   structure.

# Physical parameters needed
const R = 6.37122e6
const Ω = 7.292e-5
const g = 9.80616
const D₄ = 1.0e16 # hyperdiffusion coefficient

# Test case specifications
const test_name = get(ARGS, 1, "steady_state") # default test case to run
const test_angle_name = get(ARGS, 2, "alpha0") # default test case to run
const steady_state_test_name = "steady_state"
const steady_state_compact_test_name = "steady_state_compact"
const mountain_test_name = "mountain"
const rossby_haurwitz_test_name = "rossby_haurwitz"
const barotropic_instability_test_name = "barotropic_instability"
const alpha0_test_name = "alpha0"
const alpha30_test_name = "alpha30"
const alpha45_test_name = "alpha45"
const alpha60_test_name = "alpha60"

# Test-specific physical parameters
if test_angle_name == alpha30_test_name
    const α = 30.0
elseif test_angle_name == alpha45_test_name
    const α = 45.0
elseif test_angle_name == alpha60_test_name
    const α = 60.0
else # default test case, α = 0.0
    const α = 0.0
end

if test_name == mountain_test_name
    const u0 = 20.0
    const h0 = 5960
    const a = 20.0 # radius of conical mountain
    const λc = 90.0 # center of mountain long coord, shifted by 180 compared to the paper, because our λ ∈ [-180, 180] (in the paper it was 270, with λ ∈ [0, 360])
    const ϕc = 30.0 # center of mountain lat coord
    const h_s0 = 2e3
elseif test_name == rossby_haurwitz_test_name
    const a = 4.0
    const h0 = 8.0e3
    const ω = 7.848e-6
    const K = 7.848e-6
elseif test_name == steady_state_compact_test_name
    const u0 = 2 * pi * R / (12 * 86400)
    const h0 = 2.94e4 / g
    const ϕᵦ = -30.0
    const ϕₑ = 90.0
    const xₑ = 0.3
elseif test_name == barotropic_instability_test_name
    const u_max = 80.0
    const xₑ = 0.3
    const αₚ = 19.09859
    const βₚ = 3.81971
    const h0 = 10158.18617 # value for initial height from Tempest https://github.com/paullric/tempestmodel/blob/master/test/shallowwater_sphere/BarotropicInstabilityTest.cpp#L86
    const h_hat = 120.0
    const ϕ₀ = 25.71428
    const ϕ₁ = 64.28571
    const ϕ₂ = 45.0
    const eₙ = exp(-4.0 / (deg2rad(ϕ₁) - deg2rad(ϕ₀))^2)
else # default case, steady-state test case
    const u0 = 2 * pi * R / (12 * 86400)
    const h0 = 2.94e4 / g
end

# Plot variables and auxiliary function
ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()
dirname = "cg_sphere_shallowwater_$(test_name)"
dirname = "$(dirname)_$(test_angle_name)"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

function linkfig(figpath, alt = "")
    # Buildkite-agent upload figpath
    # Link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

# Set up discretization
ne = 9 # the rossby_haurwitz test case's initial state has a singularity at the pole. We avoid it by using odd number of elements
Nq = 4

domain = Domains.SphereDomain(R)
mesh = Meshes.EquiangularCubedSphere(domain, ne)
grid_topology = Topologies.Topology2D(mesh)
quad = Spaces.Quadratures.GLL{Nq}()
space = Spaces.SpectralElementSpace2D(grid_topology, quad)

coords = Fields.coordinate_field(space)

# Definition of Coriolis parameter
if test_name == rossby_haurwitz_test_name
    f = map(Fields.local_geometry_field(space)) do local_geometry
        coord = local_geometry.coordinates

        ϕ = coord.lat
        λ = coord.long

        f = 2 * Ω * sind(ϕ)

        # Technically this should be a WVector, but since we are only in a 2D space,
        # WVector, Contravariant3Vector, Covariant3Vector are all equivalent.
        # This _won't_ be true in 3D however!
        Geometry.Contravariant3Vector(f)
    end
else # all other test cases share the same Coriolis parameter
    f = map(Fields.local_geometry_field(space)) do local_geometry
        coord = local_geometry.coordinates
        ϕ = coord.lat
        λ = coord.long

        f = 2 * Ω * (-cosd(λ) * cosd(ϕ) * sind(α) + sind(ϕ) * cosd(α))

        # Technically this should be a WVector, but since we are only in a 2D space,
        # WVector, Contravariant3Vector, Covariant3Vector are all equivalent.
        # This _won't_ be true in 3D however!
        Geometry.Contravariant3Vector(f)
    end
end

# Definition of bottom surface topography field
if test_name == mountain_test_name # define the non-uniform reference surface h_s
    h_s = map(Fields.coordinate_field(space)) do coord
        ϕ = coord.lat
        λ = coord.long
        r = sqrt(min(a^2, (λ - λc)^2 + (ϕ - ϕc)^2)) # positive branch
        h_s = h_s0 * (1 - r / a)
    end
else
    h_s = zeros(space)
end

# Set initial condition
if test_name == rossby_haurwitz_test_name
    y0 = map(Fields.local_geometry_field(space)) do local_geometry
        coord = local_geometry.coordinates
        ϕ = coord.lat
        λ = coord.long

        A =
            ω / 2 * (2 * Ω + ω) * cosd(ϕ)^2 +
            1 / 4 *
            K^2 *
            cosd(ϕ)^(2 * a) *
            ((a + 1) * cosd(ϕ)^2 + (2 * a^2 - a - 2) - 2 * a^2 * cosd(ϕ)^-2)
        B =
            2 * (Ω + ω) * K / (a + 1) / (a + 2) *
            cosd(ϕ)^a *
            ((a^2 + 2 * a + 2) - (a + 1)^2 * cosd(ϕ)^2)
        C = 1 / 4 * K^2 * cosd(ϕ)^(2 * a) * ((a + 1) * cosd(ϕ)^2 - (a + 2))

        h =
            h0 +
            (R^2 * A + R^2 * B * cosd(a * λ) + R^2 * C * cosd(2 * a * λ)) / g

        uλ =
            R * ω * cosd(ϕ) +
            R * K * cosd(ϕ)^(a - 1) * (a * sind(ϕ)^2 - cosd(ϕ)^2) * cosd(a * λ)
        uϕ = -R * K * a * cosd(ϕ)^(a - 1) * sind(ϕ) * sind(a * λ)


        u = Geometry.transform(
            Geometry.Covariant12Axis(),
            Geometry.UVVector(uλ, uϕ),
            local_geometry,
        )
        return (h = h, u = u)
    end
elseif test_name == steady_state_compact_test_name
    y0 = map(Fields.local_geometry_field(space)) do local_geometry
        coord = local_geometry.coordinates

        ϕ = coord.lat
        λ = coord.long

        if α == 0.0
            ϕprime = ϕ
            λprime = λ
        else
            ϕprime = asind(sind(ϕ) * cosd(α) - cosd(ϕ) * cosd(λ) * sind(α))
            λprime = asind(sind(λ) * cosd(ϕ) / cosd(ϕprime)) # for alpha45, this experiences numerical precision issues. The test case is designed for either alpha0 or alpha60

            # Temporary angle to ensure λprime is in the right quadrant
            λcond = cosd(α) * cosd(λ) * cosd(ϕ) + sind(α) * sind(ϕ)

            # If λprime is not in the right quadrant, adjust
            if λcond < 0.0
                λprime = -λprime - 180.0 # shifted by 180 compared to the paper, because our λ ∈ [-180, 180]
            end
            if λprime < -180.0
                λprime += 360.0
            end
        end

        # Set auxiliary function needed for initial state of velocity field
        b(x) = x ≤ 0.0 ? 0.0 : exp(-x^(-1))

        x(ϕprime) = xₑ * (ϕprime - ϕᵦ) / (ϕₑ - ϕᵦ)
        uλprime(ϕprime) =
            u0 * b(x(ϕprime)) * b(xₑ - x(ϕprime)) * exp(4.0 / xₑ)
        uϕprime = 0.0

        # Set integral needed for height initial state
        h_int(γ) =
            abs(γ) < 90.0 ?
            (2 * Ω * sind(γ) + uλprime(γ) * tand(γ) / R) * uλprime(γ) : 0.0

        # Set initial state for height field
        h = h0 - (R / g) * (pi / 180.0) * quadgk(h_int, -90.0, ϕprime)[1]

        # Set initial state for velocity field
        uϕ = -(uλprime(ϕprime) * sind(α) * sind(λprime)) / cosd(ϕ)
        if abs(cosd(λ)) < 1e-13
            if abs(α) > 1e-13
                if cosd(λ) > 0.0
                    uλ = -uϕ * cosd(ϕ) / tand(α)
                else
                    uλ = uϕ * cosd(ϕ) / tand(α)
                end
            else
                uλ = uλprime(ϕprime)
            end
        else
            uλ =
                (uϕ * sind(ϕ) * sind(λ) + uλprime(ϕprime) * cosd(λprime)) /
                cosd(λ)
        end

        u = Geometry.transform(
            Geometry.Covariant12Axis(),
            Geometry.UVVector(uλ, uϕ),
            local_geometry,
        )

        return (h = h, u = u)
    end
elseif test_name == barotropic_instability_test_name
    y0 = map(Fields.local_geometry_field(space)) do local_geometry
        coord = local_geometry.coordinates

        ϕ = coord.lat
        λ = coord.long

        if α == 0.0
            ϕprime = ϕ
        else
            ϕprime = asind(sind(ϕ) * cosd(α) - cosd(ϕ) * cosd(λ) * sind(α))
        end

        # Set initial state of velocity field
        uλprime(ϕprime) =
            (u_max / eₙ) *
            exp(1.0 / (deg2rad(ϕprime - ϕ₀) * deg2rad(ϕprime - ϕ₁))) *
            (ϕ₀ < ϕprime < ϕ₁)
        uϕprime = 0.0

        # Set integral needed for height initial state
        h_int(γ) =
            abs(γ) < 90.0 ?
            (2 * Ω * sind(γ) + uλprime(γ) * tand(γ) / R) * uλprime(γ) : 0.0

        # Set initial state for height field
        h = h0 - (R / g) * (pi / 180.0) * quadgk(h_int, -90.0, ϕprime)[1]

        if λ > 0.0
            λ -= 360.0
        end
        if λ < -360.0 || λ > 0.0
            @info "Invalid longitude value"
        end

        # Add height perturbation
        h += h_hat * cosd(ϕ) * exp(-(λ^2 / αₚ^2) - ((ϕ₂ - ϕ)^2 / βₚ^2))

        uλ = uλprime(ϕprime)
        uϕ = uϕprime

        u = Geometry.transform(
            Geometry.Covariant12Axis(),
            Geometry.UVVector(uλ, uϕ),
            local_geometry,
        )

        return (h = h, u = u)
    end
else # steady-state and mountain test cases share the same form of fields
    y0 = map(Fields.local_geometry_field(space)) do local_geometry
        coord = local_geometry.coordinates

        ϕ = coord.lat
        λ = coord.long
        h =
            h0 -
            (R * Ω * u0 + u0^2 / 2) / g *
            (-cosd(λ) * cosd(ϕ) * sind(α) + sind(ϕ) * cosd(α))^2
        uλ = u0 * (cosd(α) * cosd(ϕ) + sind(α) * cosd(λ) * sind(ϕ))
        uϕ = -u0 * sind(α) * sind(λ)

        u = Geometry.transform(
            Geometry.Covariant12Axis(),
            Geometry.UVVector(uλ, uϕ),
            local_geometry,
        )

        return (h = h, u = u)
    end
end

function rhs!(dydt, y, parameters, t)
    f = parameters.f
    h_s = parameters.h_s

    div = Operators.Divergence()
    wdiv = Operators.WeakDivergence()
    grad = Operators.Gradient()
    wgrad = Operators.WeakGradient()
    curl = Operators.Curl()
    wcurl = Operators.WeakCurl()

    # Compute hyperviscosity first
    @. dydt.h = wdiv(grad(y.h))
    @. dydt.u = wgrad(div(y.u)) - Geometry.Covariant12Vector(wcurl(curl(y.u)))

    Spaces.weighted_dss!(dydt)

    @. dydt.h = -D₄ * wdiv(grad(dydt.h))
    @. dydt.u =
        -D₄ *
        (wgrad(div(dydt.u)) - Geometry.Covariant12Vector(wcurl(curl(dydt.u))))

    # Add in pieces
    @. begin
        dydt.h += -wdiv(y.h * y.u)
        dydt.u +=
            -grad(g * (y.h + h_s) + norm(y.u)^2 / 2) + y.u × (f + curl(y.u))
    end
    Spaces.weighted_dss!(dydt)
    return dydt
end

# Set up RHS function
dydt = similar(y0)
rhs!(dydt, y0, (f = f, h_s = h_s), 0.0)

# Solve the ODE
dt = 9 * 60
T = 86400 * 2

prob = ODEProblem(rhs!, y0, (0.0, T), (f = f, h_s = h_s))
sol = solve(
    prob,
    SSPRK33(),
    dt = dt,
    saveat = dt,
    progress = true,
    adaptive = false,
    progress_message = (dt, u, p, t) -> t,
)

@info "Test case: $(test_name)"
@info "  with α: $(test_angle_name)"
@info "Solution L₂ norm at time t = 0: ", norm(y0.h)
@info "Solution L₂ norm at time t = $(T): ", norm(sol.u[end].h)
@info "Fluid volume at time t = 0: ", sum(y0.h)
@info "Fluid volume at time t = $(T): ", sum(sol.u[end].h)

if test_name == steady_state_test_name ||
   test_name == steady_state_compact_test_name
    # In these cases, we use the IC as the reference exact solution
    @info "L₁ error at T = $(T): ", norm(sol.u[end].h .- y0.h, 1)
    @info "L₂ error at T = $(T): ", norm(sol.u[end].h .- y0.h)
    @info "L∞ error at T = $(T): ", norm(sol.u[end].h .- y0.h, Inf)
    # Pointwise final L₂ error
    Plots.png(
        Plots.plot(sol.u[end].h .- y0.h),
        joinpath(path, "final_height_L2_error.png"),
    )
    linkfig(
        relpath(
            joinpath(path, "final_height_L2_error.png"),
            joinpath(@__DIR__, "../.."),
        ),
        "Absolute error in height",
    )
    # Height errors over time
    relL1err = Array{Float64}(undef, div(T, dt))
    for t in 1:div(T, dt)
        relL1err[t] = norm(sol.u[t].h .- y0.h, 1) / norm(y0.h, 1)
    end
    Plots.png(
        Plots.plot(
            [1:dt:T],
            relL1err,
            xlabel = "time [s]",
            ylabel = "Relative L₁ err",
            label = "",
        ),
        joinpath(path, "HeightRelL1errorVstime.png"),
    )
    linkfig(
        relpath(
            joinpath(path, "HeightRelL1errorVstime.png"),
            joinpath(@__DIR__, "../.."),
        ),
        "Height relative L1 error over time",
    )

    relL2err = Array{Float64}(undef, div(T, dt))
    for t in 1:div(T, dt)
        relL2err[t] = norm(sol.u[t].h .- y0.h) / norm(y0.h)
    end
    Plots.png(
        Plots.plot(
            [1:dt:T],
            relL2err,
            xlabel = "time [s]",
            ylabel = "Relative L₂ err",
            label = "",
        ),
        joinpath(path, "HeightRelL2errorVstime.png"),
    )
    linkfig(
        relpath(
            joinpath(path, "HeightRelL2errorVstime.png"),
            joinpath(@__DIR__, "../.."),
        ),
        "Height relative L2 error over time",
    )

    RelLInferr = Array{Float64}(undef, div(T, dt))
    for t in 1:div(T, dt)
        RelLInferr[t] = norm(sol.u[t].h .- y0.h, Inf) / norm(y0.h, Inf)
    end
    Plots.png(
        Plots.plot(
            [1:dt:T],
            RelLInferr,
            xlabel = "time [s]",
            ylabel = "Relative L∞ err",
            label = "",
        ),
        joinpath(path, "HeightRelL1InferrorVstime.png"),
    )
    linkfig(
        relpath(
            joinpath(path, "HeightRelLInferrorVstime.png"),
            joinpath(@__DIR__, "../.."),
        ),
        "Height relative L_Inf error over time",
    )
else # In the non steady-state cases, we only plot the latest output of the dynamic problem
    Plots.png(Plots.plot(sol.u[end].h), joinpath(path, "final_height.png"))
    linkfig(
        relpath(
            joinpath(path, "final_height.png"),
            joinpath(@__DIR__, "../.."),
        ),
        "Height field at the final time step",
    )
end
