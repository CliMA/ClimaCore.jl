using ClimaComms
ClimaComms.@import_required_backends
using LinearAlgebra
using Colors
using DocStringExtensions

include("../nvtx.jl")

import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Spaces,
    Quadratures,
    Topologies,
    DataLayouts

import QuadGK
import OrdinaryDiffEq
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

using Logging
using ClimaComms
import TerminalLoggers
using ClimaCorePlots
import Plots

"""
    PhysicalParameters{FT}

Physical parameters needed for the simulation.

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct PhysicalParameters{FT} # rename to PhysicalParameters
    "Radius of earth"
    R::FT = FT(6.37122e6)
    "Rotation rate of earth"
    Ω::FT = FT(7.292e-5)
    "Gravitational constant"
    g::FT = FT(9.80616)
    "Hyperdiffusion coefficient"
    ν₄::FT = FT(0.25)
end
#This example solves the shallow-water equations on a cubed-sphere manifold.
#This file contains five test cases:
abstract type AbstractTest end
"""
    SteadyStateTest{FT, P} <: AbstractTest

The first one, called "steady_state", reproduces Test Case 2 in Williamson et al,
"A standard test set for numerical approximations to the shallow water
equations in spherical geometry", 1992. This test case gives the steady-state
solution to the non-linear shallow water equations. It consists of solid
body rotation or zonal flow with the corresponding geostrophic height field.
This can be run with an angle α that represents the angle between the north
pole and the center of the top cube panel.

https://doi.org/10.1016/S0021-9991(05)80016-6

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct SteadyStateTest{FT, P} <: AbstractTest
    "Physical parameters"
    params::P = PhysicalParameters{FT}()
    "advection velocity"
    u0::FT = 2 * pi * params.R / (12 * 86400)
    "peak of analytic height field"
    h0::FT = 2.94e4 / params.g
    "angle between the north pole and the center of the top cube panel"
    α::FT
end
SteadyStateTest(α::FT) where {FT} =
    SteadyStateTest{FT, PhysicalParameters{FT}}(; α = α)

"""
    SteadyStateCompactTest{FT, P} <: AbstractTest

A second one, called "steady_state_compact", reproduces Test Case 3 in the same
reference paper. This test case gives the steady-state solution to the
non-linear shallow water equations with nonlinear zonal geostrophic flow
with compact support.

https://doi.org/10.1016/S0021-9991(05)80016-6

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct SteadyStateCompactTest{FT, P} <: AbstractTest
    "Physical parameters"
    params::P = PhysicalParameters{FT}()
    "advection velocity"
    u0::FT = 2 * pi * params.R / (12 * 86400)
    "peak of analytic height field"
    h0::FT = 2.94e4 / params.g
    "latitude lower bound for coordinate transformation parameter"
    ϕᵦ::FT = -30.0
    "latitude upper bound for coordinate transformation parameter"
    ϕₑ::FT = 90.0
    "velocity perturbation parameter"
    xₑ::FT = 0.3
    "angle between the north pole and the center of the top cube panel"
    α::FT
end
SteadyStateCompactTest(α::FT) where {FT} =
    SteadyStateCompactTest{FT, PhysicalParameters{FT}}(; α = α)

"""
    MountainTest{FT, P} <: AbstractTest

A third one, called "mountain", reproduces Test Case 5 in the same
reference paper. It represents a zonal flow over an isolated mountain,
where the governing equations describe a global steady-state nonlinear
zonal geostrophic flow, with a corresponding geostrophic height field over
a non-uniform reference surface h_s.

https://doi.org/10.1016/S0021-9991(05)80016-6

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct MountainTest{FT, P} <: AbstractTest
    "Physical parameters"
    params::P = PhysicalParameters{FT}()
    "advection velocity"
    u0::FT = 20.0
    "peak of analytic height field"
    h0::FT = 5960
    "radius of conical mountain"
    a::FT = 20.0
    "center of mountain long coord, shifted by 180 compared to the paper,
    because our λ ∈ [-180, 180] (in the paper it was 270, with λ ∈ [0, 360])"
    λc::FT = 90.0
    "latitude coordinate for center of mountain"
    ϕc::FT = 30.0
    "mountain peak height"
    h_s0::FT = 2e3
    "angle between the north pole and the center of the top cube panel"
    α::FT
end
MountainTest(α::FT) where {FT} =
    MountainTest{FT, PhysicalParameters{FT}}(; α = α)

"""
    RossbyHaurwitzTest{FT, P} <: AbstractTest

A fourth one, called "rossby_haurwitz", reproduces Test Case 6 in the same
reference paper. It represents the solution of the nonlinear barotropic
vorticity equation on the sphere

https://doi.org/10.1016/S0021-9991(05)80016-6

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct RossbyHaurwitzTest{FT, P} <: AbstractTest
    "Physical parameters"
    params::P = PhysicalParameters{FT}()
    "velocity amplitude parameter"
    a::FT = 4.0
    "peak of analytic height field"
    h0::FT = 8.0e3
    "vorticity amplitude parameter (1/sec)"
    ω::FT = 7.848e-6
    "vorticity amplitude parameter (1/sec)"
    K::FT = 7.848e-6
    "angle between the north pole and the center of the top cube panel"
    α::FT
end
RossbyHaurwitzTest(α::FT) where {FT} =
    RossbyHaurwitzTest{FT, PhysicalParameters{FT}}(; α = α)

"""
    BarotropicInstabilityTest{FT, P} <: AbstractTest

A fifth one, called "barotropic_instability", reproduces the test case in
Galewsky et al, "An initial-value problem for testing numerical models of
the global shallow-water equations", 2004 (also in Sec. 7.6 of Ullirch et al,
"High-order ﬁnite-volume methods for the shallow-water equations on
the sphere", 2010). This test case consists of a zonal jet with compact
support at a latitude of 45°. A small height disturbance is then added,
which causes the jet to become unstable and collapse into a highly vortical
structure.

https://doi.org/10.3402/tellusa.v56i5.14436
https://doi.org/10.1016/j.jcp.2010.04.044

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct BarotropicInstabilityTest{FT, P} <: AbstractTest
    "Physical parameters"
    params::P = PhysicalParameters{FT}()
    "maximum zonal velocity"
    u_max::FT = 80.0
    "mountain shape parameters"
    αₚ::FT = 19.09859
    "mountain shape parameters"
    βₚ::FT = 3.81971
    "peak of balanced height field from Tempest
    https://github.com/paullric/tempestmodel/blob/master/test/shallowwater_sphere/BarotropicInstabilityTest.cpp#L86"
    h0::FT = 10158.18617
    "local perturbation peak height"
    h_hat::FT = 120.0
    "southern jet boundary"
    ϕ₀::FT = 25.71428
    "northern jet boundary"
    ϕ₁::FT = 64.28571
    "height perturbation peak location"
    ϕ₂::FT = 45.0
    "zonal velocity decay parameter"
    eₙ::FT = exp(-4.0 / (deg2rad(ϕ₁) - deg2rad(ϕ₀))^2)
    "angle between the north pole and the center of the top cube panel"
    α::FT
end
BarotropicInstabilityTest(α::FT) where {FT} =
    BarotropicInstabilityTest{FT, PhysicalParameters{FT}}(; α = α)

# Definition of Coriolis parameter
function set_coriolis_parameter(space, test::AbstractTest)
    Ω = test.params.Ω
    α = test.α
    f_rossby_haurwitz(Ω, ϕ, λ, α) = 2 * Ω * sind(ϕ) # for rossby_haurwitz test
    f_default(Ω, ϕ, λ, α) = # all other test cases
        2 * Ω * (-cosd(λ) * cosd(ϕ) * sind(α) + sind(ϕ) * cosd(α))
    f_coriolis = test isa RossbyHaurwitzTest ? f_rossby_haurwitz : f_default

    return map(Fields.local_geometry_field(space)) do local_geometry
        ϕ = local_geometry.coordinates.lat
        λ = local_geometry.coordinates.long
        f = f_coriolis(Ω, ϕ, λ, α)
        # Technically this should be a WVector, but since we are only in a 2D space,
        # WVector, Contravariant3Vector, Covariant3Vector are all equivalent.
        # This _won't_ be true in 3D however!
        Geometry.Contravariant3Vector(f)
    end
end
# Definition of bottom surface topography field
function surface_topography(space, test::AbstractTest)
    if test isa MountainTest
        (; a, λc, ϕc, h_s0) = test
        h_s = map(Fields.coordinate_field(space)) do coord
            ϕ = coord.lat
            λ = coord.long
            r = sqrt(min(a^2, (λ - λc)^2 + (ϕ - ϕc)^2)) # positive branch
            h_s = h_s0 * (1 - r / a)
        end
    else
        h_s = zeros(space)
    end
    return h_s
end
# Set initial condition
function set_initial_condition(space, test::RossbyHaurwitzTest)
    (; a, h0, ω, K, α, params) = test
    (; R, Ω, g) = params
    Y = map(Fields.local_geometry_field(space)) do local_geometry
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
    return Y
end

function set_initial_condition(space, test::SteadyStateCompactTest)
    (; u0, h0, ϕᵦ, ϕₑ, xₑ, α, params) = test
    (; R, Ω, g) = params
    Y = map(Fields.local_geometry_field(space)) do local_geometry
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
        h =
            h0 - (R / g) * (pi / 180.0) * QuadGK.quadgk(h_int, -90.0, ϕprime)[1]

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
    return Y
end
function set_initial_condition(space, test::BarotropicInstabilityTest)
    (; u_max, αₚ, βₚ, h0, h_hat, ϕ₀, ϕ₁, ϕ₂, eₙ, α, params) = test
    (; R, Ω, g) = params

    Y = map(Fields.local_geometry_field(space)) do local_geometry
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
        h =
            h0 - (R / g) * (pi / 180.0) * QuadGK.quadgk(h_int, -90.0, ϕprime)[1]

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
    return Y
end
# steady-state and mountain test cases share the same form of fields
function set_initial_condition(
    space,
    test::T,
) where {T <: Union{MountainTest, SteadyStateTest}}
    (; u0, h0, α, params) = test
    (; R, Ω, g) = params
    Y = map(Fields.local_geometry_field(space)) do local_geometry
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
    return Y
end

function rhs!(dYdt, y, parameters, t)
    @nvtx "rhs!" color = colorant"red" begin
        (; f, h_s, ghost_buffer, params) = parameters
        (; ν₄, g) = params

        space = axes(y)
        D₄ = ν₄ * Spaces.node_horizontal_length_scale(space)^3 # hyperdiffusion coefficient


        div = Operators.Divergence()
        wdiv = Operators.WeakDivergence()
        grad = Operators.Gradient()
        wgrad = Operators.WeakGradient()
        curl = Operators.Curl()
        wcurl = Operators.WeakCurl()

        # Compute hyperviscosity first
        @nvtx "Hyperviscosity (rhs!)" color = colorant"green" begin
            @. dYdt.h = wdiv(grad(y.h))
            @. dYdt.u =
                wgrad(div(y.u)) - Geometry.Covariant12Vector(
                    wcurl(Geometry.Covariant3Vector(curl(y.u))),
                )

            Spaces.weighted_dss!(dYdt, ghost_buffer)

            @. dYdt.h = -D₄ * wdiv(grad(dYdt.h))
            @. dYdt.u =
                -D₄ * (
                    wgrad(div(dYdt.u)) - Geometry.Covariant12Vector(
                        wcurl(Geometry.Covariant3Vector(curl(dYdt.u))),
                    )
                )
        end
        @nvtx "h and u (rhs!)" color = colorant"blue" begin
            # Add in pieces
            @. begin
                dYdt.h += -wdiv(y.h * y.u)
                dYdt.u +=
                    -grad(g * (y.h + h_s) + norm(y.u)^2 / 2) +
                    y.u × (f + curl(y.u))
            end
            Spaces.weighted_dss!(dYdt, ghost_buffer)
        end
    end
    return dYdt
end

function shallow_water_driver(ARGS, ::Type{FT}) where {FT}
    context = ClimaComms.context()
    usempi = context isa ClimaComms.MPICommsContext
    if usempi
        pid, nprocs = ClimaComms.init(context)
        # log output only from root process
        logger_stream = ClimaComms.iamroot(context) ? stderr : devnull
        prev_logger = global_logger(ConsoleLogger(logger_stream, Logging.Info))
        atexit() do
            global_logger(prev_logger)
        end
        if ClimaComms.iamroot(context)
            println("running distributed simulation using $nprocs processes")
        end
    else
        global_logger(TerminalLoggers.TerminalLogger())
        println("running serial simulation")
    end
    # Test case specifications
    test_name = get(ARGS, 1, "steady_state") # default test case to run
    test_angle_name = get(ARGS, 2, "alpha0") # default test case to run
    α = parse(FT, replace(test_angle_name, "alpha" => ""))

    (!usempi || ClimaComms.iamroot(context)) &&
        println("Test name: $test_name, α = $(α)⁰")
    # Test-specific physical parameters
    test =
        test_name == "steady_state_compact" ? SteadyStateCompactTest(α) :
        (
            test_name == "mountain" ? MountainTest(α) :
            (
                test_name == "rossby_haurwitz" ? RossbyHaurwitzTest(α) :
                (
                    test_name == "barotropic_instability" ?
                    BarotropicInstabilityTest(α) : SteadyStateTest(α)
                )
            )
        )
    # Set up discretization
    ne = 9 # the rossby_haurwitz test case's initial state has a singularity at the pole. We avoid it by using odd number of elements
    Nq = 4

    domain = Domains.SphereDomain(test.params.R)
    mesh = Meshes.EquiangularCubedSphere(domain, ne)
    quad = Quadratures.GLL{Nq}()
    grid_topology = Topologies.Topology2D(context, mesh)
    if usempi
        global_grid_topology =
            Topologies.Topology2D(ClimaComms.SingletonCommsContext(), mesh)
        space = Spaces.SpectralElementSpace2D(grid_topology, quad)
        global_space = Spaces.SpectralElementSpace2D(global_grid_topology, quad)
    else
        global_space =
            space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    end
    @show Spaces.node_horizontal_length_scale(space)^3

    coords = Fields.coordinate_field(space)
    f = set_coriolis_parameter(space, test)
    h_s = surface_topography(space, test)
    Y = set_initial_condition(space, test)
    if !usempi
        Y0_global = deepcopy(Y)
    else
        Y0_global_values = DataLayouts.gather(context, Fields.field_values(Y))
        if ClimaComms.iamroot(context)
            Y0_global = Fields.Field(Y0_global_values, global_space)
        end
    end
    ghost_buffer = Spaces.create_dss_buffer(Y)
    # Set up RHS function
    dYdt = similar(Y)
    parameters =
        (; f = f, h_s = h_s, ghost_buffer = ghost_buffer, params = test.params)
    rhs!(dYdt, Y, parameters, 0.0)

    # Solve the ODE
    dt = 6 * 60
    T = 60 * 60 * 24 * 2

    prob = ODEProblem(rhs!, Y, (0.0, T), parameters)
    integrator = OrdinaryDiffEq.init(
        prob,
        SSPRK33(),
        dt = dt,
        saveat = dt,
        progress = true,
        adaptive = false,
        progress_message = (dt, u, p, t) -> t,
    )

    if usempi
        walltime = @elapsed sol = OrdinaryDiffEq.solve!(integrator)
        ClimaComms.iamroot(context) && println("walltime = $walltime (sec)")
    else
        sol = @timev OrdinaryDiffEq.solve!(integrator)
    end
    sol_global = []

    if usempi
        for sol_step in sol.u
            sol_step_values_global =
                DataLayouts.gather(context, Fields.field_values(sol_step))
            if ClimaComms.iamroot(context)
                sol_step_global =
                    Fields.Field(sol_step_values_global, global_space)
                push!(sol_global, sol_step_global)
            end
        end
    end
    solution = usempi ? sol_global : sol.u

    # post processing
    if !usempi || ClimaComms.iamroot(context)
        test_params =
            (test_name = test_name, test_angle_name = test_angle_name, α = α)
        postprocessing(test, test_params, solution, Y0_global, T, dt)
    end
    return integrator # for use with performance analyzer
end

function postprocessing(test, test_params, solution, Y0_global, T, dt)
    (; test_name, test_angle_name, α) = test_params
    # post processing (called only from root for distributed runs)
    # Plot variables and auxiliary function
    ENV["GKSwstype"] = "nul"
    Plots.GRBackend()
    dir = "cg_sphere_shallowwater_$(test_name)"
    dir = "$(dir)_$(test_angle_name)"
    path = joinpath(@__DIR__, "output", dir)
    mkpath(path)

    function linkfig(figpath, alt = "")
        # Buildkite-agent upload figpath
        # Link figure in logs if we are running on CI
        if get(ENV, "BUILDKITE", "") == "true"
            artifact_url = "artifact://$figpath"
            print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
        end
    end
    @info "Test case: $(test_name)"
    @info "  with α: $(α)⁰"
    @info "Solution L₂ norm at time t = 0: ", norm(Y0_global.h)
    @info "Solution L₂ norm at time t = $(T): ", norm(solution[end].h)
    @info "Fluid volume at time t = 0: ", sum(Y0_global.h)
    @info "Fluid volume at time t = $(T): ", sum(solution[end].h)

    if test isa SteadyStateTest || test isa SteadyStateCompactTest
        # In these cases, we use the IC as the reference exact solution
        @info "L₁ error at T = $(T): ", norm(solution[end].h .- Y0_global.h, 1)
        @info "L₂ error at T = $(T): ", norm(solution[end].h .- Y0_global.h)
        @info "L∞ error at T = $(T): ",
        norm(solution[end].h .- Y0_global.h, Inf)
        # Pointwise final L₂ error
        Plots.png(
            Plots.plot(solution[end].h .- Y0_global.h),
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
            relL1err[t] =
                norm(solution[t].h .- Y0_global.h, 1) / norm(Y0_global.h, 1)
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
            relL2err[t] = norm(solution[t].h .- Y0_global.h) / norm(Y0_global.h)
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
            RelLInferr[t] =
                norm(solution[t].h .- Y0_global.h, Inf) / norm(Y0_global.h, Inf)
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
        Plots.png(
            Plots.plot(solution[end].h),
            joinpath(path, "final_height.png"),
        )
        linkfig(
            relpath(
                joinpath(path, "final_height.png"),
                joinpath(@__DIR__, "../.."),
            ),
            "Height field at the final time step",
        )
    end
    return nothing
end

integrator = shallow_water_driver(ARGS, Float64)

if haskey(ENV, "CI_PERF_SKIP_RUN") # for performance analysis
    throw(:exit_profile)
end
