# Bickley jet: a barotropically unstable shear layer in the shallow-water
# equations on a doubly periodic plane, discretized with a continuous Galerkin
# spectral element method (flux form, over-integration, DSS). A small
# perturbation seeds the instability, which rolls the jet up into vortices. The
# run tracks total energy, which the discretization should nearly conserve.
#
# See `bickleyjet_dg.jl` for the discontinuous Galerkin discretization of the
# same case, and `bickleyjet_cg_invariant_hypervisc.jl` for the vector-invariant
# form with hyperviscosity. This example uses potential temperature `ρθ`
# instead of the total energy `ρe` formulation enforced in the hybrid models,
# as `ρθ` is analytically conserved here and simplifies the setup.
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
import ClimaCore.Geometry: ⊗

import ClimaTimeSteppers as CTS

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

const context = ClimaComms.SingletonCommsContext()

const parameters = (
    ϵ = 0.1,  # perturbation size for initial condition
    l = 0.5, # Gaussian width
    k = 0.5, # Sinusoidal wavenumber
    ρ₀ = 1.0, # reference density
    g = 10,
)

domain = Domains.RectangleDomain(
    Domains.IntervalDomain(
        Geometry.XPoint(-2π),
        Geometry.XPoint(2π),
        periodic = true,
    ),
    Domains.IntervalDomain(
        Geometry.YPoint(-2π),
        Geometry.YPoint(2π),
        periodic = true,
    ),
)

n1, n2 = 16, 16
Nq = 4
Nqh = 7
mesh = Meshes.RectilinearMesh(domain, n1, n2)
grid_topology = Topologies.Topology2D(context, mesh)
quad = Quadratures.GLL{Nq}()
space = Spaces.SpectralElementSpace2D(grid_topology, quad)

Iquad = Quadratures.GLL{Nqh}()
Ispace = Spaces.SpectralElementSpace2D(grid_topology, Iquad)

function init_state(coord, p)
    x, y = coord.x, coord.y
    # set initial state
    ρ = p.ρ₀

    # set initial velocity
    U₁ = cosh(y)^(-2)

    # Ψ′ = exp(-(y + p.l / 10)^2 / 2p.l^2) * cos(p.k * x) * cos(p.k * y)
    # Vortical velocity fields (u₁′, u₂′) = (-∂²Ψ′, ∂¹Ψ′)
    gaussian = exp(-(y + p.l / 10)^2 / 2p.l^2)
    u₁′ = gaussian * (y + p.l / 10) / p.l^2 * cos(p.k * x) * cos(p.k * y)
    u₁′ += p.k * gaussian * cos(p.k * x) * sin(p.k * y)
    u₂′ = -p.k * gaussian * sin(p.k * x) * cos(p.k * y)


    u = Geometry.UVVector(U₁ + p.ϵ * u₁′, p.ϵ * u₂′)
    # set initial tracer
    θ = sin(p.k * y)

    return (ρ = ρ, ρu = ρ * u, ρθ = ρ * θ)
end

y0 = init_state.(Fields.coordinate_field(space), Ref(parameters))

function flux(state, param)
    ρ, ρu, ρθ = state.ρ, state.ρu, state.ρθ
    u = ρu / ρ
    return (
        ρ = ρu,
        ρu = ((ρu ⊗ u) + (param.g * ρ^2 / 2) * LinearAlgebra.I),
        ρθ = ρθ * u,
    )
end

function energy(state, param)
    ρ, ρu = state.ρ, state.ρu
    u = ρu / ρ
    return ρ * (u.u^2 + u.v^2) / 2 + param.g * ρ^2 / 2
end

function total_energy(y, parameters)
    sum(energy.(y, Ref(parameters)))
end


function rhs!(dydt, y, _, t)

    I = Operators.Interpolate(Ispace)
    div = Operators.WeakDivergence()
    R = Operators.Restrict(space)

    rparameters = Ref(parameters)

    @. dydt = -R(div(flux(I(y), rparameters)))

    Spaces.weighted_dss!(dydt)
    return dydt
end

dydt = similar(y0)
rhs!(dydt, y0, nothing, 0.0)


# Solve the ODE operator
prob = CTS.ODEProblem(CTS.ClimaODEFunction(; T_exp! = rhs!), y0, (0.0, 80.0), nothing)
sol = CTS.solve(
    prob,
    CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()),
    dt = 0.02,
    saveat = collect(0.0:1.0:80.0),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)

ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = "cg"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.ρθ, clim = (-1, 1))
end
Plots.mp4(anim, joinpath(path, "tracer.mp4"), fps = 10)

Es = [total_energy(u, parameters) for u in sol.u]

using Test
# The CG discretization conserves total energy up to time-integration error
# (measured drift over t = 80: +5e-5).
@test abs(Es[end] - Es[1]) / Es[1] < 1e-3

@testset "conservation and jet roll-up" begin
    y_start = sol.u[1]
    y_end = sol.u[end]
    # The domain is doubly periodic and the divergence is taken in weak form,
    # so mass and tracer mass are conserved to roundoff (measured: 5e-14
    # relative in ρ, 2e-12 absolute in ρθ, whose exact integral is zero because
    # θ = sin(k y) over a whole number of periods).
    @test abs(sum(y_end.ρ) - sum(y_start.ρ)) / sum(y_start.ρ) < 1e-12
    @test abs(sum(y_end.ρθ) - sum(y_start.ρθ)) < 1e-10
    # The shear layer is barotropically unstable, so the seeded perturbation
    # must grow: the cross-jet velocity amplifies by more than an order of
    # magnitude as the jet rolls up into vortices (measured max|v|: 0.050 at
    # t = 0, 0.78 at t = 80). Without this the energy check above is passed
    # just as well by a flow that never destabilizes.
    cross_jet_speed(y) =
        maximum(abs, Geometry.UVVector.(y.ρu ./ y.ρ).components.data.:2)
    @test cross_jet_speed(y_end) > 10 * cross_jet_speed(y_start)
    # Rolling the jet up draws the tracer into filaments the mesh cannot
    # resolve, and this discretization has no limiter, so θ overshoots its
    # initial [-1, 1] range (measured: [-3.1, 2.7]). See `limiters_advection.jl`
    # for the limited transport that does hold the bounds.
    θ_end = y_end.ρθ ./ y_end.ρ
    @test maximum(abs, θ_end) < 5
end
Plots.png(Plots.plot(Es), joinpath(path, "energy.png"))

include(joinpath(@__DIR__, "..", "example_utils.jl")) # linkfig

linkfig(
    relpath(joinpath(path, "energy.png"), joinpath(@__DIR__, "../..")),
    "Total Energy",
)
