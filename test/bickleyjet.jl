using ClimateMachineCore.Geometry, LinearAlgebra, UnPack
import ClimateMachineCore: Fields, Domains, Topologies, Meshes
import ClimateMachineCore.Operators
import ClimateMachineCore.Geometry
using LinearAlgebra

parameters = (
    ϵ = 0.1,  # perturbation size for initial condition
    l = 0.5, # Gaussian width
    k = 0.5, # Sinusoidal wavenumber
    ρ₀ = 1.0, # reference density
    c = 2,
    g = 10,
)


domain = Domains.RectangleDomain(
    x1min = -2π,
    x1max = 2π,
    x2min = -2π,
    x2max = 2π,
    x1periodic = true,
    x2periodic = false,
)

n1, n2 = 5, 5
Nq = 6
discretization = Domains.EquispacedRectangleDiscretization(domain, n1, n2)
grid_topology = Topologies.GridTopology(discretization)
mesh = Meshes.Mesh2D(grid_topology, Meshes.Quadratures.GLL{Nq}())

function init_state(x, p)
    @unpack x1,x2 = x
    # set initial state
    ρ = p.ρ₀

    # set initial velocity
    U₁ = cosh(x2)^(-2)
    Ψ′ = exp(-(x2 + p.l / 10)^2 / 2p.l^2 ) * cos(p.k * x1) * cos(p.k * x2)

    ## Vortical velocity fields
    u₁′ =  Ψ′ * (p.k * tan(p.k * x2) + x2 / p.l^2)
    u₂′ = -Ψ′ * (p.k * tan(p.k * x1))

    u = Cartesian12Vector(U₁ + p.ϵ * u₁′, p.ϵ * u₂′)

    # set initial tracer
    θ = sin(p.k * x2)

    return (
        ρ  = ρ,
        ρu = ρ * u,
        ρθ = ρ * θ
    )
end

y0 = init_state.(Fields.coordinate_field(mesh), Ref(parameters))

function flux(state, p)
    @unpack ρ, ρu, ρθ = state
    u = ρu ./ ρ
    return (
        ρ  = ρu,
        ρu = (ρu ⊗ u) + (p.g * ρ^2 / 2) * I,
        ρθ = ρθ .* u,
    )
end

flux.(y0, Ref(parameters))
