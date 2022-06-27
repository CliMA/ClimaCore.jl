using LinearAlgebra

import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Spaces,
    Topologies,
    DataLayouts


const parameters = (
    ϵ = 0.1,  # perturbation size for initial condition
    l = 0.5, # Gaussian width
    k = 0.5, # Sinusoidal wavenumber
    ρ₀ = 1.0, # reference density
    c = 2,
    g = 10,
    D₄ = 1e-4, # hyperdiffusion coefficient
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
n1, n2 = 4, 4
Nq = 4
quad = Spaces.Quadratures.GLL{Nq}()
mesh = Meshes.RectilinearMesh(domain, n1, n2)
grid_topology = Topologies.Topology2D(mesh)
global_space = space = Spaces.SpectralElementSpace2D(grid_topology, quad)

function init_state(local_geometry, p)
    coord = local_geometry.coordinates
    x, y = coord.x, coord.y
    # set initial state
    ρ = p.ρ₀

    # set initial velocity
    U₁ = cosh(y)^(-2)

    # Ψ′ = exp(-(x2 + p.l / 10)^2 / 2p.l^2) * cos(p.k * x) * cos(p.k * y)
    # Vortical velocity fields (u₁′, u₂′) = (-∂²Ψ′, ∂¹Ψ′)
    gaussian = exp(-(y + p.l / 10)^2 / 2p.l^2)
    u₁′ = gaussian * (y + p.l / 10) / p.l^2 * cos(p.k * x) * cos(p.k * y)
    u₁′ += p.k * gaussian * cos(p.k * x) * sin(p.k * y)
    u₂′ = -p.k * gaussian * sin(p.k * x) * cos(p.k * y)

    u = Geometry.Covariant12Vector(
        Geometry.UVVector(U₁ + p.ϵ * u₁′, p.ϵ * u₂′),
        local_geometry,
    )

    # set initial tracer
    θ = sin(p.k * y)
    return (ρ = ρ, u = u, ρθ = ρ * θ)
end

y0 = init_state.(Fields.local_geometry_field(space), Ref(parameters))


curl = Operators.Curl()
wcurl = Operators.WeakCurl()

x = @. Geometry.Covariant3Vector(curl(y0.u))
b = @. wcurl(x)

# this one is fine
sbc1 = Base.Broadcast.instantiate(Base.Broadcast.broadcasted(curl,y0.u))
Operators.apply_operator(sbc1.op, space, Fields.SlabIndex(nothing, 1), sbc1.args[1])

# this one is not
sbc2 = Base.Broadcast.instantiate(Base.Broadcast.broadcasted(wcurl, x))
Operators.apply_operator(sbc2.op, space, Fields.SlabIndex(nothing, 1), sbc2.args[1])

using Profile
Profile.clear_malloc_data()
@. b = wcurl(x)
