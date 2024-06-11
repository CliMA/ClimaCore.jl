using Test
using ClimaComms
ClimaComms.@import_required_backends
using LinearAlgebra
import ClimaCore
import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Spaces,
    Quadratures,
    Topologies,
    DataLayouts,
    InputOutput

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

@testset "restart test for 2D spectral element simulations" begin
    parameters = (
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
    n1, n2 = 16, 16
    Nq = 4
    quad = Quadratures.GLL{Nq}()
    mesh = Meshes.RectilinearMesh(domain, n1, n2)
    device = ClimaComms.device()
    @info "Using device" device
    context = ClimaComms.SingletonCommsContext(device)
    grid_topology = Topologies.Topology2D(context, mesh)
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    y0 = init_state.(Fields.local_geometry_field(space), Ref(parameters))
    Y = Fields.FieldVector(y0 = y0)

    # write field vector to hdf5 file
    filename = tempname(pwd())
    writer = InputOutput.HDF5Writer(filename, context)
    InputOutput.write!(writer, "Y" => Y) # write field vector from hdf5 file
    close(writer)
    reader = InputOutput.HDF5Reader(filename, context)
    restart_Y = InputOutput.read_field(reader, "Y") # read fieldvector from hdf5 file
    close(reader)
    ClimaComms.allowscalar(device) do
        @test restart_Y == Y # test if restart is exact
    end
end
