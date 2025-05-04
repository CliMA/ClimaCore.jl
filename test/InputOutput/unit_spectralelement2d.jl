#=
julia --project=.buildkite
using Revise; include("test/InputOutput/unit_spectralelement2d.jl")
=#
using Test
using ClimaComms
ClimaComms.@import_required_backends
using LinearAlgebra
import ClimaCore
import ClimaCore.Utilities.Cache
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

function init_space(context; enable_bubble, enable_mask)
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
    grid_topology = Topologies.Topology2D(context, mesh)

    return Spaces.SpectralElementSpace2D(
        grid_topology,
        quad;
        enable_bubble,
        enable_mask,
    )
end

# function mktempfile(f)
#     mktempdir() do dir
#         cd(dir) do
#             f(tempname(dir))
#         end
#     end
# end

function mktempfile(f, context)
    filename =
        ClimaComms.iamroot(context) ? tempname(pwd(); cleanup = true) : ""
    filename = ClimaComms.bcast(context, filename)
    f(filename)
end

@testset "restart test for 2D spectral element simulations" begin
    device = ClimaComms.device()
    @info "Using device" device
    context = ClimaComms.context(device)
    parameters = (
        ϵ = 0.1,  # perturbation size for initial condition
        l = 0.5, # Gaussian width
        k = 0.5, # Sinusoidal wavenumber
        ρ₀ = 1.0, # reference density
        c = 2,
        g = 10,
        D₄ = 1e-4, # hyperdiffusion coefficient
    )
    space = init_space(context; enable_bubble = true, enable_mask = false)
    y0 = init_state.(Fields.local_geometry_field(space), Ref(parameters))
    Y = Fields.FieldVector(y0 = y0)

    # write field vector to hdf5 file
    mktempfile(context) do filename
        InputOutput.HDF5Writer(filename, context) do writer
            InputOutput.write!(writer, "Y" => Y) # write field vector from hdf5 file
        end
        Cache.clean_cache!()
        InputOutput.HDF5Reader(filename, context) do reader
            restart_Y = InputOutput.read_field(reader, "Y") # read fieldvector from hdf5 file
            @test restart_Y == Y # test if restart is exact
            @test axes(restart_Y) == axes(Y) # test if restart is exact for space
        end
    end

    # Test with masks
    space = init_space(context; enable_bubble = true, enable_mask = true)
    y0 = init_state.(Fields.local_geometry_field(space), Ref(parameters))
    Y = Fields.FieldVector(y0 = y0)

    Spaces.set_mask!(space) do coords
        sin(coords.x) > 0.5
    end

    # write field vector to hdf5 file
    mktempfile(context) do filename
        InputOutput.HDF5Writer(filename, context) do writer
            InputOutput.write!(writer, "Y" => Y) # write field vector from hdf5 file
        end

        InputOutput.HDF5Reader(filename, context) do reader
            # We need to clean the cache so that the next read of space
            # does not use a pointer to the cached one.
            Cache.clean_cache!()
            restart_Y = InputOutput.read_field(reader, "Y") # read fieldvector from hdf5 file

            is_active_restart =
                parent(Spaces.get_mask(axes(restart_Y.y0)).is_active)
            is_active = parent(Spaces.get_mask(axes(Y.y0)).is_active)
            @test is_active == is_active_restart

            # Test that we're not doing trivial pointer comparisons
            @test !(axes(Y.y0) === axes(restart_Y.y0))
            @test restart_Y == Y
            @test typeof(axes(restart_Y.y0)) == typeof(axes(Y.y0))
            @test axes(restart_Y) == axes(Y) # test if restart is exact for space
        end
    end
end
