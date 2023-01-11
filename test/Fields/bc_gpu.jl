using Test
using CUDA
import ClimaCore.Fields as Fields
import ClimaCore.Device as Device
import ClimaCore.Domains as Domains
import ClimaCore.Geometry as Geometry
import ClimaCore.Meshes as Meshes
import ClimaCore.Topologies as Topologies
import ClimaCore.Spaces as Spaces
import ClimaComms

function foo!(::Type{FT}) where {FT}
    context = ClimaComms.SingletonCommsContext(Device.device())
    domain = Domains.RectangleDomain(
        Domains.IntervalDomain(
            Geometry.XPoint(FT(0)),
            Geometry.XPoint(FT(1)),
            periodic = true,
        ),
        Domains.IntervalDomain(
            Geometry.YPoint(FT(0)),
            Geometry.YPoint(FT(1)),
            periodic = true,
        ),
    )
    mesh = Meshes.RectilinearMesh(domain, 2, 2)
    topology = Topologies.Topology2D(context, mesh)
    quad = Spaces.Quadratures.GLL{3}()
    space = Spaces.SpectralElementSpace2D(topology, quad)
    (; x) = Fields.coordinate_field(space)
    ϕ = @. sin(2 * FT(π) * x)
    return nothing
end

@testset "GPU Field broadcasting" begin
    try
        foo!(Float64)
        @test_broken true
    catch
        @test_broken false
    end
end
