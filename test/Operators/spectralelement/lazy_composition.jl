using Test
using ClimaComms
ClimaComms.@import_required_backends
import ClimaCore:
    Geometry,
    Fields,
    Domains,
    Topologies,
    Meshes,
    Spaces,
    Operators,
    Quadratures
import ClimaCore.Operators as O
import Base.Broadcast as BB
using LinearAlgebra, IntervalSets

@testset "spectral broadcast style combination" begin
    Dev = O.AbstractSpectralStyle(ClimaComms.device())
    @test BB.result_style(O.SpectralStyle(), Dev()) === Dev()
    @test BB.result_style(Dev(), O.SpectralStyle()) === Dev()
    @test BB.result_style(O.SpectralStyle(), O.SpectralStyle()) ===
          O.SpectralStyle()
    @test BB.result_style(Dev(), Dev()) === Dev()
end

@testset "operator over a pre-instantiated spectral broadcast" begin
    FT = Float64
    domain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(-pi) .. Geometry.XPoint{FT}(pi),
        Geometry.YPoint{FT}(-pi) .. Geometry.YPoint{FT}(pi);
        x1periodic = true,
        x2periodic = true,
    )
    mesh = Meshes.RectilinearMesh(domain, 4, 4)
    device = ClimaComms.device()
    topo = Topologies.Topology2D(ClimaComms.SingletonCommsContext(device), mesh)
    space = Spaces.SpectralElementSpace2D(topo, Quadratures.GLL{4}())
    coords = Fields.coordinate_field(space)
    f = @. sin(coords.x) * cos(coords.y)
    K = @. FT(1) + coords.y^2 / 10
    grad = Operators.Gradient()
    wdiv = Operators.WeakDivergence()
    ref = @. wdiv(K * grad(f))
    # Instantiation assigns the device-resolved style, as when the broadcast is
    # captured with `LazyBroadcast.lazy` and materialized.
    ∇f = BB.instantiate(
        Base.broadcasted(+, Base.broadcasted(grad, f), Base.broadcasted(grad, f)),
    )
    out = @. wdiv(K * ∇f)
    @test parent(out) ≈ 2 .* parent(ref)
end
