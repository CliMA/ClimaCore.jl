using Test
using StaticArrays
using ClimaComms
ClimaComms.@import_required_backends
import ClimaCore.DataLayouts: IJFH, VF
import ClimaCore:
    Geometry,
    Fields,
    Domains,
    Topologies,
    Meshes,
    Spaces,
    Operators,
    Quadratures
using LinearAlgebra, IntervalSets

FT = Float64
domain = Domains.RectangleDomain(
    Geometry.XPoint{FT}(-pi) .. Geometry.XPoint{FT}(pi),
    Geometry.YPoint{FT}(-pi) .. Geometry.YPoint{FT}(pi);
    x1periodic = true,
    x2periodic = true,
)

Nq = 5
quad = Quadratures.GLL{Nq}()
device = ClimaComms.CPUSingleThreaded()
grid_mesh = Meshes.RectilinearMesh(domain, 17, 16)
grid_topology =
    Topologies.Topology2D(ClimaComms.SingletonCommsContext(device), grid_mesh)
grid_space = Spaces.SpectralElementSpace2D(grid_topology, quad)
grid_coords = Fields.coordinate_field(grid_space)

@testset "split divergence" begin
    split_div = Operators.SplitDivergence()
    div = Operators.Divergence()
    
    # Case 1: Constant. Div should be 0.
    # rho*u * 1
    u = Geometry.UVVector.(ones(FT, grid_space), ones(FT, grid_space))
    psi = Ref(1.0)
    
    res = split_div.(u, psi)
    @test norm(res) < 1e-12

    # Case 2: Smooth function
    # u = (cos(x), 0)
    # psi = 2 + sin(x)
    # div(u * psi)
    
    u = @. Geometry.UVVector(cos(grid_coords.x), 0.0)
    psi = @. 2 + sin(grid_coords.x)
    
    res_split = split_div.(u, psi)
    Spaces.weighted_dss!(res_split)
    
    res_div = div.(u .* psi)
    Spaces.weighted_dss!(res_div)
    
    # Compare with standard divergence
    # They should be close.
    @test norm(res_split .- res_div) < 1e-2 
    
    # Conservation test
    # Integral of split_div over periodic domain should be 0.
    integral = sum(res_split)
    @test abs(integral) < 1e-12
end
