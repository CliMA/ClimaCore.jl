using Test
using StaticArrays
using ClimaComms, ClimaCore
import ClimaCore:
    Geometry, Fields, Domains, Topologies, Meshes, Spaces, Operators
using LinearAlgebra, IntervalSets
using CUDA

FT = Float64
domain = Domains.RectangleDomain(
    Geometry.XPoint{FT}(-pi) .. Geometry.XPoint{FT}(pi),
    Geometry.YPoint{FT}(-pi) .. Geometry.YPoint{FT}(pi);
    x1periodic = true,
    x2periodic = true,
)

Nq = 5
quad = Spaces.Quadratures.GLL{Nq}()

grid_mesh = Meshes.RectilinearMesh(domain, 17, 16)


grid_topology_cpu = Topologies.Topology2D(
    ClimaComms.SingletonCommsContext(
        ClimaCore.Device.device(; disablegpu = true),
    ),
    grid_mesh,
)
grid_space_cpu = Spaces.SpectralElementSpace2D(grid_topology_cpu, quad)
coords_cpu = Fields.coordinate_field(grid_space_cpu)

f_cpu = sin.(coords_cpu.x .+ 2 .* coords_cpu.y)
g_cpu =
    Geometry.UVVector.(
        sin.(coords_cpu.x),
        2 .* cos.(coords_cpu.y .+ coords_cpu.x),
    )

grid_topology = Topologies.Topology2D(
    ClimaComms.SingletonCommsContext(ClimaCore.Device.device()),
    grid_mesh,
)
grid_space = Spaces.SpectralElementSpace2D(grid_topology, quad)
coords = Fields.coordinate_field(grid_space)

CUDA.allowscalar(false)
f = sin.(coords.x .+ 2 .* coords.y)
g = Geometry.UVVector.(sin.(coords.x), 2 .* cos.(coords.y .+ coords.x))

grad = Operators.Gradient()
wgrad = Operators.WeakGradient()

@test Array(parent(grad.(f))) ≈ parent(grad.(f_cpu))
@test Array(parent(wgrad.(f))) ≈ parent(wgrad.(f_cpu))

div = Operators.Divergence()
wdiv = Operators.WeakDivergence()

@test Array(parent(div.(g))) ≈ parent(div.(g_cpu))
@test Array(parent(wdiv.(g))) ≈ parent(wdiv.(g_cpu))
@test Array(parent(div.(grad.(f)))) ≈ parent(div.(grad.(f_cpu))) # composite

curl = Operators.Curl()
wcurl = Operators.WeakCurl()

@test Array(parent(curl.(Geometry.Covariant12Vector.(g)))) ≈
      parent(curl.(Geometry.Covariant12Vector.(g_cpu)))
@test Array(parent(curl.(Geometry.Covariant3Vector.(f)))) ≈
      parent(curl.(Geometry.Covariant3Vector.(f_cpu)))
@test Array(parent(wcurl.(Geometry.Covariant12Vector.(g)))) ≈
      parent(wcurl.(Geometry.Covariant12Vector.(g_cpu)))
@test Array(parent(wcurl.(Geometry.Covariant3Vector.(f)))) ≈
      parent(wcurl.(Geometry.Covariant3Vector.(f_cpu)))
