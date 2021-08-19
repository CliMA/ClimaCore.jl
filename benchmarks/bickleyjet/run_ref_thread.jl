push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

include("bickleyjet_dg.jl")
include("bickleyjet_dg_reference.jl")

using BenchmarkTools

n1, n2 = 4096, 4096
Nq = 4

mesh = Meshes.EquispacedRectangleMesh(domain, n1, n2)
grid_topology = Topologies.GridTopology(mesh)
quad = Spaces.Quadratures.GLL{Nq}()
space = Spaces.SpectralElementSpace2D(grid_topology, quad)

X = coordinates(Val(Nq), n1, n2)
y0_ref = init_y0_ref(X, Val(Nq), parameters)
dydt_ref = similar(y0_ref)
tendency_states = init_tendency_states(n1, n2, Val(Nq))
volume_ref!(
    dydt_ref,
    y0_ref,
    (n1, n2, parameters, Val(Nq), tendency_states),
    0.0,
)

eltm = @belapsed volume_ref!(
    $dydt_ref,
    $y0_ref,
    ($n1, $n2, $parameters, $(Val(Nq)), $tendency_states),
    0.0,
)

println(eltm)
