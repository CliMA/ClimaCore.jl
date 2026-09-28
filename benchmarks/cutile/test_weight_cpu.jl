#=
CPU-only validation of the KronGEMM weight construction and the VIJFH layout
assumptions used by the cuTile contender, against ClimaCore's Gradient.
Runs anywhere (no GPU required):

    julia +1.11 --project=benchmarks/cutile benchmarks/cutile/test_weight_cpu.jl
=#

import ClimaComms
import ClimaCore:
    DataLayouts,
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Quadratures,
    Spaces,
    Topologies

include(joinpath(@__DIR__, "gradient_kernels.jl")) # gradient_weight

FT = Float64
Nq = 4
helem = 2
zelem = 4

context = ClimaComms.context(ClimaComms.CPUSingleThreaded())
radius = FT(6.37122e6)
hdomain = Domains.SphereDomain(radius)
hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
htopology = Topologies.Topology2D(context, hmesh)
quad = Quadratures.GLL{Nq}()
hspace = Spaces.SpectralElementSpace2D(htopology, quad)
vertdomain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(0),
    Geometry.ZPoint{FT}(30e3);
    boundary_names = (:bottom, :top),
)
vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
vtopology = Topologies.IntervalTopology(context, vertmesh)
vspace = Spaces.CenterFiniteDifferenceSpace(vtopology)
space = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)

f = zeros(space)
coords = Fields.coordinate_field(space)
@. f = FT(2) + sind(coords.long) * cosd(coords.lat) * (1 + coords.z / FT(30e3))

grad = Operators.Gradient()
∇f = @. grad(f)

fv = Fields.field_values(f)
@assert fv isa DataLayouts.VIJFH
pf = parent(fv)
p∇ = parent(Fields.field_values(∇f))
(Nv, _, _, _, Nh) = size(pf)
@assert size(pf) == (Nv, Nq, Nq, 1, Nh)
@assert size(p∇) == (Nv, Nq, Nq, 2, Nh)

D = Quadratures.differentiation_matrix(FT, quad)
K = Nq * Nq
C = 2K
Wt = gradient_weight(D)

F3 = reshape(pf, Nv, K, Nh)
O3 = similar(pf, Nv, C, Nh)
for h in 1:Nh
    O3[:, :, h] = F3[:, :, h] * Wt
end
O5 = reshape(O3, Nv, Nq, Nq, 2, Nh)

err = maximum(abs.(O5 .- p∇)) / maximum(abs.(p∇))
println("max relative error KronGEMM vs Operators.Gradient: ", err)
@assert err < 1e-13
println("KronGEMM weight ordering + VIJFH layout: PASS")
