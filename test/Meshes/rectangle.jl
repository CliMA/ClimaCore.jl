using ClimaCore: Domains, Meshes, Geometry
using Test
using SparseArrays

domain = Domains.IntervalDomain(Geometry.XPoint(-1.0),Geometry.XPoint(1.0); periodic=true) * Domains.IntervalDomain(Geometry.YPoint(-1.0),Geometry.YPoint(1.0); periodic=true)
mesh = Meshes.RectangleMesh(domain, 1, 1)

