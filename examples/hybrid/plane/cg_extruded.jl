push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using Test
using StaticArrays, IntervalSets, LinearAlgebra, UnPack

import ClimaCore:
    ClimaCore,
    slab,
    Spaces,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    Spaces,
    Fields,
    Operators
using ClimaCore.Geometry

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

FT = Float64
xlim = (0, 1)
zlim = (0, 1)
xelem = 10
zelem = 10
npoly = 3

# Horizontal Space
horzdomain = Domains.IntervalDomain(
    Geometry.XPoint{FT}(xlim[1]),
    Geometry.XPoint{FT}(xlim[2]);
    periodic = true,
)
horzmesh = Meshes.IntervalMesh(horzdomain, nelems = xelem)
horztopology = Topologies.IntervalTopology(horzmesh)
quad = Spaces.Quadratures.GLL{npoly + 1}()
horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)
# Vertical Space
vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
)
vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
verttopology = Topologies.IntervalTopology(vertmesh)
quad = Spaces.Quadratures.GLL{npoly + 1}()
vertspace = Spaces.SpectralElementSpace1D(verttopology, quad)
# Collocated Extruded Space
extruded_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vertspace)
