module Domains

import ..Geometry
using IntervalSets
export RectangleDomain, EquispacedRectangleDiscretization


# QA:
# https://github.com/CliMA/ClimateMachine.jl/blob/ans/sphere/test/Numerics/DGMethods/compressible_navier_stokes_equations/shared_source/domains.jl

# DEM = array of heights + reference datum (e.g. WGS84)
#   e.g. https://www.ngdc.noaa.gov/mgg/global/global.html
#    provides vertex and cell-centered values

# Unwarped 2D domain (sphere, rectangle)
# need some sort of processing step to load into model cell / node ordering
#

# requirements
# want to support broadcasting 2D -> 3D, reduction 3D -> 2D
#

# idea: underlying 2D domain
#  - sphere
#  - rectangle
#  - unstructured mesh
#  - vertical / radial warping
#    - challenge is that we need the horizontal discretization first, then we can modify from there

# 3D construct by extruding
#  - parallel (shallow atmosphere) vs diverging (deep atmosphere)
#  - shallow vs deep geometry


# workflow:
#  1. base 2d domain (rect/sphere)
#  2. base 2d mesh:
#     - calculate coordinates + horizontal metric terms
#  3. interpolate topography to mesh given datum
#  4. construct 3d mesh from 2d mesh + topography
#     - compute 3d metric terms from horizontal * vertical
#  5. adjust metric terms if using terrain following coordinate systems
#     - single pass over the mesh  for adjustment

# what do we need coordinates for?
#  - appear as terms in equations, e.g.
#    - coriolis
#    - gravitational potential
#  - basis vector-valued quantities (velocities, flux)
#  - interpolation to/from grid
#  - initial conditions

abstract type HorizontalDomain end
abstract type VerticalDomain end

Base.@kwdef struct IntervalDomain{FT} <: VerticalDomain
    x3min::FT
    x3max::FT
end

coordinate_type(::IntervalDomain{FT}) where {FT} = Geometry.Cartesian3Point{FT}



# coordinates (x1,x2)
# TODO: should we have boundary tags?
# or just specify using the same numbering we use for faces?
struct RectangleDomain{FT, B1, B2} <: HorizontalDomain
    x1min::FT
    x1max::FT
    x2min::FT
    x2max::FT
    x1boundary::B1
    x2boundary::B2
end


RectangleDomain(
    x1::ClosedInterval,
    x2::ClosedInterval;
    x1boundary = (:west, :east),
    x2boundary = (:south, :north),
    x1periodic = false,
    x2periodic = false,
) = RectangleDomain(
    float(x1.left),
    float(x1.right),
    float(x2.left),
    float(x2.right),
    x1periodic ? nothing : x1boundary,
    x2periodic ? nothing : x2boundary,
)

function Base.show(io::IO, domain::RectangleDomain)
    print(
        io,
        "RectangleDomain($(domain.x1min)..$(domain.x1max), $(domain.x2min)..$(domain.x2max)",
    )
    if domain.x1boundary == nothing
        print(io, ", x1periodic=true")
    else
        print(io, ", x1boundary=$(domain.x1boundary)")
    end
    if domain.x2boundary == nothing
        print(io, ", x2periodic=true")
    else
        print(io, ", x2boundary=$(domain.x2boundary)")
    end
    print(io, ")")
end
coordinate_type(::RectangleDomain{FT}) where {FT} =
    Geometry.Cartesian2DPoint{FT}

# coordinates (-pi/2 < lat < pi/2, -pi < lon < pi)
struct SphereDomain{FT} <: HorizontalDomain
    radius::FT
end


abstract type HorizontalCoordinate end

struct LatLon{FT} <: HorizontalCoordinate
    latitude::FT
    longitude::FT
end

"""
object which represents how we discretize the domain into rectangles
lightweight(i.e. exists on all MPI ranks)
"""
abstract type Discretization end


"""
    EquispacedRectangleDiscretization(domain::RectangleDomain, n1::Integer, n2::Integer)

A regular discretization of `domain` with `n1` elements in dimension 1, and `n2`
in dimension 2.
"""
struct EquispacedRectangleDiscretization{FT, RD <: RectangleDomain{FT}, R} <:
       Discretization
    domain::RD
    n1::Int64 # number of elements in x1 direction
    n2::Int64 # number of elements in x2 direction
    range1::R
    range2::R
end

function EquispacedRectangleDiscretization(domain::RectangleDomain, n1, n2)
    range1 = range(domain.x1min, domain.x1max; length = n1 + 1)
    range2 = range(domain.x2min, domain.x2max; length = n2 + 1)
    EquispacedRectangleDiscretization(domain, n1, n2, range1, range2)
end
Base.eltype(::EquispacedRectangleDiscretization{FT}) where {FT} = FT
function Base.show(io::IO, disc::EquispacedRectangleDiscretization)
    print(io, disc.n1, "×", disc.n2, " EquispacedRectangleDiscretization of ")
    print(io, disc.domain)
end


struct EquiangularCubedSphereDiscretization{FT} <: Discretization
    domain::SphereDomain{FT}
    n::Int64
    # 6*n^2 elements
end

# PetSC
# https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DT/index.html

# Discretization = global object which describes the mesh
# Topology = Formed local discretization + ordering of elements + partitioning
#  - enumeration of local elements 1:nlocalelems(...)
#  - for each local (element, face) pair, we need the opposing (element, face, reversed)
#     faces are enumerated 1:4, `reversed` is a Boolean flag whether the orientation between them is reversed
# opposing_face(topology, localelem, face) => (opelement, opface, reversed)

### interfaces

# global horz element id -> some sort of numbering

# local horz element numbering (ideally a sequential range of a global horz range)


"""
    HorizontalMesh

Locally formed mesh:
  - coordinates of the quads
  - quadrature
  -
https://github.com/gridap/Gridap.jl/blob/master/src/Geometry/Grids.jl#L3
https://github.com/gridap/Gridap.jl/blob/master/src/Geometry/Triangulations.jl#L3
"""
abstract type AbstractHorizontalMesh end



"""
    HorizontalMesh

A non-distributed horizontal mesh.

contains:
 - discretization
 - quadrature style (determines locations of nodes within each element)
 - topology: connections between elements
 - contains necessary info at each node
   - coordinates
   - metrics:


"""
struct HorizontalMesh{D, Q} <: AbstractHorizontalMesh
    discretization::D
    quadraturestyle::Q
    topology::Any
    coordinates::Any
    metrics::Any
end



"""
"""
abstract type AbstractVerticalMesh end


# struct VerticalMesh{D} <: AbstractVerticalMesh
# end



#=
struct MPIHorizontalMesh <: AbstractHorizontalMesh
end
=#


# can also add objects

# struct RectangularDomain{FT} <: AbstractDomain
#     Np::Int
#     Ne::NamedTuple{(:x, :y, :z), NTuple{3, Int}}
#     L::NamedTuple{(:x, :y, :z), NTuple{3, FT}}
#     x::NTuple{2, FT}
#     y::NTuple{2, FT}
#     zfunc(x,y)::NTuple{2, FT}
#     periodicity::NamedTuple{(:x, :y, :z), NTuple{3, Bool}}
# end



end # module
