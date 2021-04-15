module Domains

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

# coordinates (x1,x2)
# TODO: should we have boundary tags?
# or just specify using the same numbering we use for faces?
Base.@kwdef struct RectangleDomain{FT} <: HorizontalDomain
    x1min::FT
    x1max::FT
    x2min::FT
    x2max::FT
    x1periodic::Bool
    x2periodic::Bool
end

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

struct EquispacedRectangleDiscretization{FT} <: Discretization
    domain::RectangleDomain{FT}
    n1::Int64 # number of elements in x1 direction
    n2::Int64 # number of elements in x2 direction
    # n1*n2 elements
end

struct EquiangularCubedSphereDiscretization{FT} <: Discretization
    domain::SphereDomain{FT}
    n::Int64
    # 6*n^2 elements
end

abstract type QuadratureStyle end
struct LGL{N} <: QuadratureStyle end

# PetSC
# https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DT/index.html

# Discretization = global object which describes the mesh
# Topology = Formed local discretization + ordering of elements + partitioning
#  - enumeration of local elements 1:nlocalelems(...)
#  - for each local (element, face) pair, we need the opposing (element, face, reversed)
#     faces are enumerated 1:4, `reversed` is a Boolean flag whether the orientation between them is reversed
# opposing_face(topology, localelem, face) => (opelement, opface, reversed)

### interfaces

# for element in 1:nelements(...)
#   for face = 1:4
#     ijk = ...
#     opposing_face(elem,face)
#
#     X[ijk,f,e] +=
#
#     if face % 2 == 0
#        @synchronize
#     end
#   end
# end


# for unique_face in unique_interior_face
#   (elem1, face1, elem2, face2, reversed) = unique_face
#
#      # potential for race conditions (since a node can belong in multiple face)

# for mesh_vertex in mesh_vertices(...)
#    for (elem, vert) in mesh_vertex

# for face in boundary_faces(topology, 1)



# for face_partition in unique_interior_face_partition(...)
#    for face in face_partition(...)
#

# periodicity / boundary labelling?

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
