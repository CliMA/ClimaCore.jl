module Meshes

import RootSolvers
using DocStringExtensions

export RectilinearMesh,
    EquiangularCubedSphere, EquidistantCubedSphere, ConformalCubedSphere

import ..Domains:
    Domains,
    IntervalDomain,
    RectangleDomain,
    SphereDomain,
    boundary_names,
    coordinate_type
import ..Geometry
import SparseArrays, CubedSphere, LinearAlgebra, StaticArrays



"""
    AbstractMesh{dim}

A `Mesh` is an object which represents how we discretize a domain into elements.

It should be lightweight (i.e. exists on all MPI ranks), e.g for meshes stored
in a file, it would contain the filename.

# Face and vertex numbering

In 1D, faces and vertices are the same, and both are numbered `[1,2]`.

In 2D, a face is a line segment between to vertices, and both are numbered `[1,2,3,4]`,
in a counter-clockwise direction.
```
 v4        f3        v3
   o-----------------o
   |                 |	    face    vertices
   |                 |	      f1 =>  v1 v2
f4 |                 | f2     f2 =>  v2 v3
   |                 |	      f3 =>  v3 v4
   |                 |        f4 =>  v4 v1
   |                 |
   o-----------------o
  v1       f1        v2
```

# Interface

A subtype of `AbstractMesh` should define the following methods:
- [`domain(mesh)`](@ref)
- [`elements(mesh)`](@ref)
- [`is_boundary_face(mesh, elem, face)`](@ref)
- [`boundary_face_name(mesh, elem, face)`](@ref)
- [`opposing_face(mesh, elem, face)`](@ref)
- [`coordinates(mesh, elem, vert)`](@ref)
- [`containing_element`](@ref) (optional)

The following types/methods are provided by `AbstractMesh`:
- [`SharedVertices(mesh, elem, vert)`](@ref)
- [`face_connectivity_matrix(mesh[,elemorder])`](@ref face_connectivity_matrix)
- [`vertex_connectivity_matrix(mesh[,elemorder])`](@ref vertex_connectivity_matrix)
"""
abstract type AbstractMesh{dim} end

const AbstractMesh1D = AbstractMesh{1}
const AbstractMesh2D = AbstractMesh{2}

"""
    Meshes.domain(mesh::AbstractMesh)

The [`Domains.AbstractDomain`](@ref) on which the mesh is defined.
"""
function domain end

"""
    Meshes.elements(mesh::AbstractMesh)

An iterator over the elements of a mesh. Elements of a mesh can be of any type.
"""
function elements end

"""
    Meshes.is_boundary_face(mesh::AbstractMesh, elem, face::Int)::Bool

Determine whether face `face` of element `elem` is on the boundary of `mesh`.

`elem` should be an element of [`elements(mesh)`](@ref).
"""
function is_boundary_face end

"""
    Meshes.boundary_face_name(mesh::AbstractMesh, elem, face::Int)::Union{Symbol,Nothing}

The name of the boundary facing `face` of element `elem`, or `nothing` if it is
not on the boundary.
"""
function boundary_face_name end

"""
    opelem, opface, reversed = Meshes.opposing_face(mesh::AbstractMesh, elem, face::Int)

The element and face (`opelem`, `opface`) that oppose face `face` of element `elem`.
"""
function opposing_face end



include("common.jl")
include("interval.jl")
include("rectangle.jl")
include("cubedsphere.jl")

# deprecations
@deprecate EquispacedRectangleMesh(args...) RectilinearMesh(args...)
@deprecate equispaced_rectangular_mesh(args...) RectilinearMesh(args...)
@deprecate TensorProductMesh(args...) RectilinearMesh(args...)
@deprecate EquiangularSphereWarp() EquiangularCubedSphere
@deprecate EquidistantSphereWarp() EquidistantCubedSphere
@deprecate ConformalSphereWarp() ConformalCubedSphere
@deprecate Mesh2D(
    domain::Domains.SphereDomain,
    ::Type{T},
    Ne,
) where {T <: AbstractCubedSphere} T(domain, Ne)
@deprecate Mesh2D(domain::RectangleDomain, x1c, x2c) RectilinearMesh(
    IntervalMesh(domain.interval1, x1c),
    IntervalMesh(domain.interval2, x2c),
)
@deprecate coordinates(mesh::AbstractMesh, elem, ξ::NTuple) coordinates(
    mesh,
    elem,
    StaticArrays.SVector(ξ),
)

end # module
