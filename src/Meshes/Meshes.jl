module Meshes

import RootSolvers

export RectilinearMesh,
    EquiangularCubedSphere,
    EquidistantCubedSphere,
    ConformalCubedSphere,
    truncate_mesh

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

Abstract supertype of meshes, which describe how a domain is discretized into
elements. Subtypes: `IntervalMesh`, `RectilinearMesh`, and the
[`AbstractCubedSphere`](@ref) meshes.

A mesh is lightweight and exists on all MPI ranks; a mesh stored in a file, for
example, holds only the filename.

# Notes

**Face and vertex numbering.** In 1D, faces and vertices coincide, and both are
numbered `[1, 2]`. In 2D, a face is a line segment between two vertices, and both
are numbered `[1, 2, 3, 4]` in counter-clockwise order.

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

**Interface.** A subtype of `AbstractMesh` defines the following methods:

  - [`domain(mesh)`](@ref)
  - [`elements(mesh)`](@ref)
  - [`is_boundary_face(mesh, elem, face)`](@ref)
  - [`boundary_face_name(mesh, elem, face)`](@ref)
  - [`opposing_face(mesh, elem, face)`](@ref)
  - [`coordinates(mesh, elem, vert)`](@ref)
  - [`containing_element`](@ref) (optional)

The following types and methods are provided for every `AbstractMesh`:

  - [`SharedVertices(mesh, elem, vert)`](@ref)
  - [`face_connectivity_matrix(mesh[,elemorder])`](@ref face_connectivity_matrix)
  - [`vertex_connectivity_matrix(mesh[,elemorder])`](@ref vertex_connectivity_matrix)
"""
abstract type AbstractMesh{dim} end

const AbstractMesh1D = AbstractMesh{1}
const AbstractMesh2D = AbstractMesh{2}

"""
    Meshes.domain(mesh::AbstractMesh)

Return the domain (a subtype of [`Domains.AbstractDomain`](@ref)) on which `mesh` is
defined.
"""
function domain end

"""
    Meshes.elements(mesh::AbstractMesh)

Return an iterator over the elements of `mesh`. Elements can be of any type.
"""
function elements end

"""
    Meshes.is_boundary_face(mesh::AbstractMesh, elem, face::Int)::Bool

Return `true` if face `face` of element `elem` is on the boundary of `mesh`.

`elem` is an element of [`elements(mesh)`](@ref).
"""
function is_boundary_face end

"""
    Meshes.boundary_face_name(mesh::AbstractMesh, elem, face::Int)::Union{Symbol,Nothing}

Return the name of the boundary containing face `face` of element `elem`, or
`nothing` if the face is not on the boundary.
"""
function boundary_face_name end

"""
    opelem, opface, reversed = Meshes.opposing_face(mesh::AbstractMesh, elem, face::Int)

Return the element and face (`opelem`, `opface`) opposite face `face` of element
`elem`, and whether the node ordering along the shared face is `reversed` relative
to `face`.
"""
function opposing_face end

"""
    Meshes.element_horizontal_length_scale(mesh::AbstractMesh)

Return the approximate length scale of the elements of `mesh`, in the units of the
domain coordinates.
"""
function element_horizontal_length_scale end

include("common.jl")
include("interval.jl")
include("rectangle.jl")
include("cubedsphere.jl")


end # module
