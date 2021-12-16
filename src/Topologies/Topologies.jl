"""
    ClimaCore.Topologies

Objects describing the horizontal connections between elements.

All elements are quadrilaterals, using the following face and vertex numbering
convention:
```
          3
      4-------3
 ^    |       |
 |  4 |       | 2
ξ2    |       |
      1-------2
          1
        ξ1-->
```
"""
module Topologies

import ..Geometry
import ..Domains: Domains, coordinate_type
import ..Meshes: Meshes, domain

# TODO: seperate types for MPI/non-MPI topologies
"""
   AbstractTopology

Subtypes of `AbstractHorizontalTopology` define connectiveness of a
mesh in the horizontal domain.
"""
abstract type AbstractTopology end

coordinate_type(topology::AbstractTopology) = coordinate_type(domain(topology))

"""
    nlocalelems(topology)

The number of local elements in `topology`.
"""
function nlocalelems end


"""
    (c1,c2,c3,c4) = vertex_coordinates(topology, elem)

The coordinates of the 4 vertices of element `elem`.
"""
function vertex_coordinates end

"""
    (opelem, opface, reversed) = opposing_face(topology, elem, face)

The opposing face of face number `face` of element `elem` in `topology`.

- `opelem` is the opposing element number, 0 for a boundary, negative for a ghost element
- `opface` is the opposite face number, or boundary face number if a boundary
- `reversed` indicates whether the opposing face has the opposite orientation.
"""
function opposing_face end

"""
    i,j = face_node_index(face, Nq, q, reversed=false)

The node indices of the `q`th node on face `face`, where `Nq` is the number of
face nodes in each direction.
"""
@inline function face_node_index(face, Nq, q, reversed = false)
    if reversed
        q = Nq - q + 1
    end
    if face == 1
        return q, 1
    elseif face == 2
        return Nq, q
    elseif face == 3
        return Nq - q + 1, Nq
    else
        return 1, Nq - q + 1
    end
end

"""
    i,j = vertex_node_index(vertex_num, Nq)

The node indices of `vertex_num`, where `Nq` is the number of face nodes in
each direction.
"""
function vertex_node_index(vertex_num, Nq)
    if vertex_num == 1
        return 1, 1
    elseif vertex_num == 2
        return Nq, 1
    elseif vertex_num == 3
        return Nq, Nq
    else
        return 1, Nq
    end
end


"""
    interior_faces(topology::AbstractTopology)

An iterator over the interior faces of `topology`. Each element of the iterator
is a 5-tuple the form

    (elem1, face1, elem2, face2, reversed)

where `elemX, faceX` are the element and face numbers, and `reversed` indicates
whether they have opposing orientations.
"""
function interior_faces(topology)
    InteriorFaceIterator(topology)
end

struct InteriorFaceIterator{T <: AbstractTopology}
    topology::T
end


"""
    ghost_faces(topology::AbstractTopology)

An iterator over the ghost faces of `topology`. Each element of the iterator
is a 5-tuple the form

    (elem1, face1, elem2, face2, reversed)

where `elemX, faceX` are the element and face numbers, and `reversed` indicates
whether they have opposing orientations.
"""
function ghost_faces(topology)
    GhostFaceIterator(topology)
end

struct GhostFaceIterator{T <: AbstractTopology}
    topology::T
end


"""
    boundary_tags(topology)

A `Tuple` or `NamedTuple` of the boundary tags of the topology. A boundary tag
is an integer that uniquely identifies a boundary.
"""
function boundary_tags end

"""
    boundary_tag(topology, name::Symbol)

The boundary tag of the topology for boundary name `name`. A boundary tag
is an integer that uniquely identifies a boundary.
"""
function boundary_tag end

"""
    boundary_faces(topology, boundarytag)

An iterator over the faces of `topology` which face the boundary with tag
`boundarytag`. Each element of the iterator is an `(elem, face)` pair.
"""
function boundary_faces end

struct Vertex{T <: AbstractTopology, V}
    topology::T
    num::V
end
Base.eltype(::Type{<:Vertex}) = Tuple{Int, Int}

# XXX: need to distinguish between real and ghost vertices
"""
    vertices(topology)

An iterator over the unique (shared) vertices of the topology `topology`.
Each vertex is an iterator over `(element, vertex_number)` pairs.
"""
function vertices(topology)
    VertexIterator(topology)
end
struct VertexIterator{T <: AbstractTopology}
    topology::T
end

"""
    interior_vertices(topology)

An iterator over the interior vertices of `topology`. Each vertex is an
iterator over `(element, vertex_number)` pairs.
"""
function interior_vertices(topology)
    InteriorVertexIterator(topology)
end
struct InteriorVertexIterator{T <: AbstractTopology}
    topology::T
end

"""
    ghost_vertices(topology)

An iterator over the ghost vertices of `topology`. Each vertex is an
iterator over `(element, vertex_number)` pairs.
"""
function ghost_vertices(topology)
    GhostVertexIterator(topology)
end
struct GhostVertexIterator{T <: AbstractTopology}
    topology::T
end


include("interval.jl")
include("topology2d.jl")
include("dtopology2d.jl")

# deprecate
const boundaries = boundary_tags
const GridTopology = Topology2D{<:Meshes.RectangleMesh}
GridTopology(mesh::Meshes.RectangleMesh) = Topology2D(mesh)

const Grid2DTopology = Topology2D


end # module
