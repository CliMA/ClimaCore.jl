"""
    ClimateMachineCore.Topologies

Objects describing the horizontal connections between elements.

All elements are quadrilaterals, using the face and vertex numbering
convention from [p4est](https://p4est.github.io/papers/BursteddeWilcoxGhattas11.pdf):
```
          4
      3-------4
 ^    |       |
 |  1 |       | 2
x2    |       |
      1-------2
          3
        x1-->
```
"""
module Topologies

import StaticArrays: SVector

import ..Domains: EquispacedRectangleDiscretization, coordinate_type

# TODO: seperate types for MPI/non-MPI topologies
"""
   AbstractTopology

Subtypes of `AbstractHorizontalTopology` define connectiveness of
discretized domain elements in the horizontal domain.
"""
abstract type AbstractTopology end

"""
    domain(topology)

The `domain` underlying the topology.
"""
function domain end

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
    boundary_faces(topology, boundary)

An iterator over the faces of `topology` which face boundary `boundary`.
Each element of the iterator is an `(elem, face)` pair.
"""
function boundary_faces(topology, boundary)
    BoundaryFaceIterator(topology, boundary)
end

struct BoundaryFaceIterator{T}
    topology::T
    boundary::Int
end


"""
    vertices(topology)

An iterator over the unique (shared) vertices of the topology `topology`.
Each vertex is an iterator over `(element, vertexnumber)` pairs.
"""
function vertices(topology)
    VertexIterator(topology)
end
struct VertexIterator{T <: AbstractTopology}
    topology::T
end
struct Vertex{T <: AbstractTopology, V}
    topology::T
    id::V
end

# implementations
include("grid.jl")

end # module
