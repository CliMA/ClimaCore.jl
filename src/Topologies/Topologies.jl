module Topologies

import ..Geometry
import ..Domains: Domains, coordinate_type
import ..Meshes: Meshes, domain, coordinates

"""
   AbstractTopology

Subtypes of `AbstractHorizontalTopology` define connectiveness of a
mesh in the horizontal domain.

# Interfaces

- [`nelems`](@ref)
- [`domain`](@ref)
- [`mesh`](@ref)
- [`nlocalelems`](@ref)
- [`nneighbors`](@ref)
- [`nsendelems`](@ref)
- [`nghostelems`](@ref)
- [`localelemindex`](@ref)
- [`vertex_coordinates`](@ref)
- [`opposing_face`](@ref)
- [`face_node_index`](@ref)
- [`interior_faces`](@ref)
- [`ghost_faces`](@ref)
- [`vertex_node_index`](@ref)
- [`vertices`](@ref)
- [`neighboring_elements`](@ref)
- [`local_vertices`](@ref)
- [`ghost_vertices`](@ref)
- [`neighbors`](@ref)
- [`boundary_tags`](@ref)
- [`boundary_tag`](@ref)
- [`boundary_faces`](@ref)

"""
abstract type AbstractTopology end
abstract type AbstractDistributedTopology <: AbstractTopology end

coordinate_type(topology::AbstractTopology) = coordinate_type(domain(topology))

"""
    domain(topology)

Returns the domain of the `topology` from the underlying `mesh`
"""
function domain end

"""
    mesh(topology)

Returns the mesh underlying the `topology`
"""
function mesh end

"""
    nelems(topology)

The total number of elements in `topology`.
"""
function nelems end

"""
    nlocalelems(topology)

The number of local elements in `topology`.
"""
function nlocalelems end

"""
    nneighbors(topology)

The number of neighbors of this process in `topology`.
"""
function nneighbors end
nneighbors(::AbstractTopology) = 0

"""
    nsendelems(topology)

The number of elements to send to neighbors in `topology`.
"""
function nsendelems end
nsendelems(::AbstractTopology) = 0
nsendelems(::AbstractTopology, _) = 0

"""
    nghostelems(topology)

The number of ghost elements in `topology`.
"""
function nghostelems end
nghostelems(::AbstractTopology) = 0
nghostelems(::AbstractTopology, _) = 0

"""
    localelemindex(topology, elem)

The local index for the specified element; useful for distributed topologies.
"""
function localelemindex end

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
    neighboring_elements(topology, elem)

The list of neighboring elements of element `elem` in `topology`.
"""
function neighboring_elements end

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
    vertices(topology)

An iterator over the unique (shared) vertices of the topology `topology`.
Each vertex returns a `Vertex` object, which is itself an iterator.
"""
struct VertexIterator{T}
    vertices::Vector{T}
    vertex_offset::Vector{Int}
end
Base.eltype(::Type{VertexIterator{T}}) where {T} = Vertex{T}
Base.eltype(::VertexIterator{T}) where {T} = Vertex{T}
Base.length(vertiter::VertexIterator{T}) where {T} =
    length(vertiter.vertex_offset) - 1

function Base.iterate(vertiter::VertexIterator, num = 1)
    if num >= length(vertiter.vertex_offset)
        return nothing
    end
    return Vertex(vertiter, num), num + 1
end


struct Vertex{T}
    vertiter::VertexIterator{T}
    num::Int
end
Base.eltype(::Type{Vertex{T}}) where {T} = T
Base.eltype(::Vertex{T}) where {T} = T
Base.length(vertex::Vertex{T}) where {T} =
    vertex.vertiter.vertex_offset[vertex.num + 1] -
    vertex.vertiter.vertex_offset[vertex.num]
function Base.iterate(
    vertex::Vertex,
    idx = vertex.vertiter.vertex_offset[vertex.num],
)
    if idx >= vertex.vertiter.vertex_offset[vertex.num + 1]
        return nothing
    end
    return vertex.vertiter.vertices[idx], idx + 1
end



"""
    local_vertices(topology)

An iterator over the interior vertices of `topology`. Each vertex is an
iterator over `(lidx, vert)` pairs.
"""
function local_vertices end

"""
    ghost_vertices(topology)

An iterator over the ghost vertices of `topology`. Each vertex is an
iterator over `(isghost, lidx/ridx, vert)` pairs.
"""
function ghost_vertices end

"""
    neighbors(topology)

Returns an array of the PIDs of the neighbors of this process.
"""
function neighbors end
neighbors(::AbstractTopology) = Int[]

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

# Topologies API implementations
include("interval.jl")
include("topology2d.jl")
include("dtopology2d.jl")

# deprecate
@deprecate boundaries(topology::AbstractTopology) boundary_tags(topology)
@deprecate GridTopology(mesh) Topology2D(mesh)
@deprecate Topology2D(mesh) Topology2D(mesh)

end # module
