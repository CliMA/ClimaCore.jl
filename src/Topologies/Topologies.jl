module Topologies

import ClimaComms, Adapt

import ..ClimaCore
import ..Utilities: Cache, AutoBroadcaster, nested_broadcast, return_type
import ..Geometry
import ..Domains: Domains, coordinate_type
import ..Meshes: Meshes, domain, coordinates
import ..DataLayouts
import ..slab, ..column, ..level

import ..DeviceSideDevice, ..DeviceSideContext

"""
    AbstractTopology

Abstract supertype of topologies, which define the connectivity of the elements of
a mesh. Subtypes: [`IntervalTopology`](@ref) and [`Topology2D`](@ref).

Subtypes implement the following interface:

  - [`nelems`](@ref)
  - [`domain(topology::AbstractTopology)`](@ref)
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
  - [`local_neighboring_elements`](@ref)
  - [`ghost_neighboring_elements`](@ref)
  - [`local_vertices`](@ref)
  - [`ghost_vertices`](@ref)
  - [`neighbors`](@ref)
  - [`boundary_tags`](@ref)
  - [`boundary_tag`](@ref)
  - [`boundary_faces`](@ref)
"""
abstract type AbstractTopology end

ClimaComms.context(topology::AbstractTopology) = topology.context
ClimaComms.device(topology::AbstractTopology) =
    ClimaComms.device(ClimaComms.context(topology))
ClimaComms.array_type(topology::AbstractTopology) =
    ClimaComms.array_type(ClimaComms.device(topology))

function Base.summary(io::IO, topology::AbstractTopology)
    print(io, nameof(typeof(topology)))
end

# TODO: move this to ClimaComms
function print_device(io::IO, device::ClimaComms.AbstractDevice)
    print(io, nameof(typeof(device)))
end

function print_context(io::IO, context::ClimaComms.SingletonCommsContext)
    print(io, "SingletonCommsContext using ")
    print_device(io, context.device)
end

function print_context(io::IO, context::ClimaComms.MPICommsContext)
    print(io, "MPICommsContext with ", ClimaComms.nprocs(context), " processes")
    print(io, " using ")
    print_device(io, context.device)
end

abstract type AbstractDistributedTopology <: AbstractTopology end

coordinate_type(topology::AbstractTopology) = coordinate_type(domain(topology))

function domain end

"""
    mesh(topology)

Return the mesh underlying `topology`.
"""
function mesh end

"""
    nelems(topology)

Return the total number of elements in `topology`.
"""
function nelems end

"""
    nlocalelems(topology)

Return the number of elements local to this process in `topology`.
"""
function nlocalelems end

"""
    nneighbors(topology)

Return the number of neighboring processes of this process in `topology`.
"""
function nneighbors end
nneighbors(::AbstractTopology) = 0

"""
    nsendelems(topology)

Return the number of elements this process sends to its neighbors in `topology`.
"""
function nsendelems end
nsendelems(::AbstractTopology) = 0
nsendelems(::AbstractTopology, _) = 0

"""
    nghostelems(topology)

Return the number of ghost elements in `topology`.
"""
function nghostelems end
nghostelems(::AbstractTopology) = 0
nghostelems(::AbstractTopology, _) = 0

"""
    localelemindex(topology, elem)

Return the local index of element `elem`; used by distributed topologies.
"""
function localelemindex end

"""
    vertex_coordinates(topology, elem)

Return a tuple of the coordinates of the vertices of element `elem` (two in 1D, four
in 2D).
"""
function vertex_coordinates end

"""
    opposing_face(topology, elem, face)

Return the face `(opelem, opface, reversed)` opposite face number `face` of element
`elem` in `topology`.

  - `opelem`: The opposing element number; 0 for a boundary, negative for a ghost
    element.
  - `opface`: The opposing face number, or the boundary face number for a boundary.
  - `reversed`: Whether the opposing face has the opposite orientation.
"""
function opposing_face end

"""
    face_node_index(face, Nq, q, reversed = false)

Return the node indices `(i, j)` of the `q`th node on face `face`, where `Nq` is the
number of nodes in each direction. If `reversed`, count from the other end of the
face.
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

Return an iterator over the interior faces of `topology`. Each item is a 5-tuple of
the form

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

Return an iterator over the ghost faces of `topology`. Each item is a 5-tuple of the
form

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
    local_neighboring_elements(topology::AbstractTopology, lidx::Integer)

Return an iterator over the local element indices of the local elements that are
neighbors of the local element `lidx` in `topology`, excluding `lidx` itself.
"""
function local_neighboring_elements end

"""
    ghost_neighboring_elements(topology::AbstractTopology, lidx::Integer)

Return an iterator over the receive buffer indices (`ridx`) of the ghost elements
that are neighbors of the local element `lidx` in `topology`.
"""
function ghost_neighboring_elements end

"""
    vertex_node_index(vertex_num, Nq)

Return the node indices `(i, j)` of vertex `vertex_num`, where `Nq` is the number of
nodes in each direction.
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
    Topologies.VertexIterator

Iterator over the unique (shared) vertices of a topology. Each item is a `Vertex`,
which is itself an iterator over the `(element, vertex)` pairs that share it.
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

Return an iterator over the interior vertices of `topology`. Each vertex is an
iterator over `(lidx, vert)` pairs.
"""
function local_vertices end

"""
    ghost_vertices(topology)

Return an iterator over the ghost vertices of `topology`. Each vertex is an iterator
over `(isghost, idx, vert)` triples, where `idx` is a local index `lidx` if
`isghost` is `false` and a receive buffer index `ridx` otherwise.
"""
function ghost_vertices end

"""
    neighbors(topology)

Return a vector of the PIDs of the neighboring processes of this process.
"""
function neighbors end
neighbors(::AbstractTopology) = Int[]

"""
    boundary_tags(topology)

Return a `Tuple` or `NamedTuple` of the boundary tags of `topology`. A boundary tag
is an integer that uniquely identifies a boundary.
"""
function boundary_tags end

"""
    boundary_tag(topology, name::Symbol)

Return the boundary tag of `topology` for the boundary named `name`. A boundary tag
is an integer that uniquely identifies a boundary.
"""
function boundary_tag end

"""
    boundary_faces(topology, boundarytag)

Return an iterator over the faces of `topology` on the boundary with tag
`boundarytag`. Each item is an `(elem, face)` pair.
"""
function boundary_faces end

# Topologies API implementations
include("interval.jl")
include("topology2d.jl")

include("dss_transform.jl")
include("dss.jl")

const DistributedTopology2D = Topology2D

end # module
