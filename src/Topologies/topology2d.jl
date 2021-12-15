

"""
    Topology2D(mesh::AbstractMesh2D, elemorder=Mesh.elements(mesh))

This is a generic non-distributed topology for 2D meshes. `elemorder` is a
vector or other linear ordering of the `Mesh.elements(mesh)`.
"""
struct Topology2D{M <: Meshes.AbstractMesh{2}, EO, OI, BF} <: AbstractTopology
    "mesh on which the topology is constructed"
    mesh::M
    "`elemorder[e]` should give the `e`th element "
    elemorder::EO
    "the inverse of `elemorder`: `orderindex[elemorder[e]] == e`"
    orderindex::OI
    "a vector of all unique internal faces, (e, face, o, oface, reversed)"
    internal_faces::Vector{Tuple{Int, Int, Int, Int, Bool}}
    "the collection of all (e,vert) pairs"
    vertices::Vector{Tuple{Int, Int}}
    "the index in `vertices` of the first `(e,vert)` pair for all unique vertices"
    vertex_offset::Vector{Int}
    "a NamedTuple of vectors of `(e,face)` "
    boundaries::BF
end



function Topology2D(
    mesh::Meshes.AbstractMesh{2},
    elemorder = Meshes.elements(mesh),
    orderindex = Meshes.linearindices(elemorder),
)
    internal_faces = Tuple{Int, Int, Int, Int, Bool}[]
    vertices = sizehint!(Tuple{Int, Int}[], length(elemorder) * 4)
    vertex_offset = sizehint!(Int[1], length(elemorder))
    boundaries = NamedTuple(
        boundary_name => Tuple{Int, Int}[] for
        boundary_name in unique(Meshes.boundary_names(mesh))
    )
    for (e, elem) in enumerate(elemorder)
        for face in 1:4
            if Meshes.is_boundary_face(mesh, elem, face)
                boundary_name = Meshes.boundary_face_name(mesh, elem, face)
                push!(boundaries[boundary_name], (e, face))
            else
                opelem, opface, reversed =
                    Meshes.opposing_face(mesh, elem, face)
                o = orderindex[opelem]
                if (o, opface) < (e, face)
                    # new internal face
                    push!(internal_faces, (e, face, o, opface, reversed))
                end
            end
        end
        for vert in 1:4
            if !any(Meshes.SharedVertices(mesh, elem, vert)) do (velem, vvert)
                (orderindex[velem], vvert) < (e, vert)
            end
                # new vertex
                for (velem, vvert) in Meshes.SharedVertices(mesh, elem, vert)
                    o = orderindex[velem]
                    push!(vertices, (o, vvert))
                end
                push!(vertex_offset, length(vertices) + 1)
            end
        end
    end
    @assert length(vertices) == 4 * length(elemorder) == vertex_offset[end] - 1
    return Topology2D(
        mesh,
        elemorder,
        orderindex,
        internal_faces,
        vertices,
        vertex_offset,
        boundaries,
    )
end

domain(topology::Topology2D) = domain(topology.mesh)
nlocalelems(topology::Topology2D) = length(topology.elemorder)
vertex_coordinates(topology::Topology2D, e::Int) =
    ntuple(4) do vert
        Meshes.coordinates(topology.mesh, topology.elemorder[e], vert)
    end


function opposing_face(topology::Topology2D, e::Int, face::Int)
    mesh = topology.mesh
    elem = topology.elemorder[e]
    if Meshes.is_boundary_face(mesh, elem, face)
        boundary_name = Meshes.boundary_face_name(mesh, elem, face)
        b = findfirst(==(boundary_name), keys(topology.boundaries))
        return (0, b, false)
    end
    opelem, opface, reversed = Meshes.opposing_face(mesh, elem, face)
    return (topology.orderindex[opelem], opface, reversed)
end
interior_faces(topology::Topology2D) = topology.internal_faces

Base.length(vertiter::VertexIterator{<:Topology2D}) =
    length(vertiter.topology.vertex_offset) - 1
Base.eltype(::VertexIterator{T}) where {T <: Topology2D} = Vertex{T, Int}
function Base.iterate(vertiter::VertexIterator{<:Topology2D}, uvert = 1)
    topology = vertiter.topology
    if uvert >= length(topology.vertex_offset)
        return nothing
    end
    return Vertex(topology, uvert), uvert + 1
end

Base.length(vertex::Vertex{<:Topology2D}) =
    vertex.topology.vertex_offset[vertex.num + 1] -
    vertex.topology.vertex_offset[vertex.num]
Base.eltype(vertex::Vertex{<:Topology2D}) = eltype(vertex.topology.vertices)
function Base.iterate(
    vertex::Vertex{<:Topology2D},
    idx = vertex.topology.vertex_offset[vertex.num],
)
    if idx >= vertex.topology.vertex_offset[vertex.num + 1]
        return nothing
    end
    return vertex.topology.vertices[idx], idx + 1
end

boundary_names(topology::Topology2D) = keys(topology.boundaries)
boundary_tags(topology::Topology2D) = NamedTuple{boundary_names(topology)}(
    ntuple(i -> i, length(topology.boundaries)),
)
boundary_tag(topology::Topology2D, boundary_name::Symbol) =
    findfirst(==(boundary_name), boundary_names(topology))

boundary_faces(topology::Topology2D, boundary) = topology.boundaries[boundary]
