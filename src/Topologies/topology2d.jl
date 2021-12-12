

"""
    Topology2D(mesh::AbstractMesh2D)



"""
struct Topology2D{M<:Meshes.AbstractMesh{2},EO,OI} <: AbstractTopology
    "mesh on which the topology is constructed"
    mesh::M
    "`elemorder[i]` should give the `i`th element"
    elemorder::EO
    "the inverse of `elemorder`: `orderindex[elemorder[i]] == i`"
    orderindex::OI
    internal_faces::Vector{Tuple{Int,Int,Int,Int,Bool}}
    vertices::Vector{Tuple{Int,Int}}
    vertex_offset::Vector{Int}
end



function Topology2D(mesh::Meshes.AbstractMesh{2}, elemorder=Meshes.elements(mesh), orderindex=Meshes.linearindices(elemorder))
    internal_faces = Tuple{Int,Int,Int,Int,Bool}[]
    vertices = sizehint!(Tuple{Int,Int}[], length(elemorder)*4)
    vertex_offset = sizehint!(Int[], length(elemorder))
    for (e,elem) in enumerate(elemorder)
        for face in 1:4
            if Meshes.is_boundary_face(mesh, elem, face)
                error("not yet implemented")
            end
            opelem, opface, reversed = Meshes.opposing_face(mesh, elem, face)
            o = orderindex[opelem]
            if (o, opface) < (e, face)
                # new internal face
                push!(internal_faces, (e, face, o, opface, reversed))
            end
        end
        for vert in 1:4
            if !any(Meshes.SharedVertices(mesh, elem, vert)) do (velem, vvert)
                    (orderindex[velem], vvert) < (e, vert)
                end
                # new vertex
                for (velem, vvert) in Meshes.SharedVertices(mesh, elem, vert)
                    o = orderindex[velem]
                    push!(vertices, (o,vvert))
                end
                push!(vertex_offset, length(vertices)+1)
            end
        end
    end
    return Topology2D(mesh, elemorder, orderindex, internal_faces, vertices, vertex_offset)
end

domain(topology::Topology2D) = topology.mesh.domain
nlocalelems(topology::Topology2D) = length(elemorder)
vertex_coordinates(topology::Topology2D, e::Int) =
    ntuple(4) do vert
        Meshes.coordinates(topology.mesh, topology.elemorder[e], vert)
    end

boundaries(topology::Topology2D) = () # TODO

opposing_face(topology::Topology2D, e::Int, face::Int) =
    Meshes.opposing_face(topology.mesh, topology.elemorder[e], face)
interior_faces(topology::Topology2D) = topology.internal_faces

function Base.iterate(vertiter::VertexIterator{<:Topology2D}, uvert=1)
    topology = vertiter.topology
    if uvert >= length(topology.vertex_offset)
        return nothing
    end
    return VertexIterator(topology, uvert), uvert+1
end
function Base.iterate(vertex::Vertex{<:Topology2D}, idx=vertex.topology.vertex_offset[vertex.num])
    if idx >= vertex.topology.vertex_offset[vertiter.num+1]
        return nothing
    end
    return vertex.topology.vertices[idx], idx+1
end




#  -
# 1. for each element:
#   - 4 opposing faces
#     interior: (elemno, faceno, reversed)
#     exterior: boundaryno, faceno
#   - 4 vertex numbers
#     uvertno
# 2. for each interior face
#    `face_verts`: (uvert1, uvert2) I don't think we actually need this?
#    `face_neighbors` [elem1, localface1, elem2, localface2, relative orientation]

# 3. for each boundary
#      for each face at boundary
#         (elem, face)
# 4. for each unique vertex, staggered array
#    - `unique_verts`: vertex number (not really used)
#    - `uverts_conn`: elemno, vertno
#    - `uverts_offset`: [1,....,length(uverts_conn)]
# 5.


