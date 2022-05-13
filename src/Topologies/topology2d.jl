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
    neighbor_elem::Vector{Int}
    neighbor_elem_offset::Vector{Int}
    "a NamedTuple of vectors of `(e,face)` "
    boundaries::BF
end

function Base.show(io::IO, topology::Topology2D)
    print(io, "Topology2D on ", topology.mesh)
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
        boundary_name in Meshes.boundary_names(mesh)
    )

    temp_neighbor_elemset = SortedSet{Int}()
    neighbor_elem = Int[]
    neighbor_elem_offset = Int[1]

    for (e, elem) in enumerate(elemorder)
        empty!(temp_neighbor_elemset)
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
            for (velem, vvert) in Meshes.SharedVertices(mesh, elem, vert)
                o = orderindex[velem]
                if o != e
                    push!(temp_neighbor_elemset, o)
                end
            end
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
        append!(neighbor_elem, temp_neighbor_elemset)
        push!(neighbor_elem_offset, length(neighbor_elem) + 1)
    end
    @assert length(vertices) == 4 * length(elemorder) == vertex_offset[end] - 1
    return Topology2D(
        mesh,
        elemorder,
        orderindex,
        internal_faces,
        vertices,
        vertex_offset,
        neighbor_elem,
        neighbor_elem_offset,
        boundaries,
    )
end

domain(topology::Topology2D) = domain(topology.mesh)
nelems(topology::Topology2D) = nlocalelems(topology)
nlocalelems(topology::Topology2D) = length(topology.elemorder)
localelems(topology::Topology2D) = topology.elemorder
nghostelems(topology::Topology2D) = 0
ghostelems(topology::Topology2D) = ()

localelemindex(topology::Topology2D, elem) = topology.orderindex[elem]
coordinates(topology::Topology2D, e::Int, arg) =
    coordinates(topology.mesh, topology.elemorder[e], arg)
vertex_coordinates(topology::Topology2D, e::Int) =
    ntuple(4) do vert
        coordinates(topology, e, vert)
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
local_vertices(topology::Topology2D) =
    VertexIterator(topology.vertices, topology.vertex_offset)
ghost_faces(topology::Topology2D) = ()
ghost_vertices(topology::Topology2D) = ()

boundary_names(topology::Topology2D) = keys(topology.boundaries)
boundary_tags(topology::Topology2D) = NamedTuple{boundary_names(topology)}(
    ntuple(i -> i, length(topology.boundaries)),
)
boundary_tag(topology::Topology2D, boundary_name::Symbol) =
    findfirst(==(boundary_name), boundary_names(topology))

boundary_faces(topology::Topology2D, boundary) = topology.boundaries[boundary]

function local_neighboring_elements(topology::Topology2D, elem)
    return view(
        topology.neighbor_elem,
        topology.neighbor_elem_offset[elem]:(topology.neighbor_elem_offset[elem + 1] - 1),
    )
end
ghost_neighboring_elements(topology::Topology2D, elem) = ()
