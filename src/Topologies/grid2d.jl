"""
    Grid2DTopology{M <: Mesh2D} <: AbstractTopology

An unstructured 2D conformal mesh of elements.
"""
struct Grid2DTopology{M, D} <: AbstractTopology
    mesh::M
    domain::D
end

Grid2DTopology(mesh::Mesh2D, domain::Unstructured2DDomain) =
    Grid2DTopology{typeof(mesh), typeof(domain)}(mesh, domain)

domain(topology::Grid2DTopology) = topology.domain #Unstructured2DDomain()

function nlocalelems(topology::Grid2DTopology)
    return topology.mesh.nelems
end

eachslabindex(topology::Grid2DTopology) = 1:nlocalelems(topology)

function vertex_coordinates(topology::Grid2DTopology, elem::Integer)
    @assert 1 <= elem <= nlocalelems(topology)
    coords = topology.mesh.coordinates
    verts = view(topology.mesh.elem_verts, elem, :)
    dim = size(coords, 2)
    if dim == 2
        return (
            Geometry.Cartesian2DPoint(coords[verts[1], 1], coords[verts[1], 2]),
            Geometry.Cartesian2DPoint(coords[verts[2], 1], coords[verts[2], 2]),
            Geometry.Cartesian2DPoint(coords[verts[3], 1], coords[verts[3], 2]),
            Geometry.Cartesian2DPoint(coords[verts[4], 1], coords[verts[4], 2]),
        )
    else
        return (
            Geometry.Cartesian123Point(
                coords[verts[1], 1],
                coords[verts[1], 2],
                coords[verts[1], 3],
            ),
            Geometry.Cartesian123Point(
                coords[verts[2], 1],
                coords[verts[2], 2],
                coords[verts[2], 3],
            ),
            Geometry.Cartesian123Point(
                coords[verts[3], 1],
                coords[verts[3], 2],
                coords[verts[3], 3],
            ),
            Geometry.Cartesian123Point(
                coords[verts[4], 1],
                coords[verts[4], 2],
                coords[verts[4], 3],
            ),
        )
    end
end

function opposing_face(topology::Grid2DTopology, elem::Integer, face::Integer)
    @assert 1 <= elem <= nlocalelems(topology)
    @assert 1 <= face <= 4

    face_neighbors = topology.mesh.face_neighbors
    fc = topology.mesh.elem_faces[elem, face]

    if face_neighbors[fc, 1] == elem
        opelem = face_neighbors[fc, 3]
        opface = face_neighbors[fc, 4]
    elseif face_neighbors[fc, 3] == elem
        opelem = face_neighbors[fc, 1]
        opface = face_neighbors[fc, 2]
    else
        error("opposing_face: fatal error in connectivity information")
    end
    if opelem == 0
        opface = face
    end
    reversed = face_neighbors[5] == -1
    return opelem, opface, reversed
end

# InteriorFaceIterator
function Base.length(fiter::InteriorFaceIterator{T}) where {T <: Grid2DTopology}
    topology = fiter.topology
    mesh = topology.mesh
    loc = findfirst(mesh.boundary_tags .== 0) # interior tag is 0
    if isnothing(loc)
        return 0
    else
        return (
            mesh.face_boundary_offset[loc + 1] - mesh.face_boundary_offset[loc]
        )
    end
end

function Base.iterate(
    fiter::InteriorFaceIterator{T},
    faceno = 1,
) where {T <: Grid2DTopology}
    mesh = fiter.topology.mesh
    face_neighbors = mesh.face_neighbors
    face_boundary = mesh.face_boundary
    face_boundary_offset = mesh.face_boundary_offset
    boundary_tags = mesh.boundary_tags

    if boundary_tags[1] ≠ 0
        return nothing
    end
    st, en = face_boundary_offset[1], face_boundary_offset[2]
    nfcs = en - st
    if faceno > nfcs
        return nothing
    end
    face = face_boundary[faceno]
    reversed = face_neighbors[face, 5] == -1
    return tuple(face_neighbors[face, 1:4]..., reversed), faceno + 1
end

# BoundaryFaceIterator
function boundary_names(topology::Grid2DTopology)
    boundary_tag_names = topology.mesh.boundary_tag_names
    if boundary_tag_names[1] == :interior
        return tuple(boundary_tag_names[2:end]...)
    else
        return boundary_tag_names
    end
end

function boundary_tag(topology::Grid2DTopology, name::Symbol)
    boundary_tag_names = topology.mesh.boundary_tag_names
    boundary_tags = topology.mesh.boundary_tags
    location = findfirst(boundary_tag_names .== name)
    if isnothing(location)
        error("Invalid boundary name")
    else
        return boundary_tags[location]
    end
end

function boundaries(topology::Grid2DTopology)
    boundary_tags = topology.mesh.boundary_tags
    if boundary_tags[1] == 0
        return tuple(boundary_tags[2:end]...)
    else
        return tuple(boundary_tags...)
    end
end

function Base.length(
    bfiter::BoundaryFaceIterator{T},
) where {T <: Grid2DTopology}
    boundary = bfiter.boundary
    topology = bfiter.topology
    face_boundary_offset = topology.mesh.face_boundary_offset
    boundary_tags = topology.mesh.boundary_tags
    location = findfirst(boundary_tags .== boundary)
    if isnothing(location)
        return 0
    end
    return face_boundary_offset[location + 1] - face_boundary_offset[location]
end

function Base.iterate(
    bfiter::BoundaryFaceIterator{T},
    faceno = 1,
) where {T <: Grid2DTopology}
    boundary = bfiter.boundary
    topology = bfiter.topology
    mesh = topology.mesh
    face_neighbors = mesh.face_neighbors
    face_boundary = mesh.face_boundary
    face_boundary_offset = mesh.face_boundary_offset
    boundary_tags = mesh.boundary_tags
    location = findfirst(boundary_tags .== boundary)

    if isnothing(location)
        return nothing
    end
    st, en = face_boundary_offset[location:(location + 1)]
    nbndry = en - st
    if faceno < 1 || faceno > nbndry
        return nothing
    end
    face = face_boundary[faceno + st - 1]
    if face_neighbors[face, 1] == 0
        return tuple(face_neighbors[face, 3:4]...), faceno + 1
    else
        return tuple(face_neighbors[face, 1:2]...), faceno + 1
    end
end

# VertexIterator
function Base.length(viter::VertexIterator{T}) where {T <: Grid2DTopology}
    return length(viter.topology.mesh.unique_verts)
end

function Base.iterate(
    viter::VertexIterator{T},
    uvertno = 1,
) where {T <: Grid2DTopology}
    topology = viter.topology
    nuverts = length(viter.topology.mesh.unique_verts)
    if uvertno > 0 && uvertno ≤ nuverts
        return Vertex(topology, uvertno), uvertno + 1
    else
        return nothing
    end
end

# Vertex
function Base.length(vertex::Vertex{T}) where {T <: Grid2DTopology}
    mesh = vertex.topology.mesh
    vtno = vertex.num
    offset = mesh.uverts_offset
    return div(offset[vtno + 1] - offset[vtno], 2)
end

function Base.iterate(vertex::Vertex{T}, iterno = 1) where {T <: Grid2DTopology}
    # iterator of (element, vertnum) that share global vertex
    mesh = vertex.topology.mesh
    vtno = vertex.num
    offset = mesh.uverts_offset
    uverts_conn = mesh.uverts_conn
    st, en = offset[vtno], offset[vtno + 1]
    niter = div(en - st, 2)

    if iterno < 1 || iterno > niter
        return nothing
    end
    loc = (iterno - 1) * 2 + st
    return tuple(uverts_conn[loc:(loc + 1)]...), iterno + 1
end
