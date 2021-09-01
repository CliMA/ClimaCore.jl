"""
    TensorProductTopology(n)

A tensor-product topology of `n` unstructured elements.
"""
struct TensorProductTopology{M <: TensorProductMesh, B} <: AbstractTopology
    mesh::M
    boundaries::B
end

"""
    TensorProductTopology(n)

A tensor-product topology of `n` unstructured elements
"""
function TensorProductTopology(mesh::TensorProductMesh)
    x1boundary = mesh.domain.x1boundary
    x2boundary = mesh.domain.x2boundary
    boundaries = if isnothing(x1boundary)
        if isnothing(x2boundary)
            NamedTuple()
        else
            NamedTuple{x2boundary}((3, 4))
        end
    else
        if isnothing(x2boundary)
            NamedTuple{x1boundary}((1, 2))
        else
            NamedTuple{(x1boundary..., x2boundary...)}((1, 2, 3, 4))
        end
    end
    TensorProductTopology(mesh, boundaries)
end

function Base.show(io::IO, topology::TensorProductTopology)
    print(io, "TensorProductTopology on ", topology.mesh)
end
domain(topology::TensorProductTopology) = topology.mesh.domain

function nlocalelems(topology::TensorProductTopology)
    n1 = topology.mesh.n1
    n2 = topology.mesh.n2
    return n1 * n2
end
eachslabindex(topology::TensorProductTopology) = 1:nlocalelems(topology)

function vertex_coordinates(topology::TensorProductTopology, elem::Integer)
    @assert 1 <= elem <= nlocalelems(topology)

    # convert to 0-based indices
    mesh = topology.mesh
    n1 = mesh.n1
    n2 = mesh.n2
    coordinates = mesh.coordinates

    z2, z1 = fldmod(elem - 1, n1)

    c1 = coordinates[z1 * (n2 + 1) + (z2 + 1)]
    c2 = coordinates[(z1 + 1) * (n2 + 1) + (z2 + 1)]
    c3 = coordinates[z1 * (n2 + 1) + (z2 + 2)]
    c4 = coordinates[(z1 + 1) * (n2 + 1) + (z2 + 2)]

    return (c1, c2, c3, c4)
end

function opposing_face(
    topology::TensorProductTopology,
    elem::Integer,
    face::Integer,
)
    @assert 1 <= elem <= nlocalelems(topology)
    @assert 1 <= face <= 4

    return topology.mesh.faces[(elem - 1) * 4 + face][3:5]
end

# InteriorFaceIterator
function Base.length(
    fiter::InteriorFaceIterator{T},
) where {T <: TensorProductTopology}
    topology = fiter.topology
    mesh = topology.mesh
    n1 = mesh.n1
    n2 = mesh.n2
    x1periodic = isnothing(mesh.domain.x1boundary)
    x2periodic = isnothing(mesh.domain.x2boundary)
    return (x1periodic ? n1 : n1 - 1) * n2 + n1 * (x2periodic ? n2 : n2 - 1)
end

function Base.iterate(
    fiter::InteriorFaceIterator{T},
    (d, z1, z2) = (1, 0, 0),
) where {T <: TensorProductTopology}
    # iteration state (major first)
    #  - d ∈ (1,2): face direction
    #  - z1 ∈ 0:n1-1: 0-based face index in direction 1
    #  - z2 ∈ 0:n2-1: 0-based face index in direction 2

    topology = fiter.topology
    mesh = topology.mesh
    n1 = mesh.n1
    n2 = mesh.n2
    x1periodic = isnothing(mesh.domain.x1boundary)
    x2periodic = isnothing(mesh.domain.x2boundary)

    # skip boundary faces
    if d == 1 && z1 == 0 && !x1periodic
        d = 2
    end
    if d == 2 && z2 == 0 && !x2periodic
        d = 1
        z1 += 1
        if z1 >= n1
            z1 = 0
            z2 += 1
            if !x1periodic
                d = 2
            end
        end
    end

    if z2 >= n2
        return nothing
    end

    if d == 1
        y1 = z1 == 0 ? n1 - 1 : z1 - 1
        y2 = z2
    else
        y1 = z1
        y2 = z2 == 0 ? n2 - 1 : z2 - 1
    end

    elem1 = z2 * n1 + z1 + 1
    elem2 = y2 * n1 + y1 + 1
    if d == 1
        nextstate = (2, z1, z2)
    else
        z1 += 1
        if z1 == n1
            z1 = 0
            z2 += 1
        end
        nextstate = (1, z1, z2)
    end
    if d == 1
        return (elem1, 1, elem2, 2, false), nextstate
    else
        return (elem1, 3, elem2, 4, false), nextstate
    end
end

# BoundaryFaceIterator
function boundary_names(topology::TensorProductTopology)
    x1boundary = topology.mesh.domain.x1boundary
    x2boundary = topology.mesh.domain.x2boundary
    if isnothing(x1boundary)
        isnothing(x2boundary) ? () : x2boundary
    else
        isnothing(x2boundary) ? x1boundary : (x1boundary..., x2boundary...)
    end
end

function boundary_tag(topology::TensorProductTopology, name::Symbol)
    x1boundary = topology.mesh.domain.x1boundary
    x2boundary = topology.mesh.domain.x2boundary
    if !isnothing(x1boundary)
        x1boundary[1] == name && return 1
        x1boundary[2] == name && return 2
    end
    if !isnothing(x2boundary)
        x2boundary[1] == name && return 3
        x2boundary[2] == name && return 4
    end
    error("Invalid boundary name")
end

function boundaries(topology::TensorProductTopology)
    return topology.boundaries
end

function Base.length(
    bfiter::BoundaryFaceIterator{T},
) where {T <: TensorProductTopology}
    boundary = bfiter.boundary
    topology = bfiter.topology
    if boundary in (1, 2)
        if isnothing(topology.mesh.domain.x1boundary)
            return 0
        else
            return topology.mesh.n2
        end
    end
    if boundary in (3, 4)
        if isnothing(topology.mesh.domain.x2boundary)
            return 0
        else
            return topology.mesh.n1
        end
    end
end

function Base.iterate(
    bfiter::BoundaryFaceIterator{T},
) where {T <: TensorProductTopology}
    boundary = bfiter.boundary
    topology = bfiter.topology
    if boundary in (1, 2) && isnothing(topology.mesh.domain.x1boundary)
        return nothing
    end
    if boundary in (3, 4) && isnothing(topology.mesh.domain.x2boundary)
        return nothing
    end
    Base.iterate(bfiter, 0)
end

function Base.iterate(
    bfiter::BoundaryFaceIterator{T},
    z,
) where {T <: TensorProductTopology}
    boundary = bfiter.boundary
    topology = bfiter.topology
    mesh = topology.mesh
    n1 = mesh.n1
    n2 = mesh.n2
    if boundary == 1
        z >= n2 && return nothing
        elem = z * n1 + 1
    elseif boundary == 2
        z >= n2 && return nothing
        elem = z * n1 + n1
    elseif boundary == 3
        z >= n1 && return nothing
        elem = z + 1
    elseif boundary == 4
        z >= n1 && return nothing
        elem = (n2 - 1) * n1 + z + 1
    end
    return (elem, boundary), z + 1
end

# VertexIterator
function Base.length(
    viter::VertexIterator{T},
) where {T <: TensorProductTopology}
    topology = viter.topology
    mesh = topology.mesh
    n1 = mesh.n1
    n2 = mesh.n2
    x1periodic = isnothing(mesh.domain.x1boundary)
    x2periodic = isnothing(mesh.domain.x2boundary)
    nv1 = x1periodic ? n1 : n1 + 1
    nv2 = x2periodic ? n2 : n2 + 1
    return nv1 * nv2
end

function Base.iterate(
    viter::VertexIterator{T},
    (z1, z2) = (0, 0),
) where {T <: TensorProductTopology}
    topology = viter.topology
    mesh = topology.mesh
    n1 = mesh.n1
    n2 = mesh.n2
    x1periodic = isnothing(mesh.domain.x1boundary)
    x2periodic = isnothing(mesh.domain.x2boundary)
    nv1 = x1periodic ? n1 : n1 + 1 # unique vertices in x1 direction
    nv2 = x2periodic ? n2 : n2 + 1 # unique vertices in x2 direction

    if z2 >= nv2
        return nothing
    end
    vertex = Vertex(topology, (z1, z2))
    z1 += 1
    if z1 >= nv1
        nextstate = (0, z2 + 1)
    else
        nextstate = (z1, z2)
    end
    return vertex, nextstate
end

# Vertex
function Base.length(vertex::Vertex{T}) where {T <: TensorProductTopology}
    topology = vertex.topology
    mesh = topology.mesh
    n1 = mesh.n1
    n2 = mesh.n2
    x1periodic = isnothing(mesh.domain.x1boundary)
    x2periodic = isnothing(mesh.domain.x2boundary)

    z1, z2 = vertex.num

    k1 = !x1periodic && (z1 == 0 || z1 == n1) ? 1 : 2
    k2 = !x2periodic && (z2 == 0 || z2 == n2) ? 1 : 2
    return k1 * k2
end

function Base.iterate(
    vertex::Vertex{T},
    vert = 0,
) where {T <: TensorProductTopology}
    # iterator of (element, vertnum) that share global vertex
    topology = vertex.topology
    mesh = topology.mesh
    n1 = mesh.n1
    n2 = mesh.n2
    x1periodic = isnothing(mesh.domain.x1boundary)
    x2periodic = isnothing(mesh.domain.x2boundary)
    nv1 = x1periodic ? n1 : n1 + 1
    nv2 = x2periodic ? n2 : n2 + 1
    z1, z2 = vertex.num

    vert += 1
    # at the boundary, skip non-existent elements
    if !x1periodic
        if z1 == 0 && (vert == 2 || vert == 4)
            vert += 1
        end
        if z1 == n1 && (vert == 1 || vert == 3)
            vert += 1
        end
    end
    if !x2periodic
        if z2 == 0 && (vert == 3 || vert == 4)
            vert += 2
        end
        if z2 == n2 && (vert == 1 || vert == 2)
            vert += 2
        end
    end

    if vert > 4
        return nothing
    end

    if vert == 2 || vert == 4
        z1 = mod(z1 - 1, nv1)
    end
    if vert == 3 || vert == 4
        z2 = mod(z2 - 1, nv2)
    end
    elem = z2 * n1 + z1 + 1
    return (elem, vert), vert
end
