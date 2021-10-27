"""
    GridTopology(mesh)

A generic tensor-product topology defined on either an equispaced, regular rectangular
structured grid or irregular one.
"""
struct GridTopology{M <: AbstractMesh, B} <: AbstractTopology
    mesh::M
    boundaries::B
end
function GridTopology(mesh::M) where {M <: AbstractMesh}
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
    GridTopology(mesh, boundaries)
end



function Base.show(io::IO, topology::GridTopology{M}) where {M <: AbstractMesh}
    print(io, "GridTopology on ", topology.mesh)
end
domain(topology::GridTopology{M}) where {M <: AbstractMesh} =
    topology.mesh.domain

function nlocalelems(topology::GridTopology{M}) where {M <: AbstractMesh}
    n1 = topology.mesh.n1
    n2 = topology.mesh.n2
    return n1 * n2
end

eachslabindex(topology::GridTopology{M}) where {M <: AbstractMesh} =
    1:nlocalelems(topology)


# InteriorFaceIterator
function Base.length(
    fiter::InteriorFaceIterator{T},
) where {M <: AbstractMesh, T <: GridTopology{M}}
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
) where {M <: AbstractMesh, T <: GridTopology{M}}
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
        return (elem1, 4, elem2, 2, true), nextstate
    else
        return (elem1, 1, elem2, 3, true), nextstate
    end
end

# BoundaryFaceIterator
function boundary_names(topology::GridTopology{M}) where {M <: AbstractMesh}
    x1boundary = topology.mesh.domain.x1boundary
    x2boundary = topology.mesh.domain.x2boundary
    if isnothing(x1boundary)
        isnothing(x2boundary) ? () : x2boundary
    else
        isnothing(x2boundary) ? x1boundary : (x1boundary..., x2boundary...)
    end
end

function boundary_tag(
    topology::GridTopology{M},
    name::Symbol,
) where {M <: AbstractMesh}
    getproperty(topology.boundaries, name)
end

function boundaries(topology::GridTopology{M}) where {M <: AbstractMesh}
    return topology.boundaries
end

function Base.length(
    bfiter::BoundaryFaceIterator{T},
) where {M <: AbstractMesh, T <: GridTopology{M}}
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
) where {M <: AbstractMesh, T <: GridTopology{M}}
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
) where {M <: AbstractMesh, T <: GridTopology{M}}
    boundary = bfiter.boundary
    topology = bfiter.topology
    mesh = topology.mesh
    n1 = mesh.n1
    n2 = mesh.n2
    if boundary == 1
        z >= n2 && return nothing
        elem = z * n1 + 1
        face = 4
    elseif boundary == 2
        z >= n2 && return nothing
        elem = z * n1 + n1
        face = 2
    elseif boundary == 3
        z >= n1 && return nothing
        elem = z + 1
        face = 1
    elseif boundary == 4
        z >= n1 && return nothing
        elem = (n2 - 1) * n1 + z + 1
        face = 3
    end
    return (elem, face), z + 1
end

# VertexIterator
function Base.length(
    viter::VertexIterator{T},
) where {M <: AbstractMesh, T <: GridTopology{M}}
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
) where {M <: AbstractMesh, T <: GridTopology{M}}
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
function Base.length(
    vertex::Vertex{T},
) where {M <: AbstractMesh, T <: GridTopology{M}}
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
) where {M <: AbstractMesh, T <: GridTopology{M}}
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
        if z1 == 0 && (vert == 2 || vert == 3)
            vert = 4
        end
        if z1 == n1 && (vert == 1 || vert == 4)
            vert += 1
        end
    end
    if !x2periodic
        if z2 == 0 && vert >= 3
            vert += 2
        end
        if z2 == n2 && vert <= 2
            vert = !x1periodic && z1 == 0 ? 4 : 3
        end
    end

    if vert > 4
        return nothing
    end

    if vert == 2 || vert == 3
        z1 = mod(z1 - 1, nv1)
    end
    if vert == 3 || vert == 4
        z2 = mod(z2 - 1, nv2)
    end
    elem = z2 * n1 + z1 + 1
    return (elem, vert), vert
end


# Uniform grid implementations, dispatching on EquispacedRectangleMesh
function vertex_coordinates(
    topology::GridTopology{M},
    elem::Integer,
) where {M <: EquispacedRectangleMesh}
    @assert 1 <= elem <= nlocalelems(topology)
    CT = Topologies.coordinate_type(topology)

    # convert to 0-based indices
    mesh = topology.mesh
    n1 = mesh.n1
    n2 = mesh.n2
    range1 = mesh.range1
    range2 = mesh.range2

    z2, z1 = fldmod(elem - 1, n1)

    c1 = CT(range1[z1 + 1], range2[z2 + 1])
    c2 = CT(range1[z1 + 2], range2[z2 + 1])
    c3 = CT(range1[z1 + 2], range2[z2 + 2])
    c4 = CT(range1[z1 + 1], range2[z2 + 2])
    return (c1, c2, c3, c4)
end

function opposing_face(
    topology::GridTopology{M},
    elem::Integer,
    face::Integer,
) where {M}
    @assert 1 <= elem <= nlocalelems(topology)
    @assert 1 <= face <= 4

    # convert to 0-based indices
    mesh = topology.mesh
    n1 = mesh.n1
    n2 = mesh.n2
    x1periodic = isnothing(mesh.domain.x1boundary)
    x2periodic = isnothing(mesh.domain.x2boundary)

    z2, z1 = fldmod(elem - 1, n1)
    if face == 4
        z1 -= 1
        if z1 < 0
            if !x1periodic
                return (0, 1, false) # boundary
            end
            z1 += n1
        end
        opface = 2
    elseif face == 2
        z1 += 1
        if z1 == n1
            if !x1periodic
                return (0, 2, false) # boundary
            end
            z1 -= n1
        end
        opface = 4
    elseif face == 1
        z2 -= 1
        if z2 < 0
            if !x2periodic
                return (0, 3, false) # boundary
            end
            z2 += n2
        end
        opface = 3
    elseif face == 3
        z2 += 1
        if z2 == n2
            if !x2periodic
                return (0, 4, false) # boundary
            end
            z2 -= n2
        end
        opface = 1
    end
    opelem = z2 * n1 + z1 + 1
    return opelem, opface, true
end


# Non-uniform grid implementations, dispatching on TensorProductMesh

function vertex_coordinates(
    topology::GridTopology{M},
    elem::Integer,
) where {M <: TensorProductMesh}
    @assert 1 <= elem <= nlocalelems(topology)

    # convert to 0-based indices
    mesh = topology.mesh
    n1 = mesh.n1
    n2 = mesh.n2
    coordinates = mesh.coordinates

    z2, z1 = fldmod(elem - 1, n1)

    c1 = coordinates[z1 * (n2 + 1) + (z2 + 1)]
    c2 = coordinates[(z1 + 1) * (n2 + 1) + (z2 + 1)]
    c3 = coordinates[(z1 + 1) * (n2 + 1) + (z2 + 2)]
    c4 = coordinates[z1 * (n2 + 1) + (z2 + 2)]

    return (c1, c2, c3, c4)
end
