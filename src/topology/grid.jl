"""
    GridTopology(n1,n2)

A periodic `n1` × `n2` topology of elements. Elements are stored sequentially in
the first dimension, then the second dimension.
"""
struct GridTopology{D <: EquispacedRectangleDiscretization} <: AbstractTopology
    discretization::D
end

domain(topology::GridTopology) = topology.discretization.domain

function nlocalelems(topology::GridTopology)
    n1 = topology.discretization.n1
    n2 = topology.discretization.n2
    return n1 * n2
end

function vertex_coordinates(topology::GridTopology, elem::Integer)
    @assert 1 <= elem <= nlocalelems(topology)

    # convert to 0-based indices
    discretization = topology.discretization
    n1 = discretization.n1
    n2 = discretization.n2
    range1 = discretization.range1
    range2 = discretization.range2

    z2, z1 = fldmod(elem - 1, n1)

    c1 = SVector(range1[z1 + 1], range2[z2 + 1])
    c2 = SVector(range1[z1 + 2], range2[z2 + 1])
    c3 = SVector(range1[z1 + 1], range2[z2 + 2])
    c4 = SVector(range1[z1 + 2], range2[z2 + 2])
    return (c1, c2, c3, c4)
end

function opposing_face(topology::GridTopology, elem::Integer, face::Integer)
    @assert 1 <= elem <= nlocalelems(topology)
    @assert 1 <= face <= 4

    # convert to 0-based indices
    discretization = topology.discretization
    n1 = discretization.n1
    n2 = discretization.n2
    x1periodic = discretization.domain.x1periodic
    x2periodic = discretization.domain.x2periodic

    z2, z1 = fldmod(elem - 1, n1)
    if face == 1
        z1 -= 1
        if z1 < 0
            if !x1periodic
                return (0, 1, false)
            end
            z1 += n1
        end
        opface = 2
    elseif face == 2
        z1 += 1
        if z1 == n1
            if !x1periodic
                return (0, 2, false)
            end
            z1 -= n1
        end
        opface = 1
    elseif face == 3
        z2 -= 1
        if z2 < 0
            if !x2periodic
                return (0, 3, false)
            end
            z2 += n2
        end
        opface = 4
    elseif face == 4
        z2 += 1
        if z2 == n2
            if !x2periodic
                return (0, 4, false)
            end
            z2 -= n2
        end
        opface = 3
    end
    opelem = z2 * n1 + z1 + 1
    return opelem, opface, false
end


# InteriorFaceIterator

function Base.length(fiter::InteriorFaceIterator{T}) where {T <: GridTopology}
    topology = fiter.topology
    discretization = topology.discretization
    n1 = discretization.n1
    n2 = discretization.n2
    x1periodic = discretization.domain.x1periodic
    x2periodic = discretization.domain.x2periodic
    return (x1periodic ? n1 : n1 - 1) * n2 + n1 * (x2periodic ? n2 : n2 - 1)
end

function Base.iterate(
    fiter::InteriorFaceIterator{T},
    (d, z1, z2) = (1, 0, 0),
) where {T <: GridTopology}
    # iteration state (major first)
    #  - d ∈ (1,2): face direction
    #  - z1 ∈ 0:n1-1: 0-based face index in direction 1
    #  - z2 ∈ 0:n2-1: 0-based face index in direction 2

    topology = fiter.topology
    discretization = topology.discretization
    n1 = discretization.n1
    n2 = discretization.n2
    x1periodic = discretization.domain.x1periodic
    x2periodic = discretization.domain.x2periodic

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


function Base.length(bfiter::BoundaryFaceIterator{T}) where {T <: GridTopology}
    boundary = bfiter.boundary
    topology = bfiter.topology
    if boundary in (1, 2)
        if topology.discretization.domain.x1periodic
            return 0
        else
            return topology.discretization.n2
        end
    end
    if boundary in (3, 4)
        if topology.discretization.domain.x2periodic
            return 0
        else
            return topology.discretization.n1
        end
    end
end

function Base.iterate(bfiter::BoundaryFaceIterator{T}) where {T <: GridTopology}
    boundary = bfiter.boundary
    topology = bfiter.topology
    if boundary in (1, 2) && topology.discretization.domain.x1periodic
        return nothing
    end
    if boundary in (3, 4) && topology.discretization.domain.x2periodic
        return nothing
    end
    Base.iterate(bfiter, 0)
end

function Base.iterate(
    bfiter::BoundaryFaceIterator{T},
    z,
) where {T <: GridTopology}
    boundary = bfiter.boundary
    topology = bfiter.topology
    discretization = topology.discretization
    n1 = discretization.n1
    n2 = discretization.n2
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
function Base.length(viter::VertexIterator{T}) where {T <: GridTopology}
    topology = viter.topology
    discretization = topology.discretization
    n1 = discretization.n1
    n2 = discretization.n2
    x1periodic = discretization.domain.x1periodic
    x2periodic = discretization.domain.x2periodic
    nv1 = x1periodic ? n1 : n1 + 1
    nv2 = x2periodic ? n2 : n2 + 1
    return nv1 * nv2
end

function Base.iterate(
    viter::VertexIterator{T},
    (z1, z2) = (0, 0),
) where {T <: GridTopology}
    topology = viter.topology
    discretization = topology.discretization
    n1 = discretization.n1
    n2 = discretization.n2
    x1periodic = discretization.domain.x1periodic
    x2periodic = discretization.domain.x2periodic
    nv1 = x1periodic ? n1 : n1 + 1
    nv2 = x2periodic ? n2 : n2 + 1

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
function Base.length(vertex::Vertex{T}) where {T <: GridTopology}
    topology = vertex.topology
    discretization = topology.discretization
    n1 = discretization.n1
    n2 = discretization.n2
    x1periodic = discretization.domain.x1periodic
    x2periodic = discretization.domain.x2periodic

    z1, z2 = vertex.id

    k1 = !x1periodic && (z1 == 0 || z1 == n1) ? 1 : 2
    k2 = !x2periodic && (z2 == 0 || z2 == n2) ? 1 : 2
    return k1 * k2
end


function Base.iterate(vertex::Vertex{T}, vert = 0) where {T <: GridTopology}
    topology = vertex.topology
    discretization = topology.discretization
    n1 = discretization.n1
    n2 = discretization.n2
    x1periodic = discretization.domain.x1periodic
    x2periodic = discretization.domain.x2periodic
    nv1 = x1periodic ? n1 : n1 + 1
    nv2 = x2periodic ? n2 : n2 + 1
    z1, z2 = vertex.id

    vert += 1
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
        z2 = mod(z2 - 1, nv1)
    end
    elem = z2 * n1 + z1 + 1
    return (elem, vert), vert
end
