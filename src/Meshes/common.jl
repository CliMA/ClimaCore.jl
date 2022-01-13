boundary_names(mesh::AbstractMesh) = boundary_names(domain(mesh))
coordinate_type(mesh::AbstractMesh) = coordinate_type(domain(mesh))

"""
    nelements(mesh::AbstractMesh)

The number of elements in the mesh.
"""
nelements(mesh::AbstractMesh) = length(elements(mesh))

"""
    i = refindex(ϕ, n)

Given a reference coordinate `ϕ` in the interval `[-1, 1]``, divide the interval
into `n` evenly spaced subintervals, and return the index of the subinterval
(`i`), and the position in the subinterval (`ϕs`), normalized to normalized `[-1, 1]`.
"""
function refindex(ϕ, n)
    ϕn = ϕ * n
    # we use Julia's range lookup here, which avoids intermediate rounding errors.
    return min(searchsortedlast((-n):2:n, ϕn), n)
end
function refcoord(ϕ, n, i)
    ϕn = ϕ * n
    return ϕn - (2 * i - n - 1)
end

"""
    Meshes.SharedVertices(mesh, elem, vert)

An iterator over (element, vertex) pairs that are shared with `(elem,vert)`.
"""
struct SharedVertices{M <: AbstractMesh, E}
    mesh::M
    elem::E
    vert::Int
end
Base.IteratorSize(::Type{<:SharedVertices}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:SharedVertices}) = Base.HasEltype()
Base.eltype(::Type{SharedVertices{M, E}}) where {M, E} = Tuple{E, Int}

function Base.iterate(vertiter::SharedVertices)
    velem = vertiter.elem
    vvert = vertiter.vert
    ccw = false
    # return initial (element, vertex)
    return (velem, vvert), (velem, vvert, ccw)
end
function Base.iterate(vertiter::SharedVertices, (velem, vvert, ccw))
    # initially we go clockwise (ccw == false), we go to the face == vert
    # if ccw, then go to face = vert - 1
    vface = ccw ? mod1(vvert - 1, 4) : vvert
    if is_boundary_face(vertiter.mesh, velem, vface)
        if ccw
            # have already gone both directions: all done
            return nothing
        end
        # try counter-clockwise
        velem = vertiter.elem
        vvert = vertiter.vert
        ccw = true
        return Base.iterate(vertiter, (velem, vvert, ccw))
    end
    opelem, opface, reversed = opposing_face(vertiter.mesh, velem, vface)
    velem = opelem
    vvert = ccw ? opface : mod1(opface + 1, 4)
    if velem == vertiter.elem && vvert == vertiter.vert
        # we're back at where we started: all done
        return nothing
    end
    return (velem, vvert), (velem, vvert, ccw)
end


"""
    Meshes.linearindices(elemorder)

Given a data structure `elemorder[i] = elem` that orders elements, construct the
inverse map from `orderindex = linearindices(elemorder)` such that
`orderindex[elem] = i`.

This will try to use the most efficient structure available.
"""
linearindices(elemorder::CartesianIndices) = LinearIndices(elemorder)
function linearindices(elemorder::AbstractVector{CartesianIndex})
    cmax = maximum(elemorder)
    L = zeros(Int, cmax.I)
    for (i, c) in enumerate(elemorder)
        L[c] = i
    end
    return L
end
function linearindices(elemorder)
    orderindex = Dict{eltype(elemorder), Int}()
    for (i, elem) in elemorder
        orderindex[elem] = i
    end
    return orderindex
end

"""
    M = Meshes.face_connectivity_matrix(mesh, elemorder = elements(mesh))

Construct a `Bool`-valued `SparseCSCMatrix` containing the face connections of
`mesh`. Elements are indexed according to `elemorder`.

Note that `M[i,i] == true` only if two distinct faces of element `i` are connected.
"""
function face_connectivity_matrix(
    mesh::AbstractMesh,
    elemorder = elements(mesh),
    orderindex = linearindices(elemorder),
)
    m = n = length(elemorder)
    I = Int[]
    J = Int[]
    for (i, elem) in enumerate(elemorder)
        for face in 1:4
            if is_boundary_face(mesh, elem, face)
                continue
            end
            opelem, opface, reversed = Meshes.opposing_face(mesh, elem, face)
            j = orderindex[opelem]
            push!(I, i)
            push!(J, j)
        end
    end
    V = trues(length(I))
    return SparseArrays.sparse(I, J, V, m, n)
end

"""
    M = Meshes.vertex_connectivity_matrix(mesh, elemorder = elements(mesh))

Construct a `Bool`-valued `SparseCSCMatrix` containing the vertex connections of
`mesh`. Elements are indexed according to `elemorder`.

Note that `M[i,i] == true` only if two distinct vertices of element `i` are connected.
"""
function vertex_connectivity_matrix(
    mesh::AbstractMesh,
    elemorder = elements(mesh),
    orderindex = linearindices(elemorder),
)
    m = n = length(elemorder)
    I = Int[]
    J = Int[]
    for (i, elem) in enumerate(elemorder)
        for vert in 1:4
            for (velem, vvert) in SharedVertices(mesh, elem, vert)
                if velem == elem && vvert == vert
                    continue
                end
                j = orderindex[velem]
                push!(I, i)
                push!(J, j)
            end
        end
    end
    V = trues(length(I))
    return SparseArrays.sparse(I, J, V, m, n)
end

"""
    Meshes.coordinates(mesh, elem, vert::Int)
    Meshes.coordinates(mesh, elem, ξ::SVector)

Return the physical coordinates of a point in an element `elem` of `mesh`. The
position of the point can either be a vertex number `vert` or the coordinates
`ξ` in the reference element.
"""
function coordinates(mesh::AbstractMesh2D, elem, vert::Integer)
    FT = Domains.float_type(domain(mesh))
    ξ1 = (vert == 1 || vert == 4) ? FT(-1) : FT(1)
    ξ2 = (vert == 1 || vert == 2) ? FT(-1) : FT(1)
    coordinates(mesh, elem, (ξ1, ξ2))
end



"""
    elem = Meshes.containing_element(mesh::AbstractMesh, coord)

The element `elem` in `mesh` containing the coordinate `coord`. If the
coordinate falls on the boundary between two or more elements, an arbitrary
element is chosen.
"""
function containing_element end

"""
    ξ = Meshes.reference_coordinates(mesh::AbstractMesh, elem, coord)

An `SVector` of coordinates in the reference element such that

    Meshes.coordinates(mesh, elem, ξ) == coord

This can be used for interpolation to a specific point.
"""
function reference_coordinates end
