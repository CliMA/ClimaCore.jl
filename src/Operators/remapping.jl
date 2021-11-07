using ..Spaces: AbstractSpace, SpectralElementSpace2D, Quadratures
using ..Topologies: GridTopology
using ..Fields: Field
using SparseArrays, LinearAlgebra

struct LinearRemap{T <: AbstractSpace, S <: AbstractSpace, M <: AbstractMatrix}
    target::T
    source::S
    map::M # linear mapping operator
end

"""
    LinearRemap(target::AbstractSpace, source::AbstractSpace)

A remapping operator from the `source` space to the `target` space.
"""
function LinearRemap(target::AbstractSpace, source::AbstractSpace)
    R = linear_remap_op(target, source)
    LinearRemap(target, source, R)
end

"""
    remap(R::LinearRemap, source_field::Field)

Applies the remapping operator `R` to `source_field`. Outputs a new field on the target space mapped to by `R`.
"""
function remap(R::LinearRemap, source_field::Field)
    target_space = R.target
    target_field = similar(source_field, target_space, eltype(source_field))
    remap!(target_field, R, source_field)
end

"""
    remap!(target_field::Field, R::LinearRemap, source_field::Field)

Applies the remapping operator `R` to `source_field`. After the call `target_field` contains the solution.
"""
function remap!(target_field::Field, R::LinearRemap, source_field::Field)
    mul!(vec(parent(target_field)), R.map, vec(parent(source_field)))
    return target_field
end

"""
    linear_remap_op(target::AbstractSpace, source::AbstractSpace)

Computes linear remapping operator `R` for remapping from `source_topo` to `target_topo` topologies.

Entry `R_{ij}` gives the contribution weight to the target element `i` from overlapping source
element `j`.
"""
function linear_remap_op(target::AbstractSpace, source::AbstractSpace)
    J = 1.0 ./ local_weights(target) # workaround for julia #26561
    W = overlap_weights(target, source)
    return W .* J
end

"""
    overlap_weights(target::T, source::S) where {T <: SpectralElementSpace2D{<:GridTopology, Quadratures.GL{1}},
            S <: SpectralElementSpace2D{<:GridTopology, Quadratures.GL{1}}}

Computes local weights of the overlap mesh for `source_topo` to `target_topo` topologies.

Entry `W_{ij}` gives the area that target element `i` overlaps source element `j`.
"""
function overlap_weights(
    target::T,
    source::S,
) where {
    T <: SpectralElementSpace2D{<:GridTopology, Quadratures.GL{1}},
    S <: SpectralElementSpace2D{<:GridTopology, Quadratures.GL{1}},
}
    target_topo = Spaces.topology(target)
    source_topo = Spaces.topology(source)
    nx1, ny1 = target_topo.mesh.n1, target_topo.mesh.n2
    nx2, ny2 = source_topo.mesh.n1, source_topo.mesh.n2

    X_ov = spzeros(nx1, nx2)
    Y_ov = spzeros(ny1, ny2)

    # Calculate element overlap pattern in x-dimension
    # X_ov[i,j] = overlap length along x-dimension between target elem i and source elem j
    for i in 1:nx1
        vertices_i = Topologies.vertex_coordinates(target_topo, i)
        xmin_i, xmax_i = vertices_i[1].x, vertices_i[2].x
        for j in 1:nx2
            vertices_j = Topologies.vertex_coordinates(source_topo, j)
            xmin_j, xmax_j = vertices_j[1].x, vertices_j[2].x
            xmin_ov, xmax_ov = max(xmin_i, xmin_j), min(xmax_i, xmax_j)
            x_overlap = xmax_ov > xmin_ov ? xmax_ov - xmin_ov : continue
            X_ov[i, j] = x_overlap
        end
    end

    # Calculate element overlap pattern in y-dimension
    for i in 1:ny1
        vertices_i =
            Topologies.vertex_coordinates(target_topo, 1 + (i - 1) * nx1)
        ymin_i, ymax_i = vertices_i[1].y, vertices_i[4].y
        for j in 1:ny2
            vertices_j =
                Topologies.vertex_coordinates(source_topo, 1 + (j - 1) * nx2)
            ymin_j, ymax_j = vertices_j[1].y, vertices_j[4].y
            ymin_ov, ymax_ov = max(ymin_i, ymin_j), min(ymax_i, ymax_j)
            y_overlap = ymax_ov > ymin_ov ? ymax_ov - ymin_ov : continue
            Y_ov[i, j] = y_overlap
        end
    end
    return kron(Y_ov, X_ov)
end

"""
    local_weights(space::AbstractSpace)

Each degree of freedom is associated with a local weight J_i.
For finite volumes the local weight J_i would represent the geometric area
of the associated region. For nodal finite elements, the local weight
represents the value of the global Jacobian, or some global integral of the
associated basis function.
"""
function local_weights(space::AbstractSpace)
    wj = space.local_geometry.WJ
    FT = eltype(wj)
    n = length(wj)
    J = zeros(FT, n, 1)
    for i in 1:n
        J[i] = slab(wj, i)[1, 1]
    end
    return J
end
