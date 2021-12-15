using ..Spaces:
    AbstractSpace, SpectralElementSpace1D, SpectralElementSpace2D, Quadratures
using ..Topologies: GridTopology, IntervalTopology
using ..Fields: Field
using ..DataLayouts
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
    overlap_weights(target::T, source::S) where {T <: SpectralElementSpace1D{<:GridTopology, Quadratures.GL{1}},
            S <: SpectralElementSpace1D{<:GridTopology, Quadratures.GL{1}}}

Computes local weights of the overlap mesh for `source_topo` to `target_topo` topologies.

Returns sparse matrix `W` where entry `W_{ij}` gives the area that target element `i` overlaps source element `j`.
"""
function overlap_weights(
    target::T,
    source::S,
) where {
    T <: SpectralElementSpace1D{<:IntervalTopology, Quadratures.GL{1}},
    S <: SpectralElementSpace1D{<:IntervalTopology, Quadratures.GL{1}},
}
    # Calculate element overlap pattern
    # X_ov[i,j] = overlap length between target elem i and source elem j
    X_ov = fv_x_overlap(target, source)

    return X_ov
end

"""
    overlap_weights(target::T, source::S) where {T <: SpectralElementSpace2D{<:GridTopology, Quadratures.GL{1}},
            S <: SpectralElementSpace2D{<:GridTopology, Quadratures.GL{1}}}

Computes local weights of the overlap mesh for `source_topo` to `target_topo` topologies.

Returns sparse matrix `W` where entry `W_{ij}` gives the area that target element `i` overlaps source element `j`.
"""
function overlap_weights(
    target::T,
    source::S,
) where {
    T <: SpectralElementSpace2D{<:GridTopology, Quadratures.GL{1}},
    S <: SpectralElementSpace2D{<:GridTopology, Quadratures.GL{1}},
}
    # Calculate element overlap pattern in x-dimension
    # X_ov[i,j] = overlap length along x-dimension between target elem i and source elem j
    X_ov = fv_x_overlap(target, source)

    # Calculate element overlap pattern in y-dimension
    Y_ov = fv_y_overlap(target, source)

    return kron(Y_ov, X_ov)
end

function fv_x_overlap(
    target::T,
    source::S,
) where {
    T <: Union{SpectralElementSpace1D, SpectralElementSpace2D},
    S <: Union{SpectralElementSpace1D, SpectralElementSpace2D},
}
    target_topo = Spaces.topology(target)
    source_topo = Spaces.topology(source)
    ntarget, nsource = nxelems(target_topo), nxelems(source_topo)
    v1, v2 = 1, 2

    W_ov = spzeros(ntarget, nsource)
    for i in 1:ntarget
        vertices_i = Topologies.vertex_coordinates(target_topo, i)
        min_i, max_i = xcomponent(vertices_i[v1]), xcomponent(vertices_i[v2])
        for j in 1:nsource
            vertices_j = Topologies.vertex_coordinates(source_topo, j)
            min_j, max_j =
                xcomponent(vertices_j[v1]), xcomponent(vertices_j[v2])
            min_ov, max_ov = max(min_i, min_j), min(max_i, max_j)
            overlap_length = max_ov > min_ov ? max_ov - min_ov : continue
            W_ov[i, j] = overlap_length
        end
    end
    return W_ov
end

function fv_y_overlap(
    target::T,
    source::S,
) where {
    T <: Union{SpectralElementSpace1D, SpectralElementSpace2D},
    S <: Union{SpectralElementSpace1D, SpectralElementSpace2D},
}
    target_topo = Spaces.topology(target)
    source_topo = Spaces.topology(source)
    ntarget, nsource = nyelems(target_topo), nyelems(source_topo)
    nx_target, nx_source = nxelems(target_topo), nxelems(source_topo)
    elem_idx = (i, n) -> 1 + (i - 1) * n
    v1, v2 = 1, 4

    W_ov = spzeros(ntarget, nsource)
    for i in 1:ntarget
        vertices_i =
            Topologies.vertex_coordinates(target_topo, elem_idx(i, nx_target))
        min_i, max_i = ycomponent(vertices_i[v1]), ycomponent(vertices_i[v2])
        for j in 1:nsource
            vertices_j = Topologies.vertex_coordinates(
                source_topo,
                elem_idx(j, nx_source),
            )
            min_j, max_j =
                ycomponent(vertices_j[v1]), ycomponent(vertices_j[v2])
            min_ov, max_ov = max(min_i, min_j), min(max_i, max_j)
            overlap_length = max_ov > min_ov ? max_ov - min_ov : continue
            W_ov[i, j] = overlap_length
        end
    end
    return W_ov
end

nxelems(topology::Topologies.IntervalTopology) =
    Topologies.nlocalelems(topology)
nxelems(topology::Topologies.GridTopology) =
    size(Meshes.elements(topology.mesh), 1)
nyelems(topology::Topologies.GridTopology) =
    size(Meshes.elements(topology.mesh), 1)

xcomponent(x::Geometry.XPoint) = Geometry.component(x, 1)
xcomponent(xy::Geometry.XYPoint) = Geometry.component(xy, 1)
ycomponent(y::Geometry.YPoint) = Geometry.component(y, 1)
ycomponent(xy::Geometry.XYPoint) = Geometry.component(xy, 2)

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
        J[i] = slab_value(wj, i)
    end
    return J
end

slab_value(data::DataLayouts.IJFH, i) = slab(data, i)[1, 1]
slab_value(data::DataLayouts.IFH, i) = slab(data, i)[1]
