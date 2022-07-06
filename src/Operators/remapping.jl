using ..Meshes
using ..Spaces:
    AbstractSpace, SpectralElementSpace1D, SpectralElementSpace2D, Quadratures
using ..Topologies: Topology2D, IntervalTopology
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

See [Ullrich2015](@cite) eqs. 3 and 4.
"""
function LinearRemap(target::AbstractSpace, source::AbstractSpace)
    R = linear_remap_op(target, source)
    LinearRemap(target, source, R)
end

"""
    remap(R::LinearRemap, source_field::Field)

Applies `R` to `source_field` and outputs a new field on the target space.
"""
function remap(R::LinearRemap, source_field::Field)
    target_space = R.target
    target_field = similar(source_field, target_space, eltype(source_field))
    remap!(target_field, R, source_field)
end

"""
    remap!(target_field::Field, R::LinearRemap, source_field::Field)

Applies the remapping operator `R` to `source_field` and stores the solution in `target_field`.
"""
function remap!(target_field::Field, R::LinearRemap, source_field::Field)
    @assert (R.source == axes(source_field) && R.target == axes(target_field)) "Remap operator and field space dimensions do not match."
    mul!(vec(parent(target_field)), R.map, vec(parent(source_field)))
    return target_field
end

"""
    linear_remap_op(target::AbstractSpace, source::AbstractSpace)

Computes linear remapping operator `R` for remapping from `source` to `target` spaces.

Entry `R_{ij}` gives the contribution weight to the target node `i` from
source node `j`; nodes are indexed by their global position, determined
by both element index and nodal order within the element.
"""
function linear_remap_op(target::AbstractSpace, source::AbstractSpace)
    J = 1.0 ./ local_weights(target) # workaround for julia #26561
    W = overlap(target, source)
    return W .* J
end

"""
    overlap(target, source)

Computes local weights of the overlap mesh for `source` to `target` spaces.
"""
function overlap(
    target::T,
    source::S,
) where {
    T <: SpectralElementSpace1D{<:IntervalTopology},
    S <: SpectralElementSpace1D{<:IntervalTopology},
}
    return x_overlap(target, source)
end

function overlap(
    target::T,
    source::S,
) where {
    T <: SpectralElementSpace2D{<:Topology2D},
    S <: SpectralElementSpace2D{<:Topology2D},
}
    @assert (
        typeof(Spaces.topology(target).mesh) <: Meshes.RectilinearMesh &&
        typeof(Spaces.topology(source).mesh) <: Meshes.RectilinearMesh
    )
    X_ov = x_overlap(target, source)
    Y_ov = y_overlap(target, source)
    src_topo = Spaces.topology(source)
    tgt_topo = Spaces.topology(target)
    src_elems_x = nxelems(src_topo)
    tgt_elems_x = nxelems(tgt_topo)
    Nq_s = Quadratures.degrees_of_freedom(Spaces.quadrature_style(source))
    Nq_t = Quadratures.degrees_of_freedom(Spaces.quadrature_style(target))
    W = spzeros(size(X_ov) .* size(Y_ov))
    # Assemble kronecker products by element.
    for i in 1:Topologies.nlocalelems(target)
        global_tgt = (Nq_t^2 * (i - 1) + 1):(Nq_t^2 * i) # global indices of target dof
        meshelem = CartesianIndices(tgt_topo.orderindex)[i]
        ix, iy = meshelem.I
        tgt_x = (Nq_t * (ix - 1) + 1):(Nq_t * ix) # indices of target dof in Xov
        tgt_y = (Nq_t * (iy - 1) + 1):(Nq_t * iy) # indices of target dof in Yov
        for j in 1:Topologies.nlocalelems(source)
            global_src = (Nq_s^2 * (j - 1) + 1):(Nq_s^2 * j)
            meshelem = CartesianIndices(src_topo.orderindex)[j]
            jx, jy = meshelem.I
            src_x = (Nq_s * (jx - 1) + 1):(Nq_s * jx)
            src_y = (Nq_s * (jy - 1) + 1):(Nq_s * jy)
            W[global_tgt, global_src] .=
                kron(Y_ov[tgt_y, src_y], X_ov[tgt_x, src_x])
        end
    end
    return W
end

"""
    x_overlap(target, source)

Computes 1D local weights of the overlap mesh for `source` to `target` spaces.

For 1D spaces, this returns the full local weight matrix of the overlap. In 2D,
this returns the overlap weights in the first dimension.
"""
function x_overlap(
    target::T,
    source::S,
) where {
    T <: Union{SpectralElementSpace1D, SpectralElementSpace2D},
    S <: Union{SpectralElementSpace1D, SpectralElementSpace2D},
}
    target_topo = Spaces.topology(target)
    source_topo = Spaces.topology(source)
    nelems_t, nelems_s = nxelems(target_topo), nxelems(source_topo)
    QS_t = Spaces.quadrature_style(target)
    QS_s = Spaces.quadrature_style(source)
    Nq_t = Quadratures.degrees_of_freedom(QS_t)
    Nq_s = Quadratures.degrees_of_freedom(QS_s)
    v1, v2 = 1, 2

    J_ov = spzeros(nelems_t * Nq_t, nelems_s * Nq_s)
    for i in 1:nelems_t
        vertices_i = Topologies.vertex_coordinates(target_topo, i)
        min_i, max_i = xcomponent(vertices_i[v1]), xcomponent(vertices_i[v2])
        for j in 1:nelems_s
            vertices_j = Topologies.vertex_coordinates(source_topo, j)
            min_j, max_j =
                xcomponent(vertices_j[v1]), xcomponent(vertices_j[v2])
            overlap_weights!(
                J_ov,
                target,
                source,
                i,
                j,
                (min_i, max_i),
                (min_j, max_j),
            )
        end
    end
    return J_ov
end

"""
    y_overlap(target, source)

Computes 1D local weights of the overlap mesh for `source` to `target` spaces.

This returns the overlap weights in the second dimension for 2D spaces.
"""
function y_overlap(
    target::T,
    source::S,
) where {
    T <: Union{SpectralElementSpace1D, SpectralElementSpace2D},
    S <: Union{SpectralElementSpace1D, SpectralElementSpace2D},
}
    target_topo = Spaces.topology(target)
    source_topo = Spaces.topology(source)
    nelems_t, nelems_s = nyelems(target_topo), nyelems(source_topo)
    nx_target, nx_source = nxelems(target_topo), nxelems(source_topo)
    elem_idx = (i, n) -> 1 + (i - 1) * n
    QS_t = Spaces.quadrature_style(target)
    QS_s = Spaces.quadrature_style(source)
    Nq_t = Quadratures.degrees_of_freedom(QS_t)
    Nq_s = Quadratures.degrees_of_freedom(QS_s)
    v1, v2 = 1, 4

    J_ov = spzeros(nelems_t * Nq_t, nelems_s * Nq_s)
    for i in 1:nelems_t
        vertices_i =
            Topologies.vertex_coordinates(target_topo, elem_idx(i, nx_target))
        min_i, max_i = ycomponent(vertices_i[v1]), ycomponent(vertices_i[v2])
        for j in 1:nelems_s
            vertices_j = Topologies.vertex_coordinates(
                source_topo,
                elem_idx(j, nx_source),
            )
            min_j, max_j =
                ycomponent(vertices_j[v1]), ycomponent(vertices_j[v2])
            overlap_weights!(
                J_ov,
                target,
                source,
                i,
                j,
                (min_i, max_i),
                (min_j, max_j),
            )
        end
    end
    return J_ov
end

"""
    overlap_weights!(J_ov, target, source, i, j, coords_t, coords_s)

Computes the overlap weights for a pair of source and target elements.

The spatial overlap of element `i` on the `target` space and element `j`
of the `source` space, computed for each nodal pair, approximating
[Ullrich2015](@cite) eq. 19 via quadrature.
"""
function overlap_weights!(J_ov, target, source, i, j, coords_t, coords_s)
    FT = Spaces.undertype(source)
    QS_t = Spaces.quadrature_style(target)
    QS_s = Spaces.quadrature_style(source)
    Nq_t = Quadratures.degrees_of_freedom(QS_t)
    Nq_s = Quadratures.degrees_of_freedom(QS_s)

    min_t, max_t = coords_t
    min_s, max_s = coords_s
    min_ov, max_ov = max(min_t, min_s), min(max_t, max_s)
    if max_ov <= min_ov
        return
    end

    ξ_s, w_s = Quadratures.quadrature_points(FT, QS_s)
    ξ_t, w_t = Quadratures.quadrature_points(FT, QS_t)
    if Nq_t >= Nq_s
        ξ_ov, w_ov = ξ_t, w_t
    else
        ξ_ov, w_ov = ξ_s, w_s
    end
    x_ov = FT(0.5) * (min_ov + max_ov) .+ FT(0.5) * (max_ov - min_ov) * ξ_ov
    x_t = FT(0.5) * (min_t + max_t) .+ FT(0.5) * (max_t - min_t) * ξ_t
    x_s = FT(0.5) * (min_s + max_s) .+ FT(0.5) * (max_s - min_s) * ξ_s

    # column k of I_mat gives the k-th target basis function defined on the overlap element
    I_mat_t = Quadratures.interpolation_matrix(x_ov, x_t)
    I_mat_s = Quadratures.interpolation_matrix(x_ov, x_s)

    for k in 1:Nq_t
        targ_idx = Nq_t * (i - 1) + k # global nodal index
        for l in 1:Nq_s
            src_idx = Nq_s * (j - 1) + l
            # (integral of src_basis * tgt_basis on overlap) / (reference elem length) * overlap elem length
            J_ov[targ_idx, src_idx] =
                (w_ov' * (I_mat_t[:, k] .* I_mat_s[:, l])) ./ 2 *
                (max_ov - min_ov)
        end
    end
end

nxelems(topology::Topologies.IntervalTopology) =
    Topologies.nlocalelems(topology)
nxelems(topology::Topologies.Topology2D) =
    size(Meshes.elements(topology.mesh), 1)
nyelems(topology::Topologies.Topology2D) =
    size(Meshes.elements(topology.mesh), 2)

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

See [Ullrich2015] section 2.
"""
function local_weights(space::AbstractSpace)
    wj = space.local_geometry.WJ
    return vec(parent(wj))
end

## RegularLatLong interpolation
struct RLLRemap{
    T <: Meshes.RegularLatLongMesh,
    S <: AbstractSpace,
    M <: AbstractMatrix,
    E <: AbstractVector,
    P <: AbstractVector,
}
    target::T # rll mesh
    source::S # source space
    map::M # linear mapping operator
    elems::E # elements associated with target pts
    p::P # coordinate permutation vector
end

function RLLRemap(target_mesh::Meshes.RegularLatLongMesh, source_space)

    source_mesh = Spaces.topology(source_space).mesh
    latitude, longitude = target_mesh.lat, target_mesh.long

    # store latlong coords and associated elements
    coords = []
    for lat in latitude
        for long in longitude
            coord = Geometry.LatLongPoint(lat, long)
            elem = Meshes.containing_element(source_mesh, coord)
            push!(coords, (coord = coord, elem = elem))
        end
    end

    # sort coords by element
    p = sortperm(coords, by = coord -> coord.elem)
    permute!(coords, p)
    coords = map(coords, 1:length(coords)) do coord, i
        (coord..., idx = i)
    end
    # get unique elements and the indices of the first associated coords
    elems = map(x -> (elem = x.elem, idx = x.idx), unique(x -> x.elem, coords))

    QS_s = Spaces.quadrature_style(source_space)
    Nq_s = Quadratures.degrees_of_freedom(QS_s)
    # rows - for each rll coord, cols - for each node in corresponding elem
    op = spzeros(length(coords), Nq_s^2)
    for i in 1:length(elems)
        # better solution than start_idx, last_idx?
        elem, start_idx = elems[i]
        last_idx = i == length(elems) ? length(coords) : elems[i + 1].idx - 1
        for j in start_idx:last_idx
            op[j, :] = remap_weights(coords[j].coord, elem, source_space)
        end
    end
    return RLLRemap(target_mesh, source_space, op, elems, p)
end

"""
    remap(R::RLLRemap, source_field::Field)

Applies `R` to `source_field` and outputs a vector of values corresponding to latlong pts.
"""
function remap(R::RLLRemap, source_field::Field)
    target_vec = zeros(Meshes.ncoordinates(R.target))
    remap!(target_vec, R, source_field)
end

"""
    remap!(target_vec::Field, R::RLLRemap, source_field::Field)

Applies the remapping operator `R` to `source_field` and stores the solution in `target_vec`.
"""
function remap!(target_vec, R::RLLRemap, source_field::Field)
    @assert (R.source == axes(source_field)) "Remap operator and field space dimensions do not match."
    @assert (length(target_vec) == Meshes.ncoordinates(R.target)) "Target vector and operator mesh dimensions do not match."
    data = parent(source_field)
    for (i, elem) in enumerate(R.elems)
        elem_idx =
            Topologies.localelemindex(axes(source_field).topology, elem.elem)
        start_idx = elem.idx
        last_idx =
            i == length(R.elems) ? length(target_vec) : R.elems[i + 1].idx - 1
        for j in start_idx:last_idx
            # multiply rows of map by nodal values in corresponding element
            target_vec[j] = R.map[j, :]' * vec(data[:, :, 1, elem_idx])
        end
    end
    invpermute!(target_vec, R.p)
    return target_vec
end

function remap_weights(coord, elem, source_space)
    FT = Spaces.undertype(source_space)
    quad = Spaces.quadrature_style(source_space)
    source_mesh = Spaces.topology(source_space).mesh

    ξ, w = Quadratures.quadrature_points(FT, quad)
    ξ_latlong = Meshes.reference_coordinates(source_mesh, elem, coord)
    Imat_x = Quadratures.interpolation_matrix(SVector(ξ_latlong[1]), ξ)
    Imat_y = Quadratures.interpolation_matrix(SVector(ξ_latlong[2]), ξ)
    return kron(Imat_y, Imat_x) #weights to get from nodes in elem to coord
end
