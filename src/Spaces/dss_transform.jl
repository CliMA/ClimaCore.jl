import ..Topologies: Topology2D
using ..RecursiveApply

"""
    dss_transform(arg, local_geometry, weight, I...)

Transfrom `arg[I...]` to a basis for direct stiffness summation (DSS).
Transformations only apply to vector quantities.

- `local_geometry[I...]` is the relevant `LocalGeometry` object. If it is `nothing`, then no transformation is performed
- `weight[I...]` is the relevant DSS weights. If `weight` is `nothing`, then the result is simply summation.

See [`Spaces.weighted_dss!`](@ref).
"""
dss_transform(arg, local_geometry, weight, i, j) =
    dss_transform(arg[i, j], local_geometry[i, j], weight[i, j])
dss_transform(arg, local_geometry, weight::Nothing, i, j) =
    dss_transform(arg[i, j], local_geometry[i, j], 1)
dss_transform(arg, local_geometry::Nothing, weight::Nothing, i, j) = arg[i, j]

dss_transform(arg, local_geometry, weight, i) =
    dss_transform(arg[i], local_geometry[i], weight[i])
dss_transform(arg, local_geometry, weight::Nothing, i) =
    dss_transform(arg[i], local_geometry[i], 1)
dss_transform(arg, local_geometry::Nothing, weight::Nothing, i) = arg[i]

@inline function dss_transform(
    arg::Tuple{},
    local_geometry::Geometry.LocalGeometry,
    weight,
)
    ()
end
@inline function dss_transform(
    arg::Tuple,
    local_geometry::Geometry.LocalGeometry,
    weight,
)
    (
        dss_transform(first(arg), local_geometry, weight),
        dss_transform(Base.tail(arg), local_geometry, weight)...,
    )
end
@inline function dss_transform(
    arg::NamedTuple{names},
    local_geometry::Geometry.LocalGeometry,
    weight,
) where {names}
    NamedTuple{names}(dss_transform(Tuple(arg), local_geometry, weight))
end
@inline dss_transform(
    arg::Number,
    local_geometry::Geometry.LocalGeometry,
    weight,
) = arg * weight
@inline dss_transform(
    arg::Geometry.AxisTensor{T, N, <:Tuple{Vararg{Geometry.CartesianAxis}}},
    local_geometry::Geometry.LocalGeometry,
    weight,
) where {T, N} = arg * weight
@inline dss_transform(
    arg::Geometry.CartesianVector,
    local_geometry::Geometry.LocalGeometry,
    weight,
) = arg * weight
@inline dss_transform(
    arg::Geometry.AxisTensor{T, N, <:Tuple{Vararg{Geometry.LocalAxis}}},
    local_geometry::Geometry.LocalGeometry,
    weight,
) where {T, N} = arg * weight
@inline dss_transform(
    arg::Geometry.LocalVector,
    local_geometry::Geometry.LocalGeometry,
    weight,
) = arg * weight
@inline dss_transform(
    arg::Geometry.Covariant3Vector,
    local_geometry::Geometry.LocalGeometry,
    weight,
) = arg * weight

@inline function dss_transform(
    arg::Geometry.AxisVector,
    local_geometry::Geometry.LocalGeometry,
    weight,
)
    ax = axes(local_geometry.∂x∂ξ, 1)
    axfrom = axes(arg, 1)
    # TODO: make this consistent for 2D / 3D
    # 2D domain axis (1,2) horizontal curl
    if ax isa Geometry.UVAxis && (
        axfrom isa Geometry.Covariant3Axis ||
        axfrom isa Geometry.Contravariant3Axis
    )
        return arg * weight
    end
    # 2D domain axis (1,3) curl
    if ax isa Geometry.UWAxis && (
        axfrom isa Geometry.Covariant2Axis ||
        axfrom isa Geometry.Contravariant2Axis
    )
        return arg * weight
    end
    # workaround for using a Covariant12Vector in a UW space
    if ax isa Geometry.UWAxis && axfrom isa Geometry.Covariant12Axis
        # return Geometry.transform(Geometry.UVWAxis(), arg, local_geometry)
        u₁, v = Geometry.components(arg)
        uw_vector = Geometry.project(
            Geometry.UWAxis(),
            Geometry.Covariant13Vector(u₁, zero(u₁)),
            local_geometry,
        )
        u, w = Geometry.components(uw_vector)
        return Geometry.UVWVector(u, v, w) * weight
    end
    Geometry.project(ax, arg, local_geometry) * weight
end

"""
    dss_untransform(T, targ, local_geometry, I...)

Transform `targ[I...]` back to a value of type `T` after performing direct stiffness summation (DSS).

See [`Spaces.weighted_dss!`](@ref).
"""
dss_untransform(::Type{T}, targ, local_geometry, i, j) where {T} =
    dss_untransform(T, targ, local_geometry[i, j])
dss_untransform(::Type{T}, targ, local_geometry::Nothing, i, j) where {T} =
    dss_untransform(T, targ, local_geometry)
dss_untransform(::Type{T}, targ, local_geometry, i) where {T} =
    dss_untransform(T, targ, local_geometry[i])
dss_untransform(::Type{T}, targ, local_geometry::Nothing, i) where {T} =
    dss_untransform(T, targ, local_geometry)
@inline function dss_untransform(
    ::Type{NamedTuple{names, T}},
    targ::NamedTuple{names},
    local_geometry,
) where {names, T}
    NamedTuple{names}(dss_untransform(T, Tuple(targ), local_geometry))
end
@inline dss_untransform(
    ::Type{Tuple{}},
    targ::Tuple{},
    local_geometry::Geometry.LocalGeometry,
) = ()
@inline function dss_untransform(
    ::Type{T},
    targ::Tuple,
    local_geometry::Geometry.LocalGeometry,
) where {T <: Tuple}
    (
        dss_untransform(
            Base.tuple_type_head(T),
            Base.first(targ),
            local_geometry,
        ),
        dss_untransform(
            Base.tuple_type_tail(T),
            Base.tail(targ),
            local_geometry,
        )...,
    )
end

@inline dss_untransform(::Type{T}, targ::T, local_geometry) where {T} = targ
@inline dss_untransform(
    ::Type{T},
    targ::T,
    local_geometry::Geometry.LocalGeometry,
) where {T <: Geometry.AxisVector} = targ
@inline function dss_untransform(
    ::Type{Geometry.AxisVector{T, A1, S}},
    targ::Geometry.AxisVector,
    local_geometry::Geometry.LocalGeometry,
) where {T, A1, S}
    ax = A1()
    # workaround for using a Covariant12Vector in a UW space
    if (
        axes(local_geometry.∂x∂ξ, 1) isa Geometry.UWAxis &&
        ax isa Geometry.Covariant12Axis
    )
        u, u₂, w = Geometry.components(targ)
        u₁_vector = Geometry.transform(
            Geometry.Covariant1Axis(),
            Geometry.UWVector(u, w),
            local_geometry,
        )
        u₁, = Geometry.components(u₁_vector)
        return Geometry.Covariant12Vector(u₁, u₂)
    end
    Geometry.transform(ax, targ, local_geometry)
end

function weighted_dss! end
function weighted_dss_start! end
function weighted_dss_internal! end
function weighted_dss_ghost! end

# for backward compatibility
function weighted_dss2! end
function weighted_dss_start2! end
function weighted_dss_internal2! end
function weighted_dss_ghost2! end
function dss2! end

# helper functions for DSS2
function _get_idx(sizet::NTuple{5, Int}, loc::NTuple{5, Int})
    (n1, n2, n3, n4, n5) = sizet
    (i1, i2, i3, i4, i5) = loc
    return i1 +
           ((i2 - 1) + ((i3 - 1) + ((i4 - 1) + (i5 - 1) * n4) * n3) * n2) * n1
end

function _get_idx(sizet::NTuple{4, Int}, loc::NTuple{4, Int})
    (n1, n2, n3, n4) = sizet
    (i1, i2, i3, i4) = loc
    return i1 + ((i2 - 1) + ((i3 - 1) + (i4 - 1) * n3) * n2) * n1
end

function _get_idx(sizet::NTuple{3, Int}, idx::Int)
    (n1, n2, n3) = sizet
    i3 = cld(idx, n1 * n2)
    i2 = cld(idx - (i3 - 1) * n1 * n2, n1)
    i1 = idx - (i3 - 1) * n1 * n2 - (i2 - 1) * n1
    return (i1, i2, i3)
end

function _get_idx(sizet::NTuple{4, Int}, idx::Int)
    (n1, n2, n3, n4) = sizet
    i4 = cld(idx, n1 * n2 * n3)
    i3 = cld(idx - (i4 - 1) * n1 * n2 * n3, n1 * n2)
    i2 = cld(idx - (i4 - 1) * n1 * n2 * n3 - (i3 - 1) * n1 * n2, n1)
    i1 = idx - (i4 - 1) * n1 * n2 * n3 - (i3 - 1) * n1 * n2 - (i2 - 1) * n1
    return (i1, i2, i3, i4)
end

function _get_idx_metric(sizet::NTuple{5, Int}, loc::NTuple{4, Int})
    nmetric = sizet[4]
    (i11, i12, i21, i22) = nmetric == 4 ? (1, 2, 3, 4) : (1, 2, 4, 5)
    (level, i, j, elem) = loc
    return (
        _get_idx(sizet, (level, i, j, i11, elem)),
        _get_idx(sizet, (level, i, j, i12, elem)),
        _get_idx(sizet, (level, i, j, i21, elem)),
        _get_idx(sizet, (level, i, j, i22, elem)),
    )
    return nothing
end

function _representative_slab(data, ::Type{DA}) where {DA}
    rebuild_flag = DA isa Array ? false : true
    if isnothing(data)
        return nothing
    elseif rebuild_flag
        return DataLayouts.rebuild(slab(data, 1, 1), Array)
    else
        return slab(data, 1, 1)
    end
end

_transformed_type(data, local_geometry, local_weights, ::Type{DA}) where {DA} =
    typeof(
        dss_transform(
            _representative_slab(data, DA),
            _representative_slab(local_geometry, DA),
            _representative_slab(local_weights, DA),
            1,
            1,
        ),
    )

struct GhostBuffer{G, D}
    graph_context::G
    send_data::D
    recv_data::D
end

recv_buffer(ghost::GhostBuffer) = ghost.recv_data

create_ghost_buffer(data, topology::Topologies.AbstractTopology) = nothing

function create_ghost_buffer(
    data::Union{DataLayouts.IJFH{S, Nij}, DataLayouts.VIJFH{S, Nij}},
    topology::Topologies.Topology2D,
) where {S, Nij}
    if data isa DataLayouts.IJFH
        send_data = DataLayouts.IJFH{S, Nij}(
            typeof(parent(data)),
            Topologies.nsendelems(topology),
        )
        recv_data = DataLayouts.IJFH{S, Nij}(
            typeof(parent(data)),
            Topologies.nrecvelems(topology),
        )
        k = stride(parent(send_data), 4)
    else
        Nv, _, _, Nf, _ = size(parent(data))
        send_data = DataLayouts.VIJFH{S, Nij}(
            similar(
                parent(data),
                (Nv, Nij, Nij, Nf, Topologies.nsendelems(topology)),
            ),
        )
        recv_data = DataLayouts.VIJFH{S, Nij}(
            similar(
                parent(data),
                (Nv, Nij, Nij, Nf, Topologies.nrecvelems(topology)),
            ),
        )
        k = stride(parent(send_data), 5)
    end

    graph_context = ClimaComms.graph_context(
        topology.context,
        parent(send_data),
        k .* topology.send_elem_lengths,
        topology.neighbor_pids,
        parent(recv_data),
        k .* topology.recv_elem_lengths,
        topology.neighbor_pids,
    )
    GhostBuffer(graph_context, send_data, recv_data)
end
