import ..Topologies: Topology2D
import UnrolledUtilities: unrolled_map

"""
    dss_transform(arg, local_geometry, weight, I)

Transfrom `arg[I]` to a basis for direct stiffness summation (DSS).
Transformations only apply to vector quantities.

- `local_geometry[I]` is the relevant `LocalGeometry` object. If it is `nothing`, then no transformation is performed
- `weight[I]` is the relevant DSS weights. If `weight` is `nothing`, then the result is simply summation.

See [`ClimaCore.Spaces.weighted_dss!`](@ref).
"""
Base.@propagate_inbounds dss_transform(arg, local_geometry, weight, I) =
    dss_transform(arg[I], local_geometry[I], weight[I])
Base.@propagate_inbounds dss_transform(
    arg,
    local_geometry,
    weight::Nothing,
    I,
) = dss_transform(arg[I], local_geometry[I], 1)
Base.@propagate_inbounds dss_transform(
    arg,
    local_geometry::Nothing,
    weight::Nothing,
    I,
) = arg[I]

@inline function dss_transform(
    arg::Tuple{},
    local_geometry::Geometry.LocalGeometry,
    weight,
)
    ()
end
@inline dss_transform(
    arg::AutoBroadcaster,
    local_geometry::Geometry.LocalGeometry,
    weight,
) = (arg -> dss_transform(arg, local_geometry, weight)).(arg)
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
    arg::Geometry.AxisTensor{T, N, <:Tuple{}},
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

See [`ClimaCore.Spaces.weighted_dss!`](@ref).
"""
Base.@propagate_inbounds dss_untransform(
    ::Type{T},
    targ,
    local_geometry,
    I,
) where {T} = dss_untransform(T, targ, local_geometry[I])
@inline dss_untransform(::Type{T}, targ, local_geometry::Nothing, I) where {T} =
    dss_untransform(T, targ, local_geometry)
@inline dss_untransform(
    ::Type{T},
    targ::AutoBroadcaster,
    local_geometry::Geometry.LocalGeometry,
) where {T <: AutoBroadcaster} =
    broadcast(zero(T), targ) do tzero, targ
        dss_untransform(typeof(tzero), targ, local_geometry)
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

# helper functions for DSS2

function _representative_slab(
    data::Union{DataLayouts.AbstractData, Nothing},
    ::Type{DA},
) where {DA}
    rebuild_flag = DA isa Array ? false : true
    if isnothing(data)
        return nothing
    elseif rebuild_flag
        return DataLayouts.rebuild(
            slab(data, CartesianIndex(1, 1, 1, 1, 1)),
            Array,
        )
    else
        return slab(data, CartesianIndex(1, 1, 1, 1, 1))
    end
end

_transformed_type(
    data::DataLayouts.AbstractData,
    local_geometry::Union{DataLayouts.AbstractData, Nothing},
    dss_weights::Union{DataLayouts.AbstractData, Nothing},
    ::Type{DA},
) where {DA} = typeof(
    dss_transform(
        _representative_slab(data, DA),
        _representative_slab(local_geometry, DA),
        _representative_slab(dss_weights, DA),
        CartesianIndex(1, 1, 1, 1, 1),
    ),
)

# currently only used in limiters (but not actually functional)
# see https://github.com/CliMA/ClimaCore.jl/issues/1511
struct GhostBuffer{G, D}
    graph_context::G
    send_data::D
    recv_data::D
end

recv_buffer(ghost::GhostBuffer) = ghost.recv_data

create_ghost_buffer(data, topology::Topologies.AbstractTopology) = nothing

create_ghost_buffer(
    data::Union{DataLayouts.IJFH{S, Nij}, DataLayouts.VIJFH{S, <:Any, Nij}},
    topology::Topologies.Topology2D,
) where {S, Nij} = create_ghost_buffer(
    data,
    topology,
    Topologies.nsendelems(topology),
    Topologies.nrecvelems(topology),
)


function create_ghost_buffer(
    data::Union{DataLayouts.IJFH{S, Nij}, DataLayouts.VIJFH{S, <:Any, Nij}},
    topology::Topologies.Topology2D,
    Nhsend,
    Nhrec,
) where {S, Nij}
    if data isa DataLayouts.IJFH
        send_data = DataLayouts.IJFH{S, Nij}(typeof(parent(data)), Nhsend)
        recv_data = DataLayouts.IJFH{S, Nij}(typeof(parent(data)), Nhrec)
    else
        Nv = DataLayouts.nlevels(data)
        Nf = DataLayouts.ncomponents(data)
        send_data = DataLayouts.VIJFH{S, Nv, Nij}(
            similar(parent(data), (Nv, Nij, Nij, Nf, Nhsend)),
        )
        recv_data = DataLayouts.VIJFH{S, Nv, Nij}(
            similar(parent(data), (Nv, Nij, Nij, Nf, Nhrec)),
        )
    end
    k = stride(parent(send_data), DataLayouts.h_dim(data))

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
