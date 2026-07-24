"""
    dss_transform(arg, local_geometry, weight, I)

Transfrom `arg[I]` to a basis for direct stiffness summation (DSS).
Transformations only apply to vector quantities.

- `local_geometry[I]` is the relevant `LocalGeometry` object. If it is `nothing`, then no transformation is performed
- `weight[I]` is the relevant DSS weights. If `weight` is `nothing`, then the result is simply summation.

See [`ClimaCore.Spaces.weighted_dss!`](@ref).
"""
Base.@propagate_inbounds dss_transform(arg, local_geometry, weight, I) =
    dss_transform(
        arg[I],
        local_geometry[I],
        # DSS weights only vary in the horizontal, so their level index is 1.
        weight[CartesianIndex(1, Base.tail(Tuple(I))...)],
    )
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

@inline dss_transform(
    arg,
    local_geometry::Geometry.LocalGeometry,
    weight,
) = arg * weight
@inline dss_transform(
    arg::AutoBroadcaster,
    local_geometry::Geometry.LocalGeometry,
    weight,
) =
    nested_broadcast(arg) do leaf
        dss_transform(leaf, local_geometry, weight)
    end
@inline dss_transform(
    arg::Geometry.OrthonormalTensor,
    local_geometry::Geometry.LocalGeometry,
    weight,
) = arg * weight
const NonTransformedAxis =
    Union{Geometry.Covariant3Axis, Geometry.Contravariant3Axis}
@inline dss_transform(
    arg::Geometry.Tensor{1, <:Any, <:Tuple{<:NonTransformedAxis}},
    local_geometry::Geometry.LocalGeometry,
    weight,
) = arg * weight
@inline function dss_transform(
    arg::Geometry.AbstractTensor{1},
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
        u₁, v = parent(arg)
        uw_vector = Geometry.project(
            Geometry.UWAxis(),
            Geometry.Covariant13Vector(u₁, zero(u₁)),
            local_geometry,
        )
        u, w = parent(uw_vector)
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

@inline dss_untransform(::Type{T}, targ::T, local_geometry) where {T} = targ
@inline dss_untransform(
    ::Type{T},
    targ::AutoBroadcaster,
    local_geometry::Geometry.LocalGeometry,
) where {T <: AutoBroadcaster} =
    nested_broadcast(zero(T), targ) do zero_value, targ
        dss_untransform(typeof(zero_value), targ, local_geometry)
    end

@inline dss_untransform(
    ::Type{T},
    targ::T,
    local_geometry::Geometry.LocalGeometry,
) where {T <: Geometry.AbstractTensor{1}} = targ
@inline function dss_untransform(
    ::Type{Geometry.Tensor{1, T, Tuple{B}, S}},
    targ::Geometry.AbstractTensor{1},
    local_geometry::Geometry.LocalGeometry,
) where {T, B <: Geometry.Components, S}
    # If `targ` already has the destination basis, dss_transform left it
    # untouched and there is nothing to undo. (Required so the workaround
    # below — which assumes dss_transform turned the input into a UVWVector —
    # doesn't fire when no transform happened.)
    targ isa Geometry.Tensor{1, T, Tuple{B}, S} && return targ
    ax = B()
    # workaround for using a Covariant12Vector in a UW space
    if (
        axes(local_geometry.∂x∂ξ, 1) isa Geometry.UWAxis &&
        ax isa Geometry.Covariant12Axis
    )
        u, u₂, w = parent(targ)
        u₁_vector = Geometry.transform(
            Geometry.Covariant1Axis(),
            Geometry.UWVector(u, w),
            local_geometry,
        )
        u₁, = parent(u₁_vector)
        return Geometry.Covariant12Vector(u₁, u₂)
    end
    Geometry.project(ax, targ, local_geometry)
end

# currently only used in limiters (but not actually functional)
# see https://github.com/CliMA/ClimaCore.jl/issues/1511
struct GhostBuffer{G, D}
    graph_context::G
    send_data::D
    recv_data::D
end

recv_buffer(ghost::GhostBuffer) = ghost.recv_data

create_ghost_buffer(data, topology::AbstractTopology) = nothing

function create_ghost_buffer(
    data::DataLayouts.VIJHWithF,
    topology::Topology2D,
    Nhsend = nsendelems(topology),
    Nhrec = nrecvelems(topology),
)
    # Ghost exchange is only required for distributed topologies
    ClimaComms.context(topology) isa ClimaComms.SingletonCommsContext &&
        return nothing
    send_data = similar(data, Base.setindex(size(data), Nhsend, 4))
    recv_data = similar(data, Base.setindex(size(data), Nhrec, 4))
    k = stride(parent(send_data), DataLayouts.f_dim(data) == 5 ? 4 : 5)
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
