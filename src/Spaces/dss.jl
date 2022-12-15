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

function dss_1d!(
    htopology::Topologies.AbstractTopology,
    data,
    local_geometry_data = nothing,
    dss_weights = nothing,
)
    Nq = size(data, 1)
    Nv = size(data, 4)
    idx1 = CartesianIndex(1, 1, 1, 1, 1)
    idx2 = CartesianIndex(Nq, 1, 1, 1, 1)
    @inbounds for (elem1, face1, elem2, face2, reversed) in
                  Topologies.interior_faces(htopology)
        for level in 1:Nv
            @assert face1 == 1 && face2 == 2 && !reversed
            local_geometry_slab1 = slab(local_geometry_data, level, elem1)
            weight_slab1 = slab(dss_weights, level, elem1)
            data_slab1 = slab(data, level, elem1)

            local_geometry_slab2 = slab(local_geometry_data, level, elem2)
            weight_slab2 = slab(dss_weights, level, elem2)
            data_slab2 = slab(data, level, elem2)
            val =
                dss_transform(
                    data_slab1,
                    local_geometry_slab1,
                    weight_slab1,
                    idx1,
                ) ⊞ dss_transform(
                    data_slab2,
                    local_geometry_slab2,
                    weight_slab2,
                    idx2,
                )

            data_slab1[idx1] = dss_untransform(
                eltype(data_slab1),
                val,
                local_geometry_slab1,
                idx1,
            )
            data_slab2[idx2] = dss_untransform(
                eltype(data_slab2),
                val,
                local_geometry_slab2,
                idx2,
            )
        end
    end
    return data
end

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

"""
    fill_send_buffer!(topology, data, ghost_buffer)

Fill the send buffer of `ghost_buffer` with the necessary data from `data`.

Part of [`Spaces.weighted_dss!`](@ref).
"""
function fill_send_buffer!(
    topology::Topologies.Topology2D,
    data::DataLayouts.AbstractData,
    ghost_buffer::GhostBuffer,
)

    Nv = size(data, 4)
    send_data = ghost_buffer.send_data
    @inbounds for (sidx, lidx) in enumerate(topology.send_elem_lidx)
        for level in 1:Nv
            src_slab = slab(data, level, lidx)
            send_slab = slab(send_data, level, sidx)
            copyto!(send_slab, src_slab)
        end
    end
    return nothing
end

"""
    dss_interior_faces!(topology, data [, local_geometry_data=nothing, local_weights=nothing])

Perform DSS on the local interior faces of the topology.

If `local_geometry` is `nothing`, no transformations are applied to vectors.
If `local_weights` is `nothing`, no weighting is applied (i.e. colocated values are summed).

Part of [`Spaces.weighted_dss!`](@ref).
"""
function dss_interior_faces!(
    topology,
    data,
    local_geometry_data = nothing,
    local_weights = nothing,
)
    Nq = size(data, 1)
    Nv = size(data, 4)

    @inbounds for (lidx1, face1, lidx2, face2, reversed) in
                  Topologies.interior_faces(topology)
        for level in 1:Nv

            data_slab1 = slab(data, level, lidx1)
            data_slab2 = slab(data, level, lidx2)
            local_geometry_slab1 = slab(local_geometry_data, level, lidx1)
            local_geometry_slab2 = slab(local_geometry_data, level, lidx2)
            weight_slab1 = slab(local_weights, level, lidx1)
            weight_slab2 = slab(local_weights, level, lidx2)
            for q in 2:(Nq - 1)
                i1, j1 = Topologies.face_node_index(face1, Nq, q, false)
                i2, j2 = Topologies.face_node_index(face2, Nq, q, reversed)
                val =
                    dss_transform(
                        data_slab1,
                        local_geometry_slab1,
                        weight_slab1,
                        i1,
                        j1,
                    ) ⊞ dss_transform(
                        data_slab2,
                        local_geometry_slab2,
                        weight_slab2,
                        i2,
                        j2,
                    )
                data_slab1[i1, j1] = dss_untransform(
                    eltype(data_slab1),
                    val,
                    local_geometry_slab1,
                    i1,
                    j1,
                )
                data_slab2[i2, j2] = dss_untransform(
                    eltype(data_slab2),
                    val,
                    local_geometry_slab2,
                    i2,
                    j2,
                )
            end
        end
    end
    return nothing
end

"""
    dss_local_vertices!(topology, data[, local_geometry_data=nothing, local_weights=nothing])

Perform DSS on the local vertices of the topology.

Part of [`Spaces.weighted_dss!`](@ref).
"""
function dss_local_vertices!(
    topology,
    data,
    local_geometry_data = nothing,
    local_weights = nothing,
)
    Nq = size(data, 1)
    Nv = size(data, 4)

    @inbounds for vertex in Topologies.local_vertices(topology)
        # for each level
        for level in 1:Nv
            # gather: compute sum over shared vertices
            sum_data = mapreduce(⊞, vertex) do (lidx, vert)
                i, j = Topologies.vertex_node_index(vert, Nq)
                data_slab = slab(data, level, lidx)
                local_geometry_slab = slab(local_geometry_data, level, lidx)
                weight_slab = slab(local_weights, level, lidx)
                dss_transform(data_slab, local_geometry_slab, weight_slab, i, j)
            end

            # scatter: assign sum to shared vertices
            for (lidx, vert) in vertex
                data_slab = slab(data, level, lidx)
                i, j = Topologies.vertex_node_index(vert, Nq)
                local_geometry_slab = slab(local_geometry_data, level, lidx)
                data_slab[i, j] = dss_untransform(
                    eltype(data_slab),
                    sum_data,
                    local_geometry_slab,
                    i,
                    j,
                )
            end
        end
    end
end

"""
    dss_ghost_faces!(topology, data, ghost_data,
        local_geometry_data=nothing, ghost_geometry_data=nothing,
        local_weights=nothing ghost_weights=nothing;
        update_ghost=false)

Perform DSS on the ghost faces of the topology. `ghost_data` should contain the ghost element data.

Part of [`Spaces.weighted_dss!`](@ref).
"""
function dss_ghost_faces!(
    topology,
    data,
    recv_buf,
    local_geometry_data = nothing,
    ghost_geometry_data = nothing,
    local_weights = nothing,
    ghost_weights = nothing;
    update_ghost = false,
)
    Nq = size(data, 1)
    Nv = size(data, 4)

    @inbounds for (lidx1, face1, ridx2, face2, reversed) in
                  Topologies.ghost_faces(topology)
        for level in 1:Nv

            data_slab1 = slab(data, level, lidx1)
            data_slab2 = slab(recv_buf, level, ridx2)
            local_geometry_slab1 = slab(local_geometry_data, level, lidx1)
            local_geometry_slab2 = slab(ghost_geometry_data, level, ridx2)
            weight_slab1 = slab(local_weights, level, lidx1)
            weight_slab2 = slab(ghost_weights, level, ridx2)
            for q in 2:(Nq - 1)
                i1, j1 = Topologies.face_node_index(face1, Nq, q, false)
                i2, j2 = Topologies.face_node_index(face2, Nq, q, reversed)
                val =
                    dss_transform(
                        data_slab1,
                        local_geometry_slab1,
                        weight_slab1,
                        i1,
                        j1,
                    ) ⊞ dss_transform(
                        data_slab2,
                        local_geometry_slab2,
                        weight_slab2,
                        i2,
                        j2,
                    )
                data_slab1[i1, j1] = dss_untransform(
                    eltype(data_slab1),
                    val,
                    local_geometry_slab1,
                    i1,
                    j1,
                )
                if update_ghost
                    data_slab2[i2, j2] = dss_untransform(
                        eltype(data_slab2),
                        val,
                        local_geometry_slab2,
                        i2,
                        j2,
                    )
                end
            end
        end
    end
end

"""
    dss_ghost_vertices!(topology, data, ghost_data,
        local_geometry_data=nothing, ghost_geometry_data=nothing,
        local_weights=nothing ghost_weights=nothing;
        update_ghost=false)

Perform DSS on the ghost faces of the topology. `ghost_data` should contain the ghost element data.

Part of [`Spaces.weighted_dss!`](@ref).
"""
function dss_ghost_vertices!(
    topology,
    data,
    recv_buf,
    local_geometry_data = nothing,
    ghost_geometry_data = nothing,
    local_weights = nothing,
    ghost_weights = nothing;
    update_ghost = false,
)
    Nq = size(data, 1)
    Nv = size(data, 4)

    @inbounds for vertex in Topologies.ghost_vertices(topology)
        # for each level
        for level in 1:Nv
            # gather: compute sum over shared vertices
            sum_data = mapreduce(⊞, vertex) do (isghost, idx, vert)
                i, j = Topologies.vertex_node_index(vert, Nq)
                if isghost
                    ridx = idx
                    src_slab = slab(recv_buf, level, ridx)
                    local_geometry_slab =
                        slab(ghost_geometry_data, level, ridx)
                    weight_slab = slab(ghost_weights, level, ridx)
                    dss_transform(
                        src_slab,
                        local_geometry_slab,
                        weight_slab,
                        i,
                        j,
                    )
                else
                    lidx = idx
                    src_slab = slab(data, level, lidx)
                    local_geometry_slab =
                        slab(local_geometry_data, level, lidx)
                    weight_slab = slab(local_weights, level, lidx)
                    dss_transform(
                        src_slab,
                        local_geometry_slab,
                        weight_slab,
                        i,
                        j,
                    )
                end
            end

            # scatter: assign sum to shared vertices
            for (isghost, idx, vert) in vertex
                if isghost
                    if !update_ghost
                        continue
                    end
                    ridx = idx
                    i, j = Topologies.vertex_node_index(vert, Nq)
                    dest_slab = slab(recv_buf, level, ridx)
                    local_geometry_slab = slab(ghost_geometry_data, level, ridx)
                    dest_slab[i, j] = dss_untransform(
                        eltype(dest_slab),
                        sum_data,
                        local_geometry_slab,
                        i,
                        j,
                    )
                else
                    lidx = idx
                    dest_slab = slab(data, level, lidx)
                    i, j = Topologies.vertex_node_index(vert, Nq)
                    local_geometry_slab = slab(local_geometry_data, level, lidx)
                    dest_slab[i, j] = dss_untransform(
                        eltype(dest_slab),
                        sum_data,
                        local_geometry_slab,
                        i,
                        j,
                    )
                end
            end
        end
    end
end

function dss_2d!(
    topology,
    data,
    ghost_buffer,
    local_geometry_data = nothing,
    ghost_geometry_data = nothing,
    local_weights = nothing,
    ghost_weights = nothing,
)


    if ghost_buffer isa GhostBuffer
        # 1) copy send data to buffer
        fill_send_buffer!(topology, data, ghost_buffer)
        # 2) start communication
        ClimaComms.start(ghost_buffer.graph_context)
    end

    dss_interior_faces!(topology, data, local_geometry_data, local_weights)

    # 4) progress communication
    if ghost_buffer isa GhostBuffer
        ClimaComms.progress(ghost_buffer.graph_context)
    end

    dss_local_vertices!(topology, data, local_geometry_data, local_weights)

    # 6) complete communication
    if ghost_buffer isa GhostBuffer
        ClimaComms.finish(ghost_buffer.graph_context)
    end

    # 7) DSS over ghost faces
    if ghost_buffer isa GhostBuffer
        recv_buf = recv_buffer(ghost_buffer)
    else
        recv_buf = ghost_buffer
    end

    dss_ghost_faces!(
        topology,
        data,
        recv_buf,
        local_geometry_data,
        ghost_geometry_data,
        local_weights,
        ghost_weights,
    )
    dss_ghost_vertices!(
        topology,
        data,
        recv_buf,
        local_geometry_data,
        ghost_geometry_data,
        local_weights,
        ghost_weights,
    )

    # 8) DSS over ghost vertices
    return data
end

"""
    weighted_dss!(data, space, ghost_buffer = nothing)

Computes weighted dss of `data`. This function consists of 

1). `weighted_dss_start!` which loads the send buffer and starts communications,

2). `weighted_dss_internal!` which progresses the communication and performs dss operations on interior vertices and faces, and

3). `weighted_dss_ghost!` which completes the communication and performs dss on ghost vertices and faces

These constituent functions can also be called separately for increased overlap between computation and communication when merging multiple dss
operations.

"""
function weighted_dss!(data, space, ghost_buffer = nothing)
    if isnothing(ghost_buffer)
        topology =
            space isa ExtrudedFiniteDifferenceSpace ?
            space.horizontal_space.topology : space.topology
        ghost_buffer = create_ghost_buffer(data, topology)
    end
    weighted_dss_start!(data, space, ghost_buffer)
    weighted_dss_internal!(data, space, ghost_buffer)
    weighted_dss_ghost!(data, space, ghost_buffer)
end

"""
    weighted_dss_start!(data, space, ghost_buffer)

Create the ghost buffer, if necessary, load the send buffer and start communication.

Part of [`Spaces.weighted_dss!`](@ref).
"""
weighted_dss_start!(data, space, ghost_buffer) =
    weighted_dss_start!(data, space, horizontal_space(space), ghost_buffer)

weighted_dss_start!(
    data,
    space::ExtrudedFiniteDifferenceSpace{S, H},
    hspace,
    ghost_buffer,
) where {S, H <: SpectralElementSpace2D{<:Topology2D}} =
    _weighted_dss_start!(data, space, hspace, ghost_buffer)

weighted_dss_start!(
    data,
    space::H,
    hspace,
    ghost_buffer,
) where {H <: SpectralElementSpace2D{<:Topology2D}} =
    _weighted_dss_start!(data, space, hspace, ghost_buffer)

function weighted_dss_start!(data, space, hspace, ghost_buffer)
    return nothing
end

function _weighted_dss_start!(data, space, hspace, ghost_buffer)
    topology = hspace.topology
    if ghost_buffer isa GhostBuffer
        # 1) copy send data to buffer
        fill_send_buffer!(topology, data, ghost_buffer)
        # 2) start communication
        ClimaComms.start(ghost_buffer.graph_context)
    end
    return nothing
end

"""
    weighted_dss_internal!(data, space, ghost_buffer)

Perform weighted dss for internal faces and vertices. Progress the communication.

Part of [`Spaces.weighted_dss!`](@ref).
"""
function weighted_dss_internal!(data, space, ghost_buffer)
    hspace =
        space isa ExtrudedFiniteDifferenceSpace ? space.horizontal_space : space
    topology = hspace.topology
    local_geometry = local_geometry_data(space)
    if hspace isa SpectralElementSpace1D
        dss_1d!(topology, data, local_geometry, hspace.dss_weights)
    else
        local_weights = hspace.local_dss_weights
        # 3) dss for interior faces
        dss_interior_faces!(topology, data, local_geometry, local_weights)
        # 4) progress communication
        if ghost_buffer isa GhostBuffer
            ClimaComms.progress(ghost_buffer.graph_context)
        end
        # 5) dss for interior local vertices
        dss_local_vertices!(topology, data, local_geometry, local_weights)
    end
    return nothing
end

"""
    weighted_dss_ghost!(data, space, ghost_buffer)

Complete communication. Perform weighted dss for ghost faces and vertices.

Part of [`Spaces.weighted_dss!`](@ref).
"""
weighted_dss_ghost!(data, space, ghost_buffer) =
    weighted_dss_ghost!(data, space, horizontal_space(space), ghost_buffer)

function weighted_dss_ghost!(
    data,
    space::ExtrudedFiniteDifferenceSpace{S, H},
    hspace,
    ghost_buffer,
) where {S, H <: SpectralElementSpace2D{<:Topology2D}}
    _weighted_dss_ghost!(data, space, hspace, ghost_buffer)
end

function weighted_dss_ghost!(
    data,
    space::H,
    hspace,
    ghost_buffer,
) where {H <: SpectralElementSpace2D{<:Topology2D}}
    _weighted_dss_ghost!(data, space, hspace, ghost_buffer)
end

function _weighted_dss_ghost!(data, space, hspace, ghost_buffer)
    topology = hspace.topology
    local_geometry = local_geometry_data(space)
    local_weights = hspace.local_dss_weights
    ghost_geometry = ghost_geometry_data(space)
    ghost_weights = hspace.ghost_dss_weights
    # 6) complete communication
    if ghost_buffer isa GhostBuffer
        ClimaComms.finish(ghost_buffer.graph_context)
    end

    if ghost_buffer isa GhostBuffer
        recv_buf = recv_buffer(ghost_buffer)
    else
        recv_buf = ghost_buffer
    end

    # 7) DSS over ghost faces
    dss_ghost_faces!(
        topology,
        data,
        recv_buf,
        local_geometry,
        ghost_geometry,
        local_weights,
        ghost_weights,
    )
    # 8) DSS over ghost vertices
    dss_ghost_vertices!(
        topology,
        data,
        recv_buf,
        local_geometry,
        ghost_geometry,
        local_weights,
        ghost_weights,
    )
    return data
end

function weighted_dss_ghost!(data, space, hspace, ghost_buffer)
    return data
end
