"""
    DSSBuffer{T, G, D, A, B, VI}

Buffers and index maps for the direct stiffness summation (DSS) of one data layout
with element type `T` on a [`Topology2D`](@ref). Construct with
`create_dss_buffer(data, topology, local_geometry = nothing, dss_weights = nothing)`.

# Fields

  - `graph_context`: `ClimaComms` graph context for communication.
  - `perimeter_data`: `DataLayout` holding the (transformed) values at the perimeter
    nodes of each element.
  - `send_data`: Send buffer, an `AbstractVector{FT}`.
  - `recv_data`: Receive buffer, an `AbstractVector{FT}`.
  - `send_buf_idx`: Indexing array for loading the send buffer from `perimeter_data`.
  - `recv_buf_idx`: Indexing array for adding the receive buffer into
    `perimeter_data`.
  - `internal_elems`: Local elements (`lidx`) that touch no ghost element.
  - `perimeter_elems`: Local elements (`lidx`) on the process boundary.
"""
struct DSSBuffer{T, G, D, A, B, VI}
    graph_context::G
    perimeter_data::D
    send_data::A
    recv_data::A
    send_buf_idx::B
    recv_buf_idx::B
    internal_elems::VI
    perimeter_elems::VI
end

function create_dss_buffer(
    data::DataLayouts.VIJHWithF,
    topology::Topology2D,
    local_geometry = nothing,
    dss_weights = nothing,
)
    context = ClimaComms.context(topology)
    DA = ClimaComms.array_type(topology)
    (Nv, Nij, _, Nh) = size(data)
    Np = length(Perimeter2D(Nij))
    FT = eltype(parent(data))
    data_type = eltype(Base.broadcastable(data))
    W = isnothing(dss_weights) ? Nothing : eltype(dss_weights)
    T =
        isnothing(local_geometry) ? data_type :
        return_type(dss_transform, Tuple{data_type, eltype(local_geometry), W})
    Nf = DataLayouts.num_basetypes(FT, T)
    perimeter_data = DataLayouts.layout_constructor(data, T; Ni = Np, Nj = 1)(DA{FT}, Nh)
    if context isa ClimaComms.SingletonCommsContext
        graph_context = ClimaComms.SingletonGraphContext(context)
        send_data = recv_data = FT[]
        send_buf_idx = recv_buf_idx = Int[]
        # internal_elems and perimeter_elems are indexed by the DSS kernels, so
        # they must be device arrays (as in the multi-process branch below); the
        # host send/recv buffer indices are only used off-device and stay host.
        perimeter_elems = DA(Int[])
        internal_elems = DA(collect(Base.OneTo(nelems(topology))))
    else
        (; comm_vertex_lengths, comm_face_lengths) = topology
        vertex_buffer_lengths = comm_vertex_lengths .* (Nv * Nf)
        face_buffer_lengths = comm_face_lengths .* (Nv * Nf * (Nij - 2))
        buffer_lengths = vertex_buffer_lengths .+ face_buffer_lengths
        buffer_size = sum(buffer_lengths)
        send_data = DA{FT}(undef, buffer_size)
        recv_data = DA{FT}(undef, buffer_size)
        neighbor_pids = topology.neighbor_pids
        graph_context = ClimaComms.graph_context(
            context,
            send_data,
            buffer_lengths,
            neighbor_pids,
            recv_data,
            buffer_lengths,
            neighbor_pids,
            persistent = true,
        )
        send_buf_idx, recv_buf_idx = compute_ghost_send_recv_idx(topology, Nij)
        internal_elems = DA(topology.internal_elems)
        perimeter_elems = DA(topology.perimeter_elems)
    end
    G = typeof(graph_context)
    D = typeof(perimeter_data)
    A = typeof(send_data)
    B = typeof(send_buf_idx)
    VI = typeof(perimeter_elems)
    return DSSBuffer{eltype(data), G, D, A, B, VI}(
        graph_context,
        perimeter_data,
        send_data,
        recv_data,
        send_buf_idx,
        recv_buf_idx,
        internal_elems,
        perimeter_elems,
    )
end

Base.eltype(::DSSBuffer{T}) where {T} = T

"""
    dss_transform!(device, dss_buffer, data, local_geometry, dss_weights, perimeter, localelems)

Transform vectors in `data` from covariant/contravariant axes to physical axes,
weight the data at perimeter nodes, and store the result in
`dss_buffer.perimeter_data`.

# Arguments

  - `dss_buffer`: [`DSSBuffer`](@ref) created by `create_dss_buffer` for `data`.
  - `data`: Field data.
  - `local_geometry`: Local metric information at each node.
  - `dss_weights`: DSS weights of the horizontal space.
  - `perimeter`: Perimeter iterator.
  - `localelems`: Local elements on which to perform the transformation.

Part of [`ClimaCore.Spaces.weighted_dss!`](@ref).
"""
dss_transform!(
    device::ClimaComms.AbstractDevice,
    (; perimeter_data)::DSSBuffer,
    data::DataLayouts.VIJHWithF,
    local_geometry::DataLayouts.VIJHWithF,
    dss_weights::DataLayouts.VIJHWithF,
    perimeter::Perimeter2D,
    localelems,
) =
    dss_transform!(
        device,
        perimeter_data,
        Base.broadcastable(data),
        perimeter,
        local_geometry,
        dss_weights,
        localelems,
    )

dss_transform!(
    ::ClimaComms.AbstractCPUDevice,
    perimeter_data::DataLayouts.VIJHWithF,
    data::DataLayouts.VIJHWithF,
    perimeter::Perimeter2D,
    local_geometry::DataLayouts.VIJHWithF,
    dss_weights::DataLayouts.VIJHWithF,
    localelems,
) =
    @inbounds for h in localelems, (p, (i, j)) in enumerate(perimeter), v in axes(data, 1)
        # dss_weights only vary in the horizontal, so their level index is 1
        perimeter_data[v, p, 1, h] = dss_transform(
            data[v, i, j, h],
            local_geometry[v, i, j, h],
            dss_weights[1, i, j, h],
        )
    end

"""
    dss_untransform!(device, dss_buffer, data, local_geometry, perimeter, localelems)

Transform physical vectors in `dss_buffer.perimeter_data` back to their original
covariant/contravariant axes, and store the result in `data`.

# Arguments

  - `dss_buffer`: [`DSSBuffer`](@ref) created by `create_dss_buffer` for `data`.
  - `data`: Field data.
  - `local_geometry`: Local metric information at each node.
  - `perimeter`: Perimeter iterator.
  - `localelems`: Local elements on which to perform the transformation.

Part of [`ClimaCore.Spaces.weighted_dss!`](@ref).
"""
dss_untransform!(
    device::ClimaComms.AbstractDevice,
    (; perimeter_data)::DSSBuffer,
    data::DataLayouts.VIJHWithF,
    local_geometry::DataLayouts.VIJHWithF,
    perimeter::Perimeter2D,
    localelems,
) =
    dss_untransform!(
        device,
        perimeter_data,
        Base.broadcastable(data),
        local_geometry,
        perimeter,
        localelems,
    )

dss_untransform!(
    ::ClimaComms.AbstractCPUDevice,
    perimeter_data::DataLayouts.VIJHWithF,
    data::DataLayouts.VIJHWithF,
    local_geometry::DataLayouts.VIJHWithF,
    perimeter::Perimeter2D,
    localelems,
) =
    @inbounds for h in localelems, (p, (i, j)) in enumerate(perimeter), v in axes(data, 1)
        data[v, i, j, h] = dss_untransform(
            eltype(data),
            perimeter_data[v, p, 1, h],
            local_geometry[v, i, j, h],
        )
    end

dss_load_perimeter_data!(
    ::ClimaComms.AbstractCPUDevice,
    (; perimeter_data)::DSSBuffer,
    data::DataLayouts.VIJHWithF,
    perimeter::Perimeter2D,
) =
    @inbounds for index in CartesianIndices(perimeter_data)
        (v, p, _, h) = index.I
        (i, j) = perimeter[p]
        perimeter_data[v, p, 1, h] = data[v, i, j, h]
    end

dss_unload_perimeter_data!(
    ::ClimaComms.AbstractCPUDevice,
    data::DataLayouts.VIJHWithF,
    (; perimeter_data)::DSSBuffer,
    perimeter::Perimeter2D,
) =
    @inbounds for index in CartesianIndices(perimeter_data)
        (v, p, _, h) = index.I
        (i, j) = perimeter[p]
        data[v, i, j, h] = perimeter_data[v, p, 1, h]
    end

"""
    dss_local!(device, perimeter_data, perimeter, topology)

Perform DSS on the local vertices and interior faces of `topology`: the values in
`perimeter_data` at nodes shared between local elements are replaced by their sum.

Part of [`ClimaCore.Spaces.weighted_dss!`](@ref).
"""
function dss_local!(
    ::ClimaComms.AbstractCPUDevice,
    perimeter_data::DataLayouts.VIJHWithF,
    perimeter::Perimeter2D,
    topology::Topology2D,
)
    @inbounds for vertex in local_vertices(topology), v in axes(perimeter_data, 1)
        # Accumulate in a loop instead of calling sum with a closure, since the
        # empty-collection error path of sum contains a runtime dispatch.
        sum_data = zero(eltype(perimeter_data))
        for (h, vert) in vertex
            p = perimeter_vertex_node_index(vert)
            sum_data += perimeter_data[v, p, 1, h]
        end
        for (h, vert) in vertex
            p = perimeter_vertex_node_index(vert)
            perimeter_data[v, p, 1, h] = sum_data
        end
    end
    @inbounds for (h1, face1, h2, face2, reversed) in interior_faces(topology)
        nfacedof = length(perimeter) ÷ 4 - 1
        pr1 = perimeter_face_indices(face1, nfacedof, false)
        pr2 = perimeter_face_indices(face2, nfacedof, reversed)
        for (p1, p2) in zip(pr1, pr2), v in axes(perimeter_data, 1)
            sum_data = perimeter_data[v, p1, 1, h1] + perimeter_data[v, p2, 1, h2]
            perimeter_data[v, p1, 1, h1] = sum_data
            perimeter_data[v, p2, 1, h2] = sum_data
        end
    end
end

"""
    dss_local_ghost!(device, perimeter_data, perimeter, topology)

Compute the "local" part of the ghost-vertex DSS: for each unique ghost vertex, sum
the values of all its local vertices and store the sum at each of those local vertex
locations in `perimeter_data`.

Part of [`ClimaCore.Spaces.weighted_dss!`](@ref).
"""
dss_local_ghost!(
    ::ClimaComms.AbstractCPUDevice,
    perimeter_data::DataLayouts.VIJHWithF,
    perimeter::Perimeter2D,
    topology::Topology2D,
) =
    @inbounds for vertex in ghost_vertices(topology), v in axes(perimeter_data, 1)
        # Accumulate in a loop instead of calling sum with a closure, since the
        # empty-collection error path of sum contains a runtime dispatch.
        sum_data = zero(eltype(perimeter_data))
        for (isghost, h, vert) in vertex
            isghost && continue
            p = perimeter_vertex_node_index(vert)
            sum_data += perimeter_data[v, p, 1, h]
        end
        for (isghost, h, vert) in vertex
            isghost && continue
            p = perimeter_vertex_node_index(vert)
            perimeter_data[v, p, 1, h] = sum_data
        end
    end

"""
    dss_ghost!(device, perimeter_data, perimeter, topology)

Set the value in `perimeter_data` of all local vertices of each unique ghost vertex
to that of the representative vertex.

Part of [`ClimaCore.Spaces.weighted_dss!`](@ref).
"""
dss_ghost!(
    ::ClimaComms.AbstractCPUDevice,
    perimeter_data::DataLayouts.VIJHWithF,
    perimeter::Perimeter2D,
    topology::Topology2D,
) =
    @inbounds for (vertex_index, vertex) in enumerate(ghost_vertices(topology))
        h_result, vert_result = topology.repr_ghost_vertex[vertex_index]
        p_result = perimeter_vertex_node_index(vert_result)
        for v in axes(perimeter_data, 1)
            result = perimeter_data[v, p_result, 1, h_result]
            for (isghost, h, vert) in vertex
                isghost && continue
                p = perimeter_vertex_node_index(vert)
                perimeter_data[v, p, 1, h] = result
            end
        end
    end

"""
    fill_send_buffer!(device, dss_buffer)

Load the send buffer of `dss_buffer` from its `perimeter_data`. For unique ghost
vertices, only the representative vertices, which hold the result of the "ghost
local" DSS, are loaded.

Part of [`ClimaCore.Spaces.weighted_dss!`](@ref).
"""
function fill_send_buffer!(
    ::ClimaComms.AbstractCPUDevice,
    (; perimeter_data, send_data, send_buf_idx)::DSSBuffer,
)
    isempty(send_buf_idx) && return nothing
    buffer_index = 1
    @inbounds for (h, p) in eachrow(send_buf_idx), v in axes(perimeter_data, 1)
        DataLayouts.set_struct!(send_data, perimeter_data[v, p, 1, h], buffer_index, Val(1))
        buffer_index += DataLayouts.ncomponents(perimeter_data)
    end
end

"""
    load_from_recv_buffer!(device, dss_buffer)

Add the data in the receive buffer of `dss_buffer` to the corresponding locations
in its `perimeter_data`. For ghost vertices, the data is added only to the
representative vertices; `dss_ghost!` then scatters the values to the other local
vertices of each unique ghost vertex.

Part of [`ClimaCore.Spaces.weighted_dss!`](@ref).
"""
function load_from_recv_buffer!(
    ::ClimaComms.AbstractCPUDevice,
    (; perimeter_data, recv_data, recv_buf_idx)::DSSBuffer,
)
    isempty(recv_buf_idx) && return nothing
    buffer_index = 1
    @inbounds for (h, p) in eachrow(recv_buf_idx), v in axes(perimeter_data, 1)
        perimeter_data[v, p, 1, h] +=
            DataLayouts.get_struct(recv_data, eltype(perimeter_data), buffer_index, Val(1))
        buffer_index += DataLayouts.ncomponents(perimeter_data)
    end
end

"""
    dss!(data, topology)

Perform unweighted DSS of `data` in place: values at nodes shared between elements
are replaced by their sum. Returns `nothing`.
"""
function dss!(data::DataLayouts.VIJHWithF, topology::IntervalTopology)
    sizeof(eltype(data)) > 0 || return nothing
    device = ClimaComms.device(topology)
    dss_1d!(device, Base.broadcastable(data), topology)
    return nothing
end
function dss!(data::DataLayouts.VIJHWithF, topology::Topology2D)
    sizeof(eltype(data)) > 0 || return nothing
    device = ClimaComms.device(topology)
    perimeter = Perimeter2D(size(data, 2))
    # create dss buffer
    dss_buffer = create_dss_buffer(data, topology)
    # load perimeter data from data
    dss_load_perimeter_data!(device, dss_buffer, data, perimeter)
    # compute local dss for ghost dof
    dss_local_ghost!(device, dss_buffer.perimeter_data, perimeter, topology)
    # load send buffer
    fill_send_buffer!(device, dss_buffer)
    # initiate communication
    ClimaComms.start(dss_buffer.graph_context)
    # compute local dss
    dss_local!(device, dss_buffer.perimeter_data, perimeter, topology)
    # finish communication
    ClimaComms.finish(dss_buffer.graph_context)
    # load from receive buffer
    load_from_recv_buffer!(device, dss_buffer)
    # finish dss computation for ghost dof
    dss_ghost!(device, dss_buffer.perimeter_data, perimeter, topology)
    # load perimeter_data into data
    dss_unload_perimeter_data!(device, data, dss_buffer, perimeter)
    return nothing
end

dss_1d!(
    ::ClimaComms.AbstractCPUDevice,
    data::DataLayouts.VIJHWithF,
    topology::IntervalTopology,
    local_geometry = nothing,
    dss_weights = nothing,
) =
    @inbounds for h in axes(data, 4), v in axes(data, 1)
        h == size(data, 4) && (isperiodic(topology) || continue)
        I1 = CartesianIndex(v, size(data, 2), 1, h)
        I2 = CartesianIndex(v, 1, 1, h == size(data, 4) ? 1 : h + 1)
        sum_data =
            dss_transform(data, local_geometry, dss_weights, I1) +
            dss_transform(data, local_geometry, dss_weights, I2)
        data[I1] = dss_untransform(eltype(data), sum_data, local_geometry, I1)
        data[I2] = dss_untransform(eltype(data), sum_data, local_geometry, I2)
    end
