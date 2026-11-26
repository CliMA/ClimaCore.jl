"""
    DSSBuffer{T, G, D, A, B, O, VI}

Buffers and index maps for the direct stiffness summation (DSS) of one data layout
with element type `T` on a [`Topology2D`](@ref). Construct with
`create_dss_buffer(data, topology, local_geometry = nothing, dss_weights = nothing)`,
or several at once, sharing one exchange, with [`create_dss_buffers`](@ref).

# Fields

  - `graph_context`: `ClimaComms` graph context for communication.
  - `perimeter_data`: `DataLayout` holding the (transformed) values at the perimeter
    nodes of each element.
  - `send_data`: Send buffer, an `AbstractVector{FT}`.
  - `recv_data`: Receive buffer, an `AbstractVector{FT}`.
  - `send_buf_idx`: `(h, p)` perimeter location of each send-buffer row.
  - `recv_buf_idx`: `(h, p)` perimeter location of each receive-buffer row.
  - `buf_offset`: Offset (0-based) of each row in `send_data` and in `recv_data`.
    A row holds the values at all levels of one perimeter node; the send and receive
    rows of each neighbor have the same layout.
  - `recv_target_idx`: `(h, p)` of each distinct perimeter node that receives data.
  - `recv_target_offset`: Offsets into `recv_target_rows`, one per receiving node
    plus one, in compressed sparse row (CSR) form.
  - `recv_target_rows`: Receive-buffer rows grouped by receiving node, in ascending
    order within each group, so that the device reduction adds them in the order of
    the CPU loop.
  - `internal_elems`: Local elements (`lidx`) that touch no ghost element.
  - `perimeter_elems`: Local elements (`lidx`) on the process boundary.
"""
struct DSSBuffer{T, G, D, A, B, O, VI}
    graph_context::G
    perimeter_data::D
    send_data::A
    recv_data::A
    send_buf_idx::B
    recv_buf_idx::B
    buf_offset::O
    recv_target_idx::B
    recv_target_offset::O
    recv_target_rows::O
    internal_elems::VI
    perimeter_elems::VI
end

"""
    MultiDSSBuffer{G, BS}

DSS buffers for several fields that share one send buffer, one receive buffer, and
one `ClimaComms` graph context, so that their DSS sends one message per neighboring
process. Construct with `Spaces.create_dss_buffer(field1, field2, ...)` and use
with `Spaces.weighted_dss!((field1, field2, ...) => buffer)`.

# Fields

  - `graph_context`: The shared graph context, or `nothing` if no field needs one.
  - `buffers`: One [`DSSBuffer`](@ref) per field, or `nothing` for a field that
    needs no buffer (a discontinuous or 1D space).
"""
struct MultiDSSBuffer{G, BS <: Tuple}
    graph_context::G
    buffers::BS
end

create_dss_buffer(
    data::DataLayouts.VIJHWithF,
    topology::Topology2D,
    local_geometry = nothing,
    dss_weights = nothing,
) = only(create_dss_buffers(((data, topology, local_geometry, dss_weights),)))

"""
    create_dss_buffers(specs)

Create one [`DSSBuffer`](@ref) for each tuple `(data, topology, local_geometry, dss_weights)` in `specs`. The buffers share one send buffer, one receive buffer,
and one graph context: the segment exchanged with each neighbor holds the rows of
all buffers for that neighbor, in the order of `specs`. With one spec, the
layout is that of a single [`create_dss_buffer`](@ref).

The topologies must share their communications context and neighbors, and the
data must share their floating-point type.
"""
function create_dss_buffers(specs::Tuple)
    topology = specs[1][2]
    context = ClimaComms.context(topology)
    DA = ClimaComms.array_type(topology)
    layouts = map(spec -> dss_perimeter_layout(spec...), specs)
    # Types are not captured by the closures below or stored in `layouts`:
    # inference only knows such a type as a `DataType` or `UnionAll`.
    FT = eltype(parent(specs[1][1]))
    for (data, other, _, _) in specs
        eltype(parent(data)) == FT || error(
            "DSS buffers can only be combined for data of one floating-point type",
        )
        ClimaComms.context(other) === context &&
        other.neighbor_pids == topology.neighbor_pids ||
            error("DSS buffers can only be combined for topologies with the same \
                   communications context and neighbors")
    end
    if context isa ClimaComms.SingletonCommsContext
        # Names differ from the multi-process branch: a variable that a closure
        # captures and that is assigned twice is boxed, which defeats inference.
        singleton_context = ClimaComms.SingletonGraphContext(context)
        empty_data = FT[]
        # internal_elems and perimeter_elems are indexed by the DSS kernels, so
        # they must be device arrays (as in the multi-process branch below); the
        # host send/recv buffer indices are only used off-device and stay host.
        return map(layouts, specs) do layout, (data, topology, _, _)
            to_device = ClimaComms.array_type(topology)
            DSSBuffer{eltype(data)}(
                singleton_context,
                layout.perimeter_data,
                empty_data,
                empty_data,
                zeros(Int, 0, 2),
                zeros(Int, 0, 2),
                Int[],
                zeros(Int, 0, 2),
                [1],
                Int[],
                to_device(collect(Base.OneTo(nelems(topology)))),
                to_device(Int[]),
            )
        end
    end
    neighbor_pids = topology.neighbor_pids
    # Rows per neighbor and row width (values per row) of each buffer. Each
    # neighbor's segment holds the rows of the first buffer, then of the second...
    rows_per_neighbor = map(layouts, specs) do layout, (_, topology, _, _)
        topology.comm_vertex_lengths .+
        topology.comm_face_lengths .* (layout.Nij - 2)
    end
    row_widths = map(layout -> layout.Nv * layout.Nf, layouts)
    field_lengths = map((rows, width) -> rows .* width, rows_per_neighbor, row_widths)
    buffer_lengths = sum(field_lengths)
    segment_starts = cumsum(buffer_lengths) .- buffer_lengths
    send_data = DA{FT}(undef, sum(buffer_lengths))
    recv_data = DA{FT}(undef, sum(buffer_lengths))
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
    field_starts = ntuple(length(specs)) do f
        segment_starts .+ sum(field_lengths[1:(f - 1)]; init = zero(segment_starts))
    end
    return map(
        layouts,
        specs,
        rows_per_neighbor,
        row_widths,
        field_starts,
    ) do layout, (data, topology, _, _), rows, row_width, starts
        to_device = ClimaComms.array_type(topology)
        send_buf_idx, recv_buf_idx = compute_ghost_send_recv_idx(topology, layout.Nij)
        # Row r of neighbor k is at starts[k] + (r - first row of k) * row_width.
        buf_offset = Vector{Int}(undef, sum(rows))
        row = 0
        for (start, nrows) in zip(starts, rows), r in 1:nrows
            buf_offset[row += 1] = start + (r - 1) * row_width
        end
        targets, target_offset, target_rows = recv_targets(recv_buf_idx)
        DSSBuffer{eltype(data)}(
            graph_context,
            layout.perimeter_data,
            send_data,
            recv_data,
            to_device(send_buf_idx),
            to_device(recv_buf_idx),
            to_device(buf_offset),
            to_device(targets),
            to_device(target_offset),
            to_device(target_rows),
            to_device(topology.internal_elems),
            to_device(topology.perimeter_elems),
        )
    end
end

DSSBuffer{T}(
    graph_context::G,
    perimeter_data::D,
    send_data::A,
    recv_data::A,
    send_buf_idx::B,
    recv_buf_idx::B,
    buf_offset::O,
    recv_target_idx::B,
    recv_target_offset::O,
    recv_target_rows::O,
    internal_elems::VI,
    perimeter_elems::VI,
) where {T, G, D, A, B, O, VI} = DSSBuffer{T, G, D, A, B, O, VI}(
    graph_context,
    perimeter_data,
    send_data,
    recv_data,
    send_buf_idx,
    recv_buf_idx,
    buf_offset,
    recv_target_idx,
    recv_target_offset,
    recv_target_rows,
    internal_elems,
    perimeter_elems,
)

# Perimeter storage and sizes for the DSS of `data`; the perimeter data has the
# element type of `dss_transform` applied to `data`.
function dss_perimeter_layout(
    data::DataLayouts.VIJHWithF,
    topology::Topology2D,
    local_geometry = nothing,
    dss_weights = nothing,
)
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
    return (; Nv, Nij, Nf, perimeter_data)
end

"""
    recv_targets(recv_buf_idx)

Group the rows of `recv_buf_idx` by the perimeter node `(h, p)` that they add to.
Returns the distinct nodes as a matrix with columns `h` and `p`, the CSR offsets of
each node's group, and the rows of all groups, in ascending order within each group.
"""
function recv_targets(recv_buf_idx::AbstractMatrix{Int})
    nrecv = size(recv_buf_idx, 1)
    keys = [(recv_buf_idx[r, 1], recv_buf_idx[r, 2], r) for r in 1:nrecv]
    rows = last.(sort!(keys))
    target(n) = (recv_buf_idx[rows[n], 1], recv_buf_idx[rows[n], 2])
    offsets = [n for n in 1:nrecv if n == 1 || target(n) != target(n - 1)]
    targets = Int[target(n)[i] for n in offsets, i in 1:2]
    push!(offsets, nrecv + 1)
    return targets, offsets, rows
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

"""
    dss_load_perimeter_data!(device, dss_buffer, data, perimeter)

Copy the values of `data` at the element-perimeter nodes given by `perimeter` into
`dss_buffer.perimeter_data`. This is the first phase of a DSS operation; the
untransformed counterpart of `dss_transform!`, used when `data` needs no local
geometry transformation.

Part of [`ClimaCore.Spaces.weighted_dss!`](@ref); see also [`DSSBuffer`](@ref).
"""
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

"""
    dss_unload_perimeter_data!(device, data, dss_buffer, perimeter)

Copy the summed values in `dss_buffer.perimeter_data` back into `data` at the
element-perimeter nodes given by `perimeter`. This is the final phase of a DSS
operation; the untransformed counterpart of `dss_untransform!`, used when `data`
needs no local geometry transformation.

Part of [`ClimaCore.Spaces.weighted_dss!`](@ref); see also [`DSSBuffer`](@ref).
"""
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

On a GPU, the copy is only enqueued: synchronize the device before passing the
send buffer to MPI. This lets a multi-field DSS fill all send buffers before one
synchronization.

Part of [`ClimaCore.Spaces.weighted_dss!`](@ref).
"""
function fill_send_buffer!(
    ::ClimaComms.AbstractCPUDevice,
    (; perimeter_data, send_data, send_buf_idx, buf_offset)::DSSBuffer,
)
    Nf = DataLayouts.ncomponents(perimeter_data)
    @inbounds for (row, (h, p)) in enumerate(eachrow(send_buf_idx)),
        v in axes(perimeter_data, 1)

        buffer_index = buf_offset[row] + (v - 1) * Nf + 1
        DataLayouts.set_struct!(send_data, perimeter_data[v, p, 1, h], buffer_index, Val(1))
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
    (; perimeter_data, recv_data, recv_buf_idx, buf_offset)::DSSBuffer,
)
    T = eltype(perimeter_data)
    Nf = DataLayouts.ncomponents(perimeter_data)
    @inbounds for (row, (h, p)) in enumerate(eachrow(recv_buf_idx)),
        v in axes(perimeter_data, 1)

        buffer_index = buf_offset[row] + (v - 1) * Nf + 1
        perimeter_data[v, p, 1, h] +=
            DataLayouts.get_struct(recv_data, T, buffer_index, Val(1))
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
    # MPI reads the send buffer on its own stream
    ClimaComms.device_synchronize(device)
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
