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

# Whole-element halo exchange: `send_data`/`recv_data` hold entire neighbour
# elements (indexed by `sidx`/`ridx`), in contrast to the perimeter-only
# `DSSBuffer` and the face-strip `GhostFaceExchange`. Used by the
# quasimonotone limiter (whose distributed test does not exercise the
# exchange; see issue #1511).
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

"""
    fill_send_buffer!(topology, data, ghost_buffer::GhostBuffer)

Loads the send buffer of `ghost_buffer` with the data of the elements
that neighboring processes need for their ghost elements.
"""
function fill_send_buffer!(
    topology::Topology2D,
    data::DataLayouts.DataLayout,
    ghost_buffer::GhostBuffer,
)
    # NOTE: this copies one element per iteration, which is a separate kernel
    # launch per send element when the arrays live on a GPU. That is
    # inconsequential at the element counts this is currently used with (the
    # limiter's ghost exchange), but a single gather over `send_elem_lidx`
    # would be preferable if it is ever used with many send elements.
    # The parent array stores H at dim 4 or 5, depending on where F is
    h_dim = DataLayouts.f_dim(data) == 5 ? 4 : 5
    send_array = parent(ghost_buffer.send_data)
    data_array = parent(data)
    for (sidx, lidx) in enumerate(topology.send_elem_lidx)
        selectdim(send_array, h_dim, sidx) .= selectdim(data_array, h_dim, lidx)
    end
    return nothing
end

"""
    GhostFaceExchange

Face-strip halo exchange for the DG ghost-face operators: ships only the `Nq`
face-node values (per level and field component) of each rank-boundary face,
instead of whole neighbour elements — the face analog of the perimeter-only
`DSSBuffer`. Only face-sharing neighbours participate; vertex-only halo
neighbours are excluded on both sides of each pair, so participation is
symmetric by construction.

`send_data`/`recv_data` are `(Nv, Nq, 1, nstrips)` layouts with one strip per
entry of [`ghost_faces`](@ref): slot `s` of `send_data` holds this rank's own
("minus") side of a shared face, and slot `s` of `recv_data` receives the
neighbouring rank's side of the same face. Both ranks assign slots by sorting
their shared faces by neighbour pid and then by the canonical face key (the
ordered pair of `(global element, face)` of the two sides), which both ranks
can compute, so the `k`-th strip sent within a pair is the `k`-th strip the
peer expects. Strips are packed in the sending face's natural node order
(`face_node_index(face, Nq, q, false)`); a receiver whose face is `reversed`
relative to the sender reads node `q` at strip index `Nq - q + 1`.

Fields:

  - `graph_context`: the `ClimaComms` exchange over face-sharing neighbours;
  - `send_data`, `recv_data`: the strip layouts described above;
  - `slot_lidx`, `slot_face`: strip slot → local element and face, stored with
    the topology's array type for the device-side pack;
  - `face_slot`: position in [`ghost_faces`](@ref) order → strip slot (host
    vector; consumed when building face connectivity on the host).
"""
struct GhostFaceExchange{G, D, IV}
    graph_context::G
    send_data::D
    recv_data::D
    slot_lidx::IV
    slot_face::IV
    face_slot::Vector{Int32}
    # Exchanges are memoized per (space, data type, argument position), so
    # distinct fields of the same type share this object; the latch is set at
    # fill and cleared at finish, turning an overlapping second start — which
    # would overwrite the in-flight send strips and double-start the graph
    # context — into an error.
    in_flight::Base.RefValue{Bool}
end

# Strip schedule shared by every `GhostFaceExchange` on a topology: for each
# entry of `ghost_faces(topology)` (in order), its strip slot, and for each
# slot the local (element, face) to pack, plus the per-neighbour strip counts.
# The counts are symmetric within each pair (the peer's `ghost_faces` mirrors
# ours face by face), so they serve as both send and receive lengths.
function ghost_face_schedule(topology::Topology2D)
    gfaces = ghost_faces(topology)
    n = length(gfaces)
    sort_keys = Vector{NTuple{5, Int}}(undef, n)
    for (f, (lidx, face, ridx, oface, _reversed)) in enumerate(gfaces)
        gidx = topology.local_elem_gidx[lidx]
        ogidx = topology.recv_elem_gidx[ridx]
        pid = topology.elempid[ogidx]
        sort_keys[f] =
            (gidx, face) < (ogidx, oface) ? (pid, gidx, face, ogidx, oface) :
            (pid, ogidx, oface, gidx, face)
    end
    perm = sortperm(sort_keys)
    face_slot = Vector{Int32}(undef, n)
    slot_lidx = Vector{Int32}(undef, n)
    slot_face = Vector{Int32}(undef, n)
    pids = Int[]
    counts = Int[]
    for (slot, f) in enumerate(perm)
        face_slot[f] = slot
        slot_lidx[slot] = gfaces[f][1]
        slot_face[slot] = gfaces[f][2]
        pid = sort_keys[f][1]
        if isempty(pids) || pids[end] != pid
            push!(pids, pid)
            push!(counts, 0)
        end
        counts[end] += 1
    end
    return (; face_slot, slot_lidx, slot_face, pids, counts)
end

"""
    create_ghost_face_exchange(data, topology)

Construct the [`GhostFaceExchange`](@ref) for exchanging the ghost-face strips
of `data` (see there for the slot and node-order conventions). Returns
`nothing` on single-process contexts and on ranks with no ghost faces — with
face-strip granularity a rank with no ghost faces shares no exchange with any
neighbour, so it can skip the exchange without stranding a peer.
"""
create_ghost_face_exchange(data, topology::AbstractTopology) = nothing

function create_ghost_face_exchange(
    data::DataLayouts.VIJHWithF,
    topology::Topology2D,
)
    ClimaComms.context(topology) isa ClimaComms.SingletonCommsContext &&
        return nothing
    isempty(ghost_faces(topology)) && return nothing
    (; face_slot, slot_lidx, slot_face, pids, counts) =
        ghost_face_schedule(topology)
    nstrips = length(slot_lidx)
    S = eltype(data)
    FT = eltype(parent(data))
    (Nv, Nq, _, _) = size(data)
    DA = ClimaComms.array_type(topology)
    send_data = DataLayouts.rebuild(
        DataLayouts.VIJFH{S, Nv, Nq, 1, nothing}(Array{FT}, nstrips),
        DA,
    )
    recv_data = DataLayouts.rebuild(
        DataLayouts.VIJFH{S, Nv, Nq, 1, nothing}(Array{FT}, nstrips),
        DA,
    )
    k = stride(parent(send_data), DataLayouts.f_dim(send_data) == 5 ? 4 : 5)
    graph_context = ClimaComms.graph_context(
        topology.context,
        parent(send_data),
        k .* counts,
        pids,
        parent(recv_data),
        k .* counts,
        pids,
    )
    return GhostFaceExchange(
        graph_context,
        send_data,
        recv_data,
        DA(slot_lidx),
        DA(slot_face),
        face_slot,
        Ref(false),
    )
end

"""
    fill_face_send_buffer!(data, exchange::GhostFaceExchange)

Load the send strips of `exchange` with the face-node values of `data`, in
each face's natural node order. Host loop — the CUDA path packs with a single
kernel instead (see `ext/cuda/operators_dg.jl`).
"""
function fill_face_send_buffer!(
    data::DataLayouts.DataLayout,
    exchange::GhostFaceExchange,
)
    (Nv, Nq, _, nstrips) = size(exchange.send_data)
    for s in 1:nstrips
        lidx = Int(exchange.slot_lidx[s])
        face = Int(exchange.slot_face[s])
        for v in 1:Nv
            data_slab = slab(data, v, lidx)
            send_slab = slab(exchange.send_data, v, s)
            for q in 1:Nq
                i, j = face_node_index(face, Nq, q, false)
                send_slab[1, q, 1, 1] = data_slab[1, i, j, 1]
            end
        end
    end
    return nothing
end
