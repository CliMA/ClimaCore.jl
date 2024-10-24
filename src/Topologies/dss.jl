using DocStringExtensions
using .DataLayouts: CartesianFieldIndex

const DSSTypesAll = Union{
    DataLayouts.IFH,
    DataLayouts.IHF,
    DataLayouts.VIFH,
    DataLayouts.VIHF,
    DataLayouts.IJFH,
    DataLayouts.IJHF,
    DataLayouts.VIJFH,
    DataLayouts.VIJHF,
}
const DSSDataTypes = Union{
    DataLayouts.IJFH,
    DataLayouts.IJHF,
    DataLayouts.VIJFH,
    DataLayouts.VIJHF,
}
const DSSPerimeterTypes = Union{DataLayouts.VIFH, DataLayouts.VIHF}
const DSSWeightTypes = Union{DataLayouts.IJFH, DataLayouts.IJHF}

"""
    DSSBuffer{G, D, A, B}

# Fields
$(DocStringExtensions.FIELDS)
"""
struct DSSBuffer{S, G, D, A, B, VI}
    "ClimaComms graph context for communication"
    graph_context::G
    """
    Perimeter `DataLayout` object: typically a `VIFH{TT,Nv,Np,Nh}` or `VIHF{TT,Nv,Np,Nh}`, where `TT` is the
    transformed type, `Nv` is the number of vertical levels, and `Np` is the length of the perimeter
    """
    perimeter_data::D
    "send buffer `AbstractVector{FT}`"
    send_data::A
    "recv buffer `AbstractVector{FT}`"
    recv_data::A
    "indexing array for loading send buffer from `perimeter_data`"
    send_buf_idx::B
    "indexing array for loading (and summing) data from recv buffer to `perimeter_data`"
    recv_buf_idx::B
    "internal local elements (lidx)"
    internal_elems::VI
    "local elements (lidx) located on process boundary"
    perimeter_elems::VI
end

"""
    create_dss_buffer(
        data::Union{DataLayouts.IJFH{S}, DataLayouts.IJHF{S}, DataLayouts.VIJFH{S}, DataLayouts.VIJHF{S}},
        topology::Topology2D,
        local_geometry::Union{DataLayouts.IJFH, DataLayouts.IJHF, DataLayouts.VIJFH, DataLayouts.VIJHF, Nothing} = nothing,
        local_weights::Union{DataLayouts.IJFH, DataLayouts.IJHF, DataLayouts.VIJFH, DataLayouts.VIJHF, Nothing} = nothing,
    ) where {S}

Creates a [`DSSBuffer`](@ref) for the field data corresponding to `data`
"""
create_dss_buffer(
    data::DSSDataTypes,
    topology::Topology2D,
    local_geometry::Union{DSSDataTypes, Nothing} = nothing,
    local_weights::Union{DSSDataTypes, Nothing} = nothing,
) = create_dss_buffer(
    data,
    topology,
    DataLayouts.VIFH,
    local_geometry,
    local_weights,
)

function create_dss_buffer(
    data::DSSDataTypes,
    topology::Topology2D,
    ::Type{PerimeterLayout},
    local_geometry::Union{DSSDataTypes, Nothing} = nothing,
    local_weights::Union{DSSDataTypes, Nothing} = nothing,
) where {PerimeterLayout}
    S = eltype(data)
    Nij = DataLayouts.get_Nij(data)
    Nij_lg =
        isnothing(local_geometry) ? Nij : DataLayouts.get_Nij(local_geometry)
    Nij_weights =
        isnothing(local_weights) ? Nij : DataLayouts.get_Nij(local_weights)
    @assert Nij == Nij_lg == Nij_weights
    perimeter::Perimeter2D = Perimeter2D(Nij)
    context = ClimaComms.context(topology)
    DA = ClimaComms.array_type(topology)
    convert_to_array = DA isa Array ? false : true
    (_, _, _, Nv, Nh) = Base.size(data)
    Np = length(perimeter)
    Nf = DataLayouts.ncomponents(data)
    nfacedof = Nij - 2
    T = eltype(parent(data))
    # Add TS for Covariant123Vector
    # For DSS of Covariant123Vector, the third component is treated like a scalar
    # and is not transformed
    TS = if eltype(data) <: Geometry.Covariant123Vector
        Geometry.UVWVector{T}
    elseif eltype(data) <: Geometry.Contravariant123Vector
        Geometry.UVWVector{T}
    else
        _transformed_type(data, local_geometry, local_weights, DA) # extract transformed type
    end

    perimeter_data = if !isnothing(local_geometry)
        fdim = DataLayouts.field_dim(DataLayouts.singleton(local_geometry))
        if fdim == ndims(local_geometry)
            DataLayouts.VIHF{TS, Nv, Np}(DA{T}(undef, Nv, Np, Nh, Nf))
        else
            DataLayouts.VIFH{TS, Nv, Np}(DA{T}(undef, Nv, Np, Nf, Nh))
        end
    else
        PerimeterLayout{TS, Nv, Np}(DA{T}(undef, Nv, Np, Nf, Nh))
    end

    if context isa ClimaComms.SingletonCommsContext
        graph_context = ClimaComms.SingletonGraphContext(context)
        send_data, recv_data = T[], T[]
        send_buf_idx, recv_buf_idx = Int[], Int[]
        send_data, recv_data = DA{T}(undef, 0), DA{T}(undef, 0)
        send_buf_idx, recv_buf_idx = DA{Int}(undef, 0), DA{Int}(undef, 0)
        internal_elems = DA{Int}(1:nelems(topology))
        perimeter_elems = DA{Int}(undef, 0)
    else
        (; comm_vertex_lengths, comm_face_lengths) = topology
        vertex_buffer_lengths = comm_vertex_lengths .* (Nv * Nf)
        face_buffer_lengths = comm_face_lengths .* (Nv * Nf * nfacedof)
        buffer_lengths = vertex_buffer_lengths .+ face_buffer_lengths
        buffer_size = sum(buffer_lengths)
        send_data = DA{T}(undef, buffer_size)
        recv_data = DA{T}(undef, buffer_size)
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
    return DSSBuffer{S, G, D, A, B, VI}(
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

Base.eltype(::DSSBuffer{S}) where {S} = S

assert_same_eltype(::DataLayouts.AbstractData, ::DSSBuffer) =
    error("Incorrect buffer eltype")
assert_same_eltype(::DataLayouts.AbstractData{S}, ::DSSBuffer{S}) where {S} =
    nothing
assert_same_eltype(::DataLayouts.AbstractData, ::Nothing) = nothing

"""
    dss_transform!(
        device::ClimaComms.AbstractDevice,
        dss_buffer::DSSBuffer,
        data::Union{DataLayouts.IJFH, DataLayouts.IJHF, DataLayouts.VIJFH, DataLayouts.VIJHF},
        local_geometry::Union{DataLayouts.IJFH, DataLayouts.IJHF, DataLayouts.VIJFH, DataLayouts.VIJHF},
        weight::Union{DataLayouts.IJFH, DataLayouts.IJHF},
        perimeter::Perimeter2D,
        localelems::AbstractVector{Int},
    )

Transforms vectors from Covariant axes to physical (local axis), weights the data at perimeter nodes, 
and stores result in the `perimeter_data` array. This function calls the appropriate version of 
`dss_transform!` based on the data layout of the input arguments.

Arguments:

- `dss_buffer`: [`DSSBuffer`](@ref) generated by `create_dss_buffer` function for field data
- `data`: field data
- `local_geometry`: local metric information defined at each node
- `weight`: local dss weights for horizontal space
- `perimeter`: perimeter iterator
- `localelems`: list of local elements to perform transformation operations on

Part of [`ClimaCore.Spaces.weighted_dss!`](@ref).
"""
function dss_transform!(
    device::ClimaComms.AbstractDevice,
    dss_buffer::DSSBuffer,
    data::DSSDataTypes,
    local_geometry::DSSDataTypes,
    weight::DSSWeightTypes,
    perimeter::Perimeter2D,
    localelems::AbstractVector{Int},
)
    if !isempty(localelems)
        dss_transform!(
            device,
            dss_buffer.perimeter_data,
            data,
            perimeter,
            local_geometry,
            weight,
            localelems,
        )
    end
    return nothing
end

# `dss_transform` of a `Covariant12Vector` returns a
# `UVWVector`, however, we only need to store a `UVVector`
# in `perimeter_data`. Therefore, we drop the vertical dimension:
# via `drop_vert_dim`
"""
    drop_vert_dim(::Type{T}, X)

Convert the type of `X` to type `T` recursively
using `_drop_vert_dim`, which converts from `UVWVector`
to `UVVector` if `T <: UVVector`.
"""
@inline drop_vert_dim(::Type{T}, X) where {T} =
    RecursiveApply.rmap(RecursiveApply.rzero(T), X) do zero_value, x
        _drop_vert_dim(typeof(zero_value), x)
    end
@inline _drop_vert_dim(
    ::Type{T},
    x::Geometry.UVWVector,
) where {T <: Geometry.UVVector} = Geometry.UVVector(x.u, x.v)
@inline _drop_vert_dim(::Type{T}, x::T) where {T} = x

"""
    function dss_transform!(
        ::ClimaComms.AbstractCPUDevice,
        perimeter_data::Union{DataLayouts.VIFH, DataLayouts.VIHF},
        data::Union{DataLayouts.IJFH, DataLayouts.IJHF, DataLayouts.VIJFH, DataLayouts.VIJHF},
        perimeter::AbstractPerimeter,
        bc,
        localelems::Vector{Int},
    )

Transforms vectors from Covariant axes to physical (local axis), weights
the data at perimeter nodes, and stores result in the `perimeter_data` array.

Arguments:

- `perimeter_data`: contains the perimeter field data, represented on the physical axis, corresponding to the full field data in `data`
- `data`: field data
- `perimeter`: perimeter iterator
- `bc`: A broadcasted object representing the dss transform.
- `localelems`: list of local elements to perform transformation operations on

Part of [`ClimaCore.Spaces.weighted_dss!`](@ref).
"""
function dss_transform!(
    ::ClimaComms.AbstractCPUDevice,
    perimeter_data::DSSPerimeterTypes,
    data::DSSDataTypes,
    perimeter::Perimeter2D{Nq},
    local_geometry,
    weight::DSSWeightTypes,
    localelems::Vector{Int},
) where {Nq}
    (_, _, _, nlevels, _) = DataLayouts.universal_size(perimeter_data)
    CI = CartesianIndex
    (; ∂ξ∂x) = local_geometry
    @inbounds for elem in localelems
        for (p, (ip, jp)) in enumerate(perimeter)
            for level in 1:nlevels
                loc = CI(ip, jp, 1, level, elem)
                src = dss_transform(
                    data[loc],
                    local_geometry[CI(ip, jp, 1, level, elem)],
                    weight[loc],
                )
                perimeter_data[CI(p, 1, 1, level, elem)] =
                    drop_vert_dim(eltype(perimeter_data), src)

            end
        end
    end
    return nothing
end

"""
    dss_untransform!(
        device::ClimaComms.AbstractDevice,
        dss_buffer::DSSBuffer,
        data::Union{DataLayouts.IJFH, DataLayouts.IJHF, DataLayouts.VIJFH, DataLayouts.VIJHF},
        local_geometry::Union{DataLayouts.IJFH, DataLayouts.IJHF, DataLayouts.VIJFH, DataLayouts.VIJHF},
        perimeter::AbstractPerimeter,
    )

Transforms the DSS'd local vectors back to Covariant12 vectors, and copies the DSS'd data from the
`perimeter_data` to `data`. This function calls the appropriate version of `dss_transform!` function
based on the data layout of the input arguments.

Arguments:

- `dss_buffer`: [`DSSBuffer`](@ref) generated by `create_dss_buffer` function for field data
- `data`: field data
- `local_geometry`: local metric information defined at each node
- `perimeter`: perimeter iterator
- `localelems`: list of local elements to perform transformation operations on

Part of [`ClimaCore.Spaces.weighted_dss!`](@ref).
"""
function dss_untransform!(
    device::ClimaComms.AbstractDevice,
    dss_buffer::DSSBuffer,
    data::DSSDataTypes,
    local_geometry::DSSDataTypes,
    perimeter::Perimeter2D,
    localelems::AbstractVector{Int},
)

    (; perimeter_data) = dss_buffer
    dss_untransform!(
        device,
        perimeter_data,
        data,
        local_geometry,
        perimeter,
        localelems,
    )
    return nothing
end

"""
    function dss_untransform!(
        ::ClimaComms.AbstractCPUDevice,
        perimeter_data::Union{DataLayouts.VIFH, DataLayouts.VIHF},
        data::Union{DataLayouts.IJFH, DataLayouts.IJHF, DataLayouts.VIJFH, DataLayouts.VIJHF},
        local_geometry,
        localelems::Vector{Int},
    )

Transforms the DSS'd local vectors back to Covariant12 vectors, and copies the DSS'd data from the
`perimeter_data` to `data`.

Arguments:

- `perimeter_data`: contains the perimeter field data, represented on the physical axis, corresponding to the full field data in `data`
- `data`: field data
- `local_geometry`: Field data containing local geometry

Part of [`ClimaCore.Spaces.weighted_dss!`](@ref).
"""

function dss_untransform!(
    ::ClimaComms.AbstractCPUDevice,
    perimeter_data::DSSPerimeterTypes,
    data::DSSDataTypes,
    local_geometry::DSSDataTypes,
    perimeter::Perimeter2D,
    localelems::Vector{Int},
)
    (_, _, _, nlevels, _) = DataLayouts.universal_size(perimeter_data)
    CI = CartesianIndex
    @inbounds for elem in localelems
        for (p, (ip, jp)) in enumerate(perimeter)
            for level in 1:nlevels
                data[CI(ip, jp, 1, level, elem)] = dss_untransform(
                    eltype(data),
                    perimeter_data[CI(p, 1, 1, level, elem)],
                    local_geometry[CI(ip, jp, 1, level, elem)],
                )
            end
        end
    end
    return nothing
end

function dss_load_perimeter_data!(
    ::ClimaComms.AbstractCPUDevice,
    dss_buffer::DSSBuffer,
    data::DSSDataTypes,
    perimeter::Perimeter2D,
)
    (; perimeter_data) = dss_buffer
    (_, _, _, nlevels, nelems) = DataLayouts.universal_size(perimeter_data)
    CI = CartesianIndex
    @inbounds for elem in 1:nelems, (p, (ip, jp)) in enumerate(perimeter)
        for level in 1:nlevels
            perimeter_data[CI(p, 1, 1, level, elem)] =
                data[CI(ip, jp, 1, level, elem)]
        end
    end
    return nothing
end

function dss_unload_perimeter_data!(
    ::ClimaComms.AbstractCPUDevice,
    data::DSSDataTypes,
    dss_buffer::DSSBuffer,
    perimeter::Perimeter2D,
)
    (; perimeter_data) = dss_buffer
    (_, _, _, nlevels, nelems) = DataLayouts.universal_size(perimeter_data)
    CI = CartesianIndex
    @inbounds for elem in 1:nelems, (p, (ip, jp)) in enumerate(perimeter)
        for level in 1:nlevels
            data[CI(ip, jp, 1, level, elem)] =
                perimeter_data[CI(p, 1, 1, level, elem)]
        end
    end
    return nothing
end

"""
    function dss_local!(
        ::ClimaComms.AbstractCPUDevice,
        perimeter_data::DataLayouts.VIFH,
        perimeter::AbstractPerimeter,
        topology::AbstractTopology,
    )

Performs DSS on local vertices and faces.

Part of [`ClimaCore.Spaces.weighted_dss!`](@ref).
"""
function dss_local!(
    ::ClimaComms.AbstractCPUDevice,
    perimeter_data::DSSPerimeterTypes,
    perimeter::Perimeter2D,
    topology::Topology2D,
)
    dss_local_vertices!(perimeter_data, perimeter, topology)
    dss_local_faces!(perimeter_data, perimeter, topology)
    return nothing
end

"""
    dss_local_vertices!(
        perimeter_data::DataLayouts.VIFH,
        perimeter::Perimeter2D,
        topology::Topology2D,
    )

Apply dss to local vertices.
"""
function dss_local_vertices!(
    perimeter_data::DSSPerimeterTypes,
    perimeter::Perimeter2D,
    topology::Topology2D,
)
    Nv = size(perimeter_data, 4)
    @inbounds for vertex in local_vertices(topology)
        # for each level
        for level in 1:Nv
            # gather: compute sum over shared vertices
            sum_data = mapreduce(
                ⊞,
                vertex;
                init = RecursiveApply.rzero(eltype(slab(perimeter_data, 1, 1))),
            ) do (lidx, vert)
                ip = perimeter_vertex_node_index(vert)
                perimeter_slab = slab(perimeter_data, level, lidx)
                perimeter_slab[slab_index(ip)]
            end
            # scatter: assign sum to shared vertices
            for (lidx, vert) in vertex
                perimeter_slab = slab(perimeter_data, level, lidx)
                ip = perimeter_vertex_node_index(vert)
                perimeter_slab[slab_index(ip)] = sum_data
            end
        end
    end
    return nothing
end

function dss_local_faces!(
    perimeter_data::DSSPerimeterTypes,
    perimeter::Perimeter2D,
    topology::Topology2D,
)
    (Np, _, _, Nv, _) = size(perimeter_data)
    nfacedof = div(Np - 4, 4)

    @inbounds for (lidx1, face1, lidx2, face2, reversed) in
                  interior_faces(topology)
        pr1 = perimeter_face_indices(face1, nfacedof, false)
        pr2 = perimeter_face_indices(face2, nfacedof, reversed)
        for level in 1:Nv
            perimeter_slab1 = slab(perimeter_data, level, lidx1)
            perimeter_slab2 = slab(perimeter_data, level, lidx2)
            for (ip1, ip2) in zip(pr1, pr2)
                val =
                    perimeter_slab1[slab_index(ip1)] ⊞
                    perimeter_slab2[slab_index(ip2)]
                perimeter_slab1[slab_index(ip1)] = val
                perimeter_slab2[slab_index(ip2)] = val
            end
        end
    end
    return nothing
end
"""
    function dss_local_ghost!(
        ::ClimaComms.AbstractCPUDevice,
        perimeter_data::DataLayouts.VIFH,
        perimeter::AbstractPerimeter,
        topology::AbstractTopology,
    )

Computes the "local" part of ghost vertex dss. (i.e. it computes the summation of all the shared local
vertices of a unique ghost vertex and stores the value in each of the local vertex locations in 
`perimeter_data`)

Part of [`ClimaCore.Spaces.weighted_dss!`](@ref).
"""
function dss_local_ghost!(
    ::ClimaComms.AbstractCPUDevice,
    perimeter_data::DSSPerimeterTypes,
    perimeter::AbstractPerimeter,
    topology::AbstractTopology,
)
    nghostvertices = length(topology.ghost_vertex_offset) - 1
    if nghostvertices > 0
        (Np, _, _, Nv, _) = size(perimeter_data)
        @inbounds for vertex in ghost_vertices(topology)
            for level in 1:Nv
                # gather: compute sum over shared vertices
                sum_data = mapreduce(
                    ⊞,
                    vertex;
                    init = RecursiveApply.rzero(
                        eltype(slab(perimeter_data, 1, 1)),
                    ),
                ) do (isghost, idx, vert)
                    ip = perimeter_vertex_node_index(vert)
                    if !isghost
                        lidx = idx
                        perimeter_slab = slab(perimeter_data, level, lidx)
                        perimeter_slab[slab_index(ip)]
                    else
                        RecursiveApply.rmap(
                            zero,
                            slab(perimeter_data, 1, 1)[slab_index(1)],
                        )
                    end
                end
                for (isghost, idx, vert) in vertex
                    if !isghost
                        ip = perimeter_vertex_node_index(vert)
                        lidx = idx
                        perimeter_slab = slab(perimeter_data, level, lidx)
                        perimeter_slab[slab_index(ip)] = sum_data
                    end
                end
            end
        end
    end
    return nothing
end
"""
    dss_ghost!(
        device::ClimaComms.AbstractCPUDevice,
        perimeter_data::DataLayouts.VIFH,
        perimeter::AbstractPerimeter,
        topology::AbstractTopology,
    )

Sets the value for all local vertices of each unique ghost vertex, in `perimeter_data`, to that of 
the representative ghost vertex.

Part of [`ClimaCore.Spaces.weighted_dss!`](@ref).
"""
function dss_ghost!(
    device::ClimaComms.AbstractCPUDevice,
    perimeter_data::DSSPerimeterTypes,
    perimeter::AbstractPerimeter,
    topology::AbstractTopology,
)
    nghostvertices = length(topology.ghost_vertex_offset) - 1
    if nghostvertices > 0
        nlevels = size(perimeter_data, 4)
        (; repr_ghost_vertex) = topology
        @inbounds for (i, vertex) in enumerate(ghost_vertices(topology))
            idxresult, lvertresult = repr_ghost_vertex[i]
            ipresult = perimeter_vertex_node_index(lvertresult)
            for level in 1:nlevels
                result_slab = slab(perimeter_data, level, idxresult)
                result = result_slab[slab_index(ipresult)]
                for (isghost, idx, vert) in vertex
                    if !isghost
                        ip = perimeter_vertex_node_index(vert)
                        lidx = idx
                        perimeter_slab = slab(perimeter_data, level, lidx)
                        perimeter_slab[slab_index(ip)] = result
                    end
                end
            end
        end
    end
    return nothing
end

"""
    fill_send_buffer!(::ClimaComms.AbstractCPUDevice, dss_buffer::DSSBuffer; synchronize=true)

Loads the send buffer from `perimeter_data`. For unique ghost vertices, only data from the
representative vertices which store result of "ghost local" DSS are loaded.

Part of [`ClimaCore.Spaces.weighted_dss!`](@ref).
"""
function fill_send_buffer!(
    ::ClimaComms.AbstractCPUDevice,
    dss_buffer::DSSBuffer;
    synchronize = true,
)
    (; perimeter_data, send_buf_idx, send_data) = dss_buffer
    (Np, _, _, Nv, nelems) = size(perimeter_data)
    Nf = DataLayouts.ncomponents(perimeter_data)
    nsend = size(send_buf_idx, 1)
    ctr = 1
    CI = CartesianFieldIndex
    @inbounds for i in 1:nsend
        lidx = send_buf_idx[i, 1]
        ip = send_buf_idx[i, 2]
        for f in 1:Nf, v in 1:Nv
            send_data[ctr] = perimeter_data[CI(ip, 1, f, v, lidx)]
            ctr += 1
        end
    end
    return nothing
end
"""
    load_from_recv_buffer!(::ClimaComms.AbstractCPUDevice, dss_buffer::DSSBuffer)

Adds data from the recv buffer to the corresponding location in `perimeter_data`.
For ghost vertices, this data is added only to the representative vertices. The values are 
then scattered to other local vertices corresponding to each unique ghost vertex in `dss_local_ghost`.

Part of [`ClimaCore.Spaces.weighted_dss!`](@ref).
"""
function load_from_recv_buffer!(
    ::ClimaComms.AbstractCPUDevice,
    dss_buffer::DSSBuffer,
)
    (; perimeter_data, recv_buf_idx, recv_data) = dss_buffer
    (Np, _, _, Nv, nelems) = size(perimeter_data)
    Nf = DataLayouts.ncomponents(perimeter_data)
    nrecv = size(recv_buf_idx, 1)
    ctr = 1
    CI = CartesianFieldIndex
    @inbounds for i in 1:nrecv
        lidx = recv_buf_idx[i, 1]
        ip = recv_buf_idx[i, 2]
        for f in 1:Nf, v in 1:Nv
            ci = CI(ip, 1, f, v, lidx)
            perimeter_data[ci] += recv_data[ctr]
            ctr += 1
        end
    end
    return nothing
end

"""
    dss!(data, topology)

Computed unweighted/pure DSS of `data`.
"""
function dss!(data::DSSDataTypes, topology::Topology2D)
    Nij = DataLayouts.get_Nij(data)
    length(parent(data)) == 0 && return nothing
    device = ClimaComms.device(topology)
    perimeter = Perimeter2D(Nij)
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

function dss_1d!(
    htopology::AbstractTopology,
    data,
    local_geometry_data = nothing,
    dss_weights = nothing,
)
    Nq = size(data, 1)
    Nv = size(data, 4)
    idx1 = CartesianIndex(1, 1, 1, 1, 1)
    idx2 = CartesianIndex(Nq, 1, 1, 1, 1)
    @inbounds for (elem1, face1, elem2, face2, reversed) in
                  interior_faces(htopology)
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
