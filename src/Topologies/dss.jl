using DocStringExtensions

"""
    DSSBuffer{G, D, A, B}

# Fields
$(DocStringExtensions.FIELDS)
"""
struct DSSBuffer{S, G, D, A, B, VI}
    "ClimaComms graph context for communication"
    graph_context::G
    "Array for storing perimeter data"
    perimeter_data::D
    "send buffer"
    send_data::A
    "recv buffer"
    recv_data::A
    "indexing array for loading send buffer from `perimeter_data`"
    send_buf_idx::B
    "indexing array for loading (and summing) data from recv buffer to `perimeter_data`"
    recv_buf_idx::B
    "field id for all scalar fields stored in the `data` array"
    scalarfidx::VI
    "field id for all covariant12vector fields stored in the `data` array"
    covariant12fidx::VI
    "field id for all contravariant12vector fields stored in the `data` array"
    contravariant12fidx::VI
    "internal local elements (lidx)"
    internal_elems::VI
    "local elements (lidx) located on process boundary"
    perimeter_elems::VI
end

"""
    create_dss_buffer(
        data::Union{DataLayouts.IJFH{S, Nij}, DataLayouts.VIJFH{S, Nij}},
        hspace::AbstractSpectralElementSpace,
    ) where {S, Nij}

Creates a [`DSSBuffer`](@ref) for the field data corresponding to `data`
"""
function create_dss_buffer(
    data::Union{DataLayouts.IJFH{S, Nij}, DataLayouts.VIJFH{S, Nij}},
    hspace::AbstractSpectralElementSpace,
) where {S, Nij}
    @assert quadrature_style(hspace) isa Spaces.Quadratures.GLL "DSS2 is only compatible with GLL quadrature"
    local_geometry = local_geometry_data(hspace)
    local_weights = local_dss_weights(hspace)
    perimeter = Spaces.perimeter(hspace)
    create_dss_buffer(
        data,
        topology(hspace),
        perimeter,
        local_geometry,
        local_weights,
    )
end

function create_dss_buffer(
    data::Union{DataLayouts.IJFH{S, Nij}, DataLayouts.VIJFH{S, Nij}},
    topology,
    perimeter,
    local_geometry = nothing,
    local_weights = nothing,
) where {S, Nij}
    context = topology.context
    DA = ClimaComms.array_type(topology)
    convert_to_array = DA isa Array ? false : true
    (_, _, _, Nv, nelems) = Base.size(data)
    Np = Spaces.nperimeter(perimeter)
    Nf =
        length(parent(data)) == 0 ? 0 :
        cld(length(parent(data)), (Nij * Nij * Nv * nelems))
    nfacedof = Nij - 2
    T = eltype(parent(data))
    TS = _transformed_type(data, local_geometry, local_weights, DA) # extract transformed type
    # Add TS for Covariant123Vector
    # For DSS of Covariant123Vector, the third component is treated like a scalar
    # and is not transformed
    if eltype(data) <: Geometry.Covariant123Vector
        TS = Geometry.UVWVector{T}
    end
    perimeter_data = DataLayouts.VIFH{TS, Np}(DA{T}(undef, Nv, Np, Nf, nelems))
    if context isa ClimaComms.SingletonCommsContext
        graph_context = ClimaComms.SingletonGraphContext(context)
        send_data, recv_data = T[], T[]
        send_buf_idx, recv_buf_idx = Int[], Int[]
        send_data, recv_data = DA{T}(undef, 0), DA{T}(undef, 0)
        send_buf_idx, recv_buf_idx = DA{Int}(undef, 0), DA{Int}(undef, 0)
        internal_elems = DA{Int}(1:Topologies.nelems(topology))
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
        send_buf_idx, recv_buf_idx =
            Topologies.compute_ghost_send_recv_idx(topology, Nij)
        internal_elems = DA(topology.internal_elems)
        perimeter_elems = DA(topology.perimeter_elems)
    end
    scalarfidx, covariant12fidx, contravariant12fidx = Int[], Int[], Int[]
    supportedvectortypes = Union{
        Geometry.UVector,
        Geometry.VVector,
        Geometry.WVector,
        Geometry.UVVector,
        Geometry.UWVector,
        Geometry.VWVector,
        Geometry.UVWVector,
        Geometry.Covariant12Vector,
        Geometry.Covariant3Vector,
        Geometry.Covariant123Vector,
        Geometry.Contravariant12Vector,
        Geometry.Contravariant3Vector,
    }

    if S <: NamedTuple
        for (i, fieldtype) in enumerate(S.parameters[2].types)
            offset = DataLayouts.fieldtypeoffset(T, S, i)
            ncomponents = DataLayouts.typesize(T, fieldtype)
            if fieldtype <: Geometry.AxisVector # vector fields
                if !(fieldtype <: supportedvectortypes)
                    @show fieldtype
                    @show supportedvectortypes
                end
                @assert fieldtype <: supportedvectortypes
                if fieldtype <: Geometry.Covariant12Vector
                    push!(covariant12fidx, offset + 1)
                elseif fieldtype <: Geometry.Covariant123Vector
                    push!(covariant12fidx, offset + 1)
                    push!(scalarfidx, offset + 3)
                elseif fieldtype <: Geometry.Contravariant12Vector
                    push!(contravariant12fidx, offset + 1)
                else
                    append!(
                        scalarfidx,
                        Vector((offset + 1):(offset + ncomponents)),
                    )
                end
            elseif fieldtype <: NTuple # support a NTuple of primitive types
                append!(scalarfidx, Vector((offset + 1):(offset + ncomponents)))
            else # scalar fields
                push!(scalarfidx, offset + 1)
            end
        end
    else # deals with simple type, with single field (e.g: S = Float64, S = CovariantVector12, etc.)
        ncomponents = DataLayouts.typesize(T, S)
        if S <: Geometry.AxisVector # vector field
            if !(S <: supportedvectortypes)
                @show S
                @show supportedvectortypes
            end
            @assert S <: supportedvectortypes
            if S <: Geometry.Covariant12Vector
                push!(covariant12fidx, 1)
            elseif S <: Geometry.Covariant123Vector
                push!(covariant12fidx, 1)
                push!(scalarfidx, 3)
            elseif S <: Geometry.Contravariant12Vector
                push!(contravariant12fidx, 1)
            else
                append!(scalarfidx, Vector(1:ncomponents))
            end
        elseif S <: NTuple # support a NTuple of primitive types
            append!(scalarfidx, Vector(1:ncomponents))
        else # scalar field
            push!(scalarfidx, 1)
        end
    end
    scalarfidx = DA(scalarfidx)
    covariant12fidx = DA(covariant12fidx)
    contravariant12fidx = DA(contravariant12fidx)
    G = typeof(graph_context)
    D = typeof(perimeter_data)
    A = typeof(send_data)
    B = typeof(send_buf_idx)
    VI = typeof(scalarfidx)
    return DSSBuffer{S, G, D, A, B, VI}(
        graph_context,
        perimeter_data,
        send_data,
        recv_data,
        send_buf_idx,
        recv_buf_idx,
        scalarfidx,
        covariant12fidx,
        contravariant12fidx,
        internal_elems,
        perimeter_elems,
    )
end

Base.eltype(::DSSBuffer{S}) where {S} = S

create_dss_buffer(data::DataLayouts.AbstractData, hspace) = nothing

assert_same_eltype(::DataLayouts.AbstractData, ::DSSBuffer) =
    error("Incorrect buffer eltype")
assert_same_eltype(::DataLayouts.AbstractData{S}, ::DSSBuffer{S}) where {S} =
    nothing
assert_same_eltype(::DataLayouts.AbstractData, ::Nothing) = nothing

"""
    function weighted_dss!(
        data::Union{
            DataLayouts.IFH,
            DataLayouts.VIFH,
            DataLayouts.IJFH,
            DataLayouts.VIJFH,
        },
        space::Union{
            AbstractSpectralElementSpace,
            ExtrudedFiniteDifferenceSpace,
        },
        dss_buffer::Union{DSSBuffer, Nothing},
    )

Computes weighted dss of `data`. 

It comprises of the following steps:

1). [`Spaces.weighted_dss_start!`](@ref)

2). [`Spaces.weighted_dss_internal!`](@ref)

3). [`Spaces.weighted_dss_ghost!`](@ref)
"""
function weighted_dss!(
    data::Union{
        DataLayouts.IFH,
        DataLayouts.VIFH,
        DataLayouts.IJFH,
        DataLayouts.VIJFH,
    },
    space::Union{AbstractSpectralElementSpace, ExtrudedFiniteDifferenceSpace},
    dss_buffer::Union{DSSBuffer, Nothing},
)
    assert_same_eltype(data, dss_buffer)
    weighted_dss_start!(data, space, dss_buffer)
    weighted_dss_internal!(data, space, dss_buffer)
    weighted_dss_ghost!(data, space, dss_buffer)
end

"""
    weighted_dss_start!(
        data::Union{
            DataLayouts.IFH,
            DataLayouts.VIFH,
            DataLayouts.IJFH,
            DataLayouts.VIJFH,
        },
        space::Union{
            AbstractSpectralElementSpace,
            ExtrudedFiniteDifferenceSpace,
        },
        dss_buffer::Union{DSSBuffer, Nothing},
    )

It comprises of the following steps:

1). Apply [`Spaces.dss_transform!`](@ref) on perimeter elements. This weights and tranforms vector 
fields to physical basis if needed. Scalar fields are weighted. The transformed and/or weighted 
perimeter `data` is stored in `perimeter_data`.

2). Apply [`Spaces.dss_local_ghost!`](@ref)
This computes partial weighted DSS on ghost vertices, using only the information from `local` vertices.

3). [`Spaces.fill_send_buffer!`](@ref) 
Loads the send buffer from `perimeter_data`. For unique ghost vertices, only data from the
representative ghost vertices which store result of "ghost local" DSS are loaded.

4). Start DSS communication with neighboring processes
"""
weighted_dss_start!(
    data::Union{
        DataLayouts.IFH,
        DataLayouts.VIFH,
        DataLayouts.IJFH,
        DataLayouts.VIJFH,
    },
    space::Union{AbstractSpectralElementSpace, ExtrudedFiniteDifferenceSpace},
    dss_buffer::Union{DSSBuffer, Nothing},
) = weighted_dss_start!(data, space, horizontal_space(space), dss_buffer)

function weighted_dss_start!(
    data::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
    space::Union{
        Spaces.SpectralElementSpace2D,
        Spaces.ExtrudedFiniteDifferenceSpace,
    },
    hspace::SpectralElementSpace2D{<:Topology2D},
    dss_buffer::DSSBuffer,
)
    assert_same_eltype(data, dss_buffer)
    length(parent(data)) == 0 && return nothing
    device = ClimaComms.device(topology(hspace))
    dss_transform!(
        device,
        dss_buffer,
        data,
        local_geometry_data(space),
        local_dss_weights(hspace),
        Spaces.perimeter(hspace),
        dss_buffer.perimeter_elems,
    )
    dss_local_ghost!(
        device,
        dss_buffer.perimeter_data,
        Spaces.perimeter(hspace),
        topology(hspace),
    )
    fill_send_buffer!(device, dss_buffer)
    ClimaComms.start(dss_buffer.graph_context)
    return nothing
end

weighted_dss_start!(data, space, hspace, dss_buffer) = nothing
"""
    weighted_dss_internal!(
        data::Union{
            DataLayouts.IFH,
            DataLayouts.VIFH,
            DataLayouts.IJFH,
            DataLayouts.VIJFH,
        },
        space::Union{
            AbstractSpectralElementSpace,
            ExtrudedFiniteDifferenceSpace,
        },
        dss_buffer::DSSBuffer,
    )

1). Apply [`Spaces.dss_transform!`](@ref) on interior elements. Local elements are split into interior 
and perimeter elements to facilitate overlapping of communication with computation.

2). Probe communication

3). [`Spaces.dss_local!`](@ref) computes the weighted DSS on local vertices and faces.
"""
weighted_dss_internal!(
    data::Union{
        DataLayouts.IFH,
        DataLayouts.VIFH,
        DataLayouts.IJFH,
        DataLayouts.VIJFH,
    },
    space::Union{AbstractSpectralElementSpace, ExtrudedFiniteDifferenceSpace},
    dss_buffer::Union{DSSBuffer, Nothing},
) = weighted_dss_internal!(data, space, horizontal_space(space), dss_buffer)

function weighted_dss_internal!(
    data::Union{
        DataLayouts.IFH,
        DataLayouts.VIFH,
        DataLayouts.IJFH,
        DataLayouts.VIJFH,
    },
    space::Union{AbstractSpectralElementSpace, ExtrudedFiniteDifferenceSpace},
    hspace::AbstractSpectralElementSpace,
    dss_buffer::Union{DSSBuffer, Nothing},
)
    assert_same_eltype(data, dss_buffer)
    length(parent(data)) == 0 && return nothing
    if hspace isa SpectralElementSpace1D
        dss_1d!(
            topology(hspace),
            data,
            local_geometry_data(space),
            local_dss_weights(space),
        )
    else
        device = ClimaComms.device(topology(hspace))
        dss_transform!(
            device,
            dss_buffer,
            data,
            local_geometry_data(space),
            local_dss_weights(space),
            Spaces.perimeter(hspace),
            dss_buffer.internal_elems,
        )
        dss_local!(
            device,
            dss_buffer.perimeter_data,
            Spaces.perimeter(hspace),
            topology(hspace),
        )
        dss_untransform!(
            device,
            dss_buffer,
            data,
            local_geometry_data(space),
            Spaces.perimeter(hspace),
            dss_buffer.internal_elems,
        )
    end
    return nothing
end
"""
    weighted_dss_ghost!(
        data::Union{
            DataLayouts.IFH,
            DataLayouts.VIFH,
            DataLayouts.IJFH,
            DataLayouts.VIJFH,
        },
        space::Union{
            AbstractSpectralElementSpace,
            ExtrudedFiniteDifferenceSpace,
        },
        dss_buffer::Union{DSSBuffer, Nothing},
    )

1). Finish communications.

2). Call [`Spaces.load_from_recv_buffer!`](@ref)
After the communication is complete, this adds data from the recv buffer to the corresponding location in 
`perimeter_data`. For ghost vertices, this data is added only to the representative vertices. The values are 
then scattered to other local vertices corresponding to each unique ghost vertex in `dss_local_ghost`.

3). Call [`Spaces.dss_untransform!`](@ref) on all local elements.
This transforms the DSS'd local vectors back to Covariant12 vectors, and copies the DSS'd data from the
`perimeter_data` to `data`.
"""
weighted_dss_ghost!(
    data::Union{
        DataLayouts.IFH,
        DataLayouts.VIFH,
        DataLayouts.IJFH,
        DataLayouts.VIJFH,
    },
    space::Union{AbstractSpectralElementSpace, ExtrudedFiniteDifferenceSpace},
    dss_buffer::Union{DSSBuffer, Nothing},
) = weighted_dss_ghost!(data, space, horizontal_space(space), dss_buffer)

function weighted_dss_ghost!(
    data::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
    space::Union{AbstractSpectralElementSpace, ExtrudedFiniteDifferenceSpace},
    hspace::SpectralElementSpace2D{<:Topology2D},
    dss_buffer::DSSBuffer,
)
    assert_same_eltype(data, dss_buffer)
    length(parent(data)) == 0 && return data
    device = ClimaComms.device(topology(hspace))
    ClimaComms.finish(dss_buffer.graph_context)
    load_from_recv_buffer!(device, dss_buffer)
    dss_ghost!(
        device,
        dss_buffer.perimeter_data,
        Spaces.perimeter(hspace),
        topology(hspace),
    )
    dss_untransform!(
        device,
        dss_buffer,
        data,
        local_geometry_data(space),
        Spaces.perimeter(hspace),
        dss_buffer.perimeter_elems,
    )
    return data
end

weighted_dss_ghost!(data, space, hspace, dss_buffer) = data

"""
    function dss_transform!(
        device::ClimaComms.AbstractDevice,
        dss_buffer::DSSBuffer,
        data::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
        local_geometry::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
        weight::DataLayouts.IJFH,
        perimeter::AbstractPerimeter,
        localelems::Vector{Int},
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

Part of [`Spaces.weighted_dss!`](@ref).
"""
function dss_transform!(
    device::ClimaComms.AbstractDevice,
    dss_buffer::DSSBuffer,
    data::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
    local_geometry::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
    weight::DataLayouts.IJFH,
    perimeter::Perimeter2D,
    localelems::AbstractVector{Int},
)
    if !isempty(localelems)
        (; scalarfidx, covariant12fidx, contravariant12fidx, perimeter_data) =
            dss_buffer
        (; ∂ξ∂x, ∂x∂ξ) = local_geometry
        dss_transform!(
            device,
            perimeter_data,
            data,
            ∂ξ∂x,
            ∂x∂ξ,
            weight,
            perimeter,
            scalarfidx,
            covariant12fidx,
            contravariant12fidx,
            localelems,
        )
    end
    return nothing
end
"""
    dss_untransform!(
        device::ClimaComms.AbstractDevice,
        dss_buffer::DSSBuffer,
        data::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
        local_geometry::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
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

Part of [`Spaces.weighted_dss!`](@ref).
"""
function dss_untransform!(
    device::ClimaComms.AbstractDevice,
    dss_buffer::DSSBuffer,
    data::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
    local_geometry::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
    perimeter::Perimeter2D,
    localelems::AbstractVector{Int},
)
    (; scalarfidx, covariant12fidx, contravariant12fidx, perimeter_data) =
        dss_buffer
    (; ∂ξ∂x, ∂x∂ξ) = local_geometry
    dss_untransform!(
        device,
        perimeter_data,
        data,
        ∂ξ∂x,
        ∂x∂ξ,
        perimeter,
        scalarfidx,
        covariant12fidx,
        contravariant12fidx,
        localelems,
    )
    return nothing
end

"""
    function dss_transform!(
        ::ClimaComms.AbstractCPUDevice,
        perimeter_data::DataLayouts.VIFH,
        data::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
        ∂ξ∂x::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
        ∂x∂ξ::Union{DataLayouts.VIJFH, DataLayouts.IJFH},
        weight::DataLayouts.IJFH,
        perimeter::AbstractPerimeter,
        scalarfidx::Vector{Int},
        covariant12fidx::Vector{Int},
        contravariant12fidx::Vector{Int},
        localelems::Vector{Int},
    )

Transforms vectors from Covariant axes to physical (local axis), weights
the data at perimeter nodes, and stores result in the `perimeter_data` array.

Arguments:

- `perimeter_data`: contains the perimeter field data, represented on the physical axis, corresponding to the full field data in `data`
- `data`: field data
- `∂ξ∂x`: partial derivatives of the map from `x` to `ξ`: `∂ξ∂x[i,j]` is ∂ξⁱ/∂xʲ
- `weight`: local dss weights for horizontal space
- `perimeter`: perimeter iterator
- `scalarfidx`: field index for scalar fields in the data layout
- `covariant12fidx`: field index for Covariant12 vector fields in the data layout
- `localelems`: list of local elements to perform transformation operations on

Part of [`Spaces.weighted_dss!`](@ref).
"""
function dss_transform!(
    ::ClimaComms.AbstractCPUDevice,
    perimeter_data::DataLayouts.VIFH,
    data::Union{DataLayouts.VIJFH, DataLayouts.IJFH},
    ∂ξ∂x::Union{DataLayouts.VIJFH, DataLayouts.IJFH},
    ∂x∂ξ::Union{DataLayouts.VIJFH, DataLayouts.IJFH},
    weight::DataLayouts.IJFH,
    perimeter::Perimeter2D{Nq},
    scalarfidx::Vector{Int},
    covariant12fidx::Vector{Int},
    contravariant12fidx::Vector{Int},
    localelems::Vector{Int},
) where {Nq}
    pdata = parent(data)
    pweight = parent(weight)
    p∂x∂ξ = parent(∂x∂ξ)
    p∂ξ∂x = parent(∂ξ∂x)
    pperimeter_data = parent(perimeter_data)
    (nlevels, _, nfid, nelems) = size(pperimeter_data)
    nmetric = cld(length(p∂ξ∂x), prod(size(∂ξ∂x)))
    sizet_data = (nlevels, Nq, Nq, nfid, nelems)
    sizet_wt = (Nq, Nq, 1, nelems)
    sizet_metric = (nlevels, Nq, Nq, nmetric, nelems)

    @inbounds for elem in localelems
        for (p, (ip, jp)) in enumerate(perimeter)
            pw = pweight[_get_idx(sizet_wt, (ip, jp, 1, elem))]

            for fidx in scalarfidx, level in 1:nlevels
                data_idx = _get_idx(sizet_data, (level, ip, jp, fidx, elem))
                pperimeter_data[level, p, fidx, elem] = pdata[data_idx] * pw
            end

            for fidx in covariant12fidx, level in 1:nlevels
                data_idx1 = _get_idx(sizet_data, (level, ip, jp, fidx, elem))
                data_idx2 =
                    _get_idx(sizet_data, (level, ip, jp, fidx + 1, elem))
                (idx11, idx12, idx21, idx22) =
                    _get_idx_metric(sizet_metric, (level, ip, jp, elem))
                pperimeter_data[level, p, fidx, elem] =
                    (
                        p∂ξ∂x[idx11] * pdata[data_idx1] +
                        p∂ξ∂x[idx12] * pdata[data_idx2]
                    ) * pw
                pperimeter_data[level, p, fidx + 1, elem] =
                    (
                        p∂ξ∂x[idx21] * pdata[data_idx1] +
                        p∂ξ∂x[idx22] * pdata[data_idx2]
                    ) * pw
            end

            for fidx in contravariant12fidx, level in 1:nlevels
                data_idx1 = _get_idx(sizet_data, (level, ip, jp, fidx, elem))
                data_idx2 =
                    _get_idx(sizet_data, (level, ip, jp, fidx + 1, elem))
                (idx11, idx12, idx21, idx22) =
                    _get_idx_metric(sizet_metric, (level, ip, jp, elem))
                pperimeter_data[level, p, fidx, elem] =
                    (
                        p∂x∂ξ[idx11] * pdata[data_idx1] +
                        p∂x∂ξ[idx21] * pdata[data_idx2]
                    ) * pw
                pperimeter_data[level, p, fidx + 1, elem] =
                    (
                        p∂x∂ξ[idx12] * pdata[data_idx1] +
                        p∂x∂ξ[idx22] * pdata[data_idx2]
                    ) * pw
            end
        end
    end
    return nothing
end
"""
    function dss_untransform!(
        ::ClimaComms.AbstractCPUDevice,
        perimeter_data::DataLayouts.VIFH,
        data::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
        ∂ξ∂x::Union{DataLayouts.VIJFH, DataLayouts.IJFH},
        ∂x∂ξ::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
        perimeter::AbstractPerimeter,
        scalarfidx::Vector{Int},
        covariant12fidx::Vector{Int},
        contravariant12fidx::Vector{Int},
        localelems::Vector{Int},
    )

Transforms the DSS'd local vectors back to Covariant12 vectors, and copies the DSS'd data from the
`perimeter_data` to `data`.

Arguments:

- `perimeter_data`: contains the perimeter field data, represented on the physical axis, corresponding to the full field data in `data`
- `data`: field data
- `∂x∂ξ`: partial derivatives of the map from `ξ` to `x`: `∂x∂ξ[i,j]` is ∂xⁱ/∂ξʲ
- `perimeter`: perimeter iterator
- `scalarfidx`: field index for scalar fields in the data layout
- `covariant12fidx`: field index for Covariant12 vector fields in the data layout

Part of [`Spaces.weighted_dss!`](@ref).
"""

function dss_untransform!(
    ::ClimaComms.AbstractCPUDevice,
    perimeter_data::DataLayouts.VIFH,
    data::Union{DataLayouts.VIJFH, DataLayouts.IJFH},
    ∂ξ∂x::Union{DataLayouts.VIJFH, DataLayouts.IJFH},
    ∂x∂ξ::Union{DataLayouts.VIJFH, DataLayouts.IJFH},
    perimeter::Perimeter2D{Nq},
    scalarfidx::Vector{Int},
    covariant12fidx::Vector{Int},
    contravariant12fidx::Vector{Int},
    localelems::Vector{Int},
) where {Nq}
    pdata = parent(data)
    p∂x∂ξ = parent(∂x∂ξ)
    p∂ξ∂x = parent(∂ξ∂x)
    pperimeter_data = parent(perimeter_data)
    (nlevels, _, nfid, nelems) = size(pperimeter_data)
    nmetric = cld(length(p∂ξ∂x), prod(size(∂ξ∂x)))
    sizet_data = (nlevels, Nq, Nq, nfid, nelems)
    sizet_metric = (nlevels, Nq, Nq, nmetric, nelems)

    @inbounds for elem in localelems
        for (p, (ip, jp)) in enumerate(perimeter)
            for fidx in scalarfidx
                for level in 1:nlevels
                    data_idx = _get_idx(sizet_data, (level, ip, jp, fidx, elem))
                    pdata[data_idx] = pperimeter_data[level, p, fidx, elem]
                end
            end
            for fidx in covariant12fidx
                for level in 1:nlevels
                    data_idx1 =
                        _get_idx(sizet_data, (level, ip, jp, fidx, elem))
                    data_idx2 =
                        _get_idx(sizet_data, (level, ip, jp, fidx + 1, elem))
                    (idx11, idx12, idx21, idx22) =
                        _get_idx_metric(sizet_metric, (level, ip, jp, elem))
                    pdata[data_idx1] =
                        p∂x∂ξ[idx11] * pperimeter_data[level, p, fidx, elem] +
                        p∂x∂ξ[idx12] * pperimeter_data[level, p, fidx + 1, elem]
                    pdata[data_idx2] =
                        p∂x∂ξ[idx21] * pperimeter_data[level, p, fidx, elem] +
                        p∂x∂ξ[idx22] * pperimeter_data[level, p, fidx + 1, elem]
                end
            end
            for fidx in contravariant12fidx
                for level in 1:nlevels
                    data_idx1 =
                        _get_idx(sizet_data, (level, ip, jp, fidx, elem))
                    data_idx2 =
                        _get_idx(sizet_data, (level, ip, jp, fidx + 1, elem))
                    (idx11, idx12, idx21, idx22) =
                        _get_idx_metric(sizet_metric, (level, ip, jp, elem))
                    pdata[data_idx1] =
                        p∂ξ∂x[idx11] * pperimeter_data[level, p, fidx, elem] +
                        p∂ξ∂x[idx21] * pperimeter_data[level, p, fidx + 1, elem]
                    pdata[data_idx2] =
                        p∂ξ∂x[idx12] * pperimeter_data[level, p, fidx, elem] +
                        p∂ξ∂x[idx22] * pperimeter_data[level, p, fidx + 1, elem]
                end
            end
        end
    end
    return nothing
end

function dss_load_perimeter_data!(
    ::ClimaComms.AbstractCPUDevice,
    dss_buffer::DSSBuffer,
    data::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
    perimeter::Perimeter2D{Nq},
) where {Nq}
    pperimeter_data = parent(dss_buffer.perimeter_data)
    pdata = parent(data)
    (nlevels, _, nfid, nelems) = size(pperimeter_data)
    sizet = (nlevels, Nq, Nq, nfid, nelems)
    for elem in 1:nelems, (p, (ip, jp)) in enumerate(perimeter)
        for fidx in 1:nfid, level in 1:nlevels
            idx = _get_idx(sizet, (level, ip, jp, fidx, elem))
            pperimeter_data[level, p, fidx, elem] = pdata[idx]
        end
    end
    return nothing
end

function dss_unload_perimeter_data!(
    ::ClimaComms.AbstractCPUDevice,
    data::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
    dss_buffer::DSSBuffer,
    perimeter::Perimeter2D{Nq},
) where {Nq}
    pperimeter_data = parent(dss_buffer.perimeter_data)
    pdata = parent(data)
    (nlevels, _, nfid, nelems) = size(pperimeter_data)
    sizet = (nlevels, Nq, Nq, nfid, nelems)
    for elem in 1:nelems, (p, (ip, jp)) in enumerate(perimeter)
        for fidx in 1:nfid, level in 1:nlevels
            idx = _get_idx(sizet, (level, ip, jp, fidx, elem))
            pdata[idx] = pperimeter_data[level, p, fidx, elem]
        end
    end
    return nothing
end

"""
    function dss_local!(
        ::ClimaComms.AbstractCPUDevice,
        perimeter_data::DataLayouts.VIFH,
        perimeter::AbstractPerimeter,
        topology::Topologies.AbstractTopology,
    )

Performs DSS on local vertices and faces.

Part of [`Spaces.weighted_dss!`](@ref).
"""
function dss_local!(
    ::ClimaComms.AbstractCPUDevice,
    perimeter_data::DataLayouts.VIFH,
    perimeter::Perimeter2D,
    topology::Topologies.Topology2D,
)
    dss_local_vertices!(perimeter_data, perimeter, topology)
    dss_local_faces!(perimeter_data, perimeter, topology)
    return nothing
end

"""
    dss_local_vertices!(
        perimeter_data::DataLayouts.VIFH,
        perimeter::Perimeter2D,
        topology::Topologies.Topology2D,
    )

Apply dss to local vertices.
"""
function dss_local_vertices!(
    perimeter_data::DataLayouts.VIFH,
    perimeter::Perimeter2D,
    topology::Topologies.Topology2D,
)
    Nv = size(perimeter_data, 4)
    @inbounds for vertex in Topologies.local_vertices(topology)
        # for each level
        for level in 1:Nv
            # gather: compute sum over shared vertices
            sum_data = mapreduce(
                ⊞,
                vertex;
                init = RecursiveApply.rzero(eltype(slab(perimeter_data, 1, 1))),
            ) do (lidx, vert)
                ip = Topologies.perimeter_vertex_node_index(vert)
                perimeter_slab = slab(perimeter_data, level, lidx)
                perimeter_slab[ip]
            end
            # scatter: assign sum to shared vertices
            for (lidx, vert) in vertex
                perimeter_slab = slab(perimeter_data, level, lidx)
                ip = Topologies.perimeter_vertex_node_index(vert)
                perimeter_slab[ip] = sum_data
            end
        end
    end
    return nothing
end

function dss_local_faces!(
    perimeter_data::DataLayouts.VIFH,
    perimeter::Perimeter2D,
    topology::Topologies.Topology2D,
)
    (Np, _, _, Nv, _) = size(perimeter_data)
    nfacedof = div(Np - 4, 4)

    @inbounds for (lidx1, face1, lidx2, face2, reversed) in
                  Topologies.interior_faces(topology)
        pr1 = Topologies.perimeter_face_indices(face1, nfacedof, false)
        pr2 = Topologies.perimeter_face_indices(face2, nfacedof, reversed)
        for level in 1:Nv
            perimeter_slab1 = slab(perimeter_data, level, lidx1)
            perimeter_slab2 = slab(perimeter_data, level, lidx2)
            for (ip1, ip2) in zip(pr1, pr2)
                val = perimeter_slab1[ip1] ⊞ perimeter_slab2[ip2]
                perimeter_slab1[ip1] = val
                perimeter_slab2[ip2] = val
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
        topology::Topologies.AbstractTopology,
    )

Computes the "local" part of ghost vertex dss. (i.e. it computes the summation of all the shared local
vertices of a unique ghost vertex and stores the value in each of the local vertex locations in 
`perimeter_data`)

Part of [`Spaces.weighted_dss!`](@ref).
"""
function dss_local_ghost!(
    ::ClimaComms.AbstractCPUDevice,
    perimeter_data::DataLayouts.VIFH,
    perimeter::AbstractPerimeter,
    topology::Topologies.AbstractTopology,
)
    nghostvertices = length(topology.ghost_vertex_offset) - 1
    if nghostvertices > 0
        (Np, _, _, Nv, _) = size(perimeter_data)
        @inbounds for vertex in Topologies.ghost_vertices(topology)
            for level in 1:Nv
                # gather: compute sum over shared vertices
                sum_data = mapreduce(
                    ⊞,
                    vertex;
                    init = RecursiveApply.rzero(
                        eltype(slab(perimeter_data, 1, 1)),
                    ),
                ) do (isghost, idx, vert)
                    ip = Topologies.perimeter_vertex_node_index(vert)
                    if !isghost
                        lidx = idx
                        perimeter_slab = slab(perimeter_data, level, lidx)
                        perimeter_slab[ip]
                    else
                        RecursiveApply.rmap(zero, slab(perimeter_data, 1, 1)[1])
                    end
                end
                for (isghost, idx, vert) in vertex
                    if !isghost
                        ip = Topologies.perimeter_vertex_node_index(vert)
                        lidx = idx
                        perimeter_slab = slab(perimeter_data, level, lidx)
                        perimeter_slab[ip] = sum_data
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
        topology::Topologies.AbstractTopology,
    )

Sets the value for all local vertices of each unique ghost vertex, in `perimeter_data`, to that of 
the representative ghost vertex.

Part of [`Spaces.weighted_dss!`](@ref).
"""
function dss_ghost!(
    device::ClimaComms.AbstractCPUDevice,
    perimeter_data::DataLayouts.VIFH,
    perimeter::AbstractPerimeter,
    topology::Topologies.AbstractTopology,
)
    nghostvertices = length(topology.ghost_vertex_offset) - 1
    if nghostvertices > 0
        nlevels = size(perimeter_data, 4)
        perimeter_vertex_node_index = Topologies.perimeter_vertex_node_index
        perimeter_face_indices = Topologies.perimeter_face_indices
        (; repr_ghost_vertex) = topology
        @inbounds for (i, vertex) in
                      enumerate(Topologies.ghost_vertices(topology))
            idxresult, lvertresult = repr_ghost_vertex[i]
            ipresult = perimeter_vertex_node_index(lvertresult)
            for level in 1:nlevels
                result_slab = slab(perimeter_data, level, idxresult)
                result = result_slab[ipresult]
                for (isghost, idx, vert) in vertex
                    if !isghost
                        ip = perimeter_vertex_node_index(vert)
                        lidx = idx
                        perimeter_slab = slab(perimeter_data, level, lidx)
                        perimeter_slab[ip] = result
                    end
                end
            end
        end
    end
    return nothing
end

"""
    fill_send_buffer!(::ClimaComms.AbstractCPUDevice, dss_buffer::DSSBuffer)

Loads the send buffer from `perimeter_data`. For unique ghost vertices, only data from the
representative vertices which store result of "ghost local" DSS are loaded.

Part of [`Spaces.weighted_dss!`](@ref).
"""
function fill_send_buffer!(
    ::ClimaComms.AbstractCPUDevice,
    dss_buffer::DSSBuffer,
)
    (; perimeter_data, send_buf_idx, send_data) = dss_buffer
    (Np, _, _, Nv, nelems) = size(perimeter_data)
    Nf = cld(length(parent(perimeter_data)), (Nv * Np * nelems))
    pdata = parent(perimeter_data)
    nsend = size(send_buf_idx, 1)
    ctr = 1
    @inbounds for i in 1:nsend
        lidx = send_buf_idx[i, 1]
        ip = send_buf_idx[i, 2]
        for f in 1:Nf, v in 1:Nv
            send_data[ctr] = pdata[v, ip, f, lidx]
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

Part of [`Spaces.weighted_dss!`](@ref).
"""
function load_from_recv_buffer!(
    ::ClimaComms.AbstractCPUDevice,
    dss_buffer::DSSBuffer,
)
    (; perimeter_data, recv_buf_idx, recv_data) = dss_buffer
    (Np, _, _, Nv, nelems) = size(perimeter_data)
    Nf = cld(length(parent(perimeter_data)), (Nv * Np * nelems))
    pdata = parent(perimeter_data)
    nrecv = size(recv_buf_idx, 1)
    ctr = 1
    @inbounds for i in 1:nrecv
        lidx = recv_buf_idx[i, 1]
        ip = recv_buf_idx[i, 2]
        for f in 1:Nf, v in 1:Nv
            pdata[v, ip, f, lidx] += recv_data[ctr]
            ctr += 1
        end
    end
    return nothing
end

"""
    dss!(data, topology, quadrature_style)

Computed unweighted/pure DSS of `data`.
"""
function dss!(data, topology, quadrature_style)
    length(parent(data)) == 0 && return nothing
    device = ClimaComms.device(topology)
    perimeter = Perimeter2D(Quadratures.degrees_of_freedom(quadrature_style))
    # create dss buffer
    dss_buffer = create_dss_buffer(data, topology, perimeter)
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

dss2!(data, topology, quadrature_style) = dss!(data, topology, quadrature_style)

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

include("dss_cuda.jl")
