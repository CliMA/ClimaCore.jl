using DocStringExtensions

"""
    DSSBuffer{G, D, A, B}

# Fields
$(DocStringExtensions.FIELDS)
"""
struct DSSBuffer{G, D, A, B}
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
    scalarfidx::Vector{Int}
    "field id for all vector fields stored in the `data` array"
    vectorfidx::Vector{Int}
    "internal local elements (lidx)"
    internal_elems::Vector{Int}
    "local elements (lidx) located on process boundary"
    perimeter_elems::Vector{Int}
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
    @assert hspace.quadrature_style isa Spaces.Quadratures.GLL "DSS2 is only compatible with GLL quadrature"
    topology = hspace.topology
    context =
        topology isa Topologies.Topology2D ? topology.context :
        ClimaComms.SingletonCommsContext()
    local_geometry = local_geometry_data(hspace)
    local_weights = hspace.local_dss_weights

    (_, _, _, Nv, nelems) = Base.size(data)
    perimeter = Spaces.perimeter(hspace)
    Np = Spaces.nperimeter(perimeter)
    Nf = cld(length(parent(data)), (Nij * Nij * Nv * nelems))
    nfacedof = Nij - 2
    T = eltype(parent(data))
    TS = typeof(
        dss_transform(
            slab(data, 1, 1),
            slab(local_geometry, 1, 1),
            slab(local_weights, 1, 1),
            1,
            1,
        ),
    ) # extract transformed type
    perimeter_data =
        DataLayouts.VIFH{TS, Np}(Array{T}(undef, Nv, Np, Nf, nelems))
    if context isa ClimaComms.SingletonCommsContext
        send_data, recv_data = T[], T[]
        send_buf_idx, recv_buf_idx = Int[], Int[]
        graph_context = ClimaComms.SingletonGraphContext(context)
        internal_elems = Vector{Int}(1:Topologies.nelems(topology))
        perimeter_elems = Int[]
    else
        (; comm_vertex_lengths, comm_face_lengths) = topology
        vertex_buffer_lengths = comm_vertex_lengths .* (Nv * Nf)
        face_buffer_lengths = comm_face_lengths .* (Nv * Nf * nfacedof)
        buffer_lengths = vertex_buffer_lengths .+ face_buffer_lengths
        buffer_size = sum(buffer_lengths)
        send_data = Vector{T}(undef, buffer_size)
        recv_data = Vector{T}(undef, buffer_size)
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
        internal_elems = topology.internal_elems
        perimeter_elems = topology.perimeter_elems
    end
    scalarfidx, vectorfidx = Int[], Int[], Int[], Int[]
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
    }
    if S <: NamedTuple
        for (i, fieldtype) in enumerate(S.parameters[2].types)
            offset = DataLayouts.fieldtypeoffset(T, S, i)
            ncomponents = DataLayouts.typesize(T, fieldtype)
            if fieldtype <: Geometry.AxisVector # vector fields
                @assert fieldtype <: supportedvectortypes
                if fieldtype <: Geometry.Covariant12Vector
                    push!(vectorfidx, offset + 1)
                else
                    append!(
                        scalarfidx,
                        Vector((offset + 1):(offset + ncomponents)),
                    )
                end
            else # scalar fields
                push!(scalarfidx, offset + 1)
            end
        end
    else # deals with simple type, with single field (e.g: S = Float64, S = CovariantVector12, etc.)
        ncomponents = DataLayouts.typesize(T, S)
        if S <: Geometry.AxisVector # vector field
            @assert S <: supportedvectortypes
            if S <: Geometry.Covariant12Vector
                push!(vectorfidx, 1)
            else
                append!(scalarfidx, Vector(1:ncomponents))
            end
        else # scalar field
            push!(scalarfidx, 1)
        end
    end
    return DSSBuffer{
        typeof(graph_context),
        typeof(perimeter_data),
        typeof(send_data),
        typeof(send_buf_idx),
    }(
        graph_context,
        perimeter_data,
        send_data,
        recv_data,
        send_buf_idx,
        recv_buf_idx,
        scalarfidx,
        vectorfidx,
        internal_elems,
        perimeter_elems,
    )
end

create_dss_buffer(data::DataLayouts.AbstractData, hspace) = nothing

"""
    function weighted_dss2!(
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

1). [`Spaces.weighted_dss_start2!`](@ref)

2). [`Spaces.weighted_dss_internal2!`](@ref)

3). [`Spaces.weighted_dss_ghost2!`](@ref)
"""
function weighted_dss2!(
    data::Union{
        DataLayouts.IFH,
        DataLayouts.VIFH,
        DataLayouts.IJFH,
        DataLayouts.VIJFH,
    },
    space::Union{AbstractSpectralElementSpace, ExtrudedFiniteDifferenceSpace},
    dss_buffer::Union{DSSBuffer, Nothing},
)
    weighted_dss_start2!(data, space, dss_buffer)
    weighted_dss_internal2!(data, space, dss_buffer)
    weighted_dss_ghost2!(data, space, dss_buffer)
end

"""
    weighted_dss_start2!(
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

1). Apply [`Spaces.dss_transform2!`](@ref) on perimeter elements. This weights and tranforms vector 
fields to physical basis if needed. Scalar fields are weighted. The transformed and/or weighted 
perimeter `data` is stored in `perimeter_data`.

2). Apply [`Spaces.dss_local_ghost2!`](@ref)
This computes partial weighted DSS on ghost vertices, using only the information from `local` vertices.

3). [`Spaces.fill_send_buffer2!`](@ref) 
Loads the send buffer from `perimeter_data`. For unique ghost vertices, only data from the
representative ghost vertices which store result of "ghost local" DSS are loaded.

4). Start DSS communication with neighboring processes
"""
weighted_dss_start2!(
    data::Union{
        DataLayouts.IFH,
        DataLayouts.VIFH,
        DataLayouts.IJFH,
        DataLayouts.VIJFH,
    },
    space::Union{AbstractSpectralElementSpace, ExtrudedFiniteDifferenceSpace},
    dss_buffer::Union{DSSBuffer, Nothing},
) = weighted_dss_start2!(data, space, horizontal_space(space), dss_buffer)

function weighted_dss_start2!(
    data::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
    space::Union{
        Spaces.SpectralElementSpace2D,
        Spaces.ExtrudedFiniteDifferenceSpace,
    },
    hspace::SpectralElementSpace2D{<:Topology2D},
    dss_buffer::DSSBuffer,
)
    dss_transform2!(
        dss_buffer,
        data,
        local_geometry_data(space),
        hspace.local_dss_weights,
        Spaces.perimeter(hspace),
        dss_buffer.perimeter_elems,
    )
    dss_local_ghost2!(
        dss_buffer.perimeter_data,
        Spaces.perimeter(hspace),
        hspace.topology,
    )
    fill_send_buffer2!(dss_buffer)
    ClimaComms.start(dss_buffer.graph_context)
    return nothing
end

weighted_dss_start2!(data, space, hspace, dss_buffer) = nothing
"""
    weighted_dss_internal2!(
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

1). Apply [`Spaces.dss_transform2!`](@ref) on interior elements. Local elements are split into interior 
and perimeter elements to facilitate overlapping of communication with computation.

2). Probe communication

3). [`Spaces.dss_local2!`](@ref) computes the weighted DSS on local vertices and faces.
"""
weighted_dss_internal2!(
    data::Union{
        DataLayouts.IFH,
        DataLayouts.VIFH,
        DataLayouts.IJFH,
        DataLayouts.VIJFH,
    },
    space::Union{AbstractSpectralElementSpace, ExtrudedFiniteDifferenceSpace},
    dss_buffer::Union{DSSBuffer, Nothing},
) = weighted_dss_internal2!(data, space, horizontal_space(space), dss_buffer)

function weighted_dss_internal2!(
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
    if hspace isa SpectralElementSpace1D
        dss_1d!(
            hspace.topology,
            data,
            local_geometry_data(space),
            hspace.dss_weights,
        )
    else
        dss_transform2!(
            dss_buffer,
            data,
            local_geometry_data(space),
            hspace.local_dss_weights,
            Spaces.perimeter(hspace),
            dss_buffer.internal_elems,
        )
        dss_local2!(
            dss_buffer.perimeter_data,
            Spaces.perimeter(hspace),
            hspace.topology,
        )
        dss_untransform2!(
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
    weighted_dss_ghost2!(
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

2). Call [`Spaces.load_from_recv_buffer2!`](@ref)
After the communication is complete, this adds data from the recv buffer to the corresponding location in 
`perimeter_data`. For ghost vertices, this data is added only to the representative vertices. The values are 
then scattered to other local vertices corresponding to each unique ghost vertex in `dss_local_ghost`.

3). Call [`Spaces.dss_untransform2!`](@ref) on all local elements.
This transforms the DSS'd local vectors back to Covariant12 vectors, and copies the DSS'd data from the
`perimeter_data` to `data`.
"""
weighted_dss_ghost2!(
    data::Union{
        DataLayouts.IFH,
        DataLayouts.VIFH,
        DataLayouts.IJFH,
        DataLayouts.VIJFH,
    },
    space::Union{AbstractSpectralElementSpace, ExtrudedFiniteDifferenceSpace},
    dss_buffer::Union{DSSBuffer, Nothing},
) = weighted_dss_ghost2!(data, space, horizontal_space(space), dss_buffer)

function weighted_dss_ghost2!(
    data::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
    space::Union{AbstractSpectralElementSpace, ExtrudedFiniteDifferenceSpace},
    hspace::SpectralElementSpace2D{<:Topology2D},
    dss_buffer::DSSBuffer,
)
    ClimaComms.finish(dss_buffer.graph_context)
    load_from_recv_buffer2!(dss_buffer)
    dss_ghost2!(
        dss_buffer.perimeter_data,
        Spaces.perimeter(hspace),
        hspace.topology,
    )
    dss_untransform2!(
        dss_buffer,
        data,
        local_geometry_data(space),
        Spaces.perimeter(hspace),
        dss_buffer.perimeter_elems,
    )
    return data
end

weighted_dss_ghost2!(data, space, hspace, dss_buffer) = data

"""
    function dss_transform2!(
        dss_buffer::DSSBuffer,
        data::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
        local_geometry::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
        weight::DataLayouts.IJFH,
        perimeter::AbstractPerimeter,
        localelems::Vector{Int},
    )

Transforms vectors from Covariant axes to physical (local axis), weights the data at perimeter nodes, 
and stores result in the `perimeter_data` array. This function calls the appropriate version of 
`dss_transform2!` based on the data layout of the input arguments.

Arguments:

- `dss_buffer`: [`DSSBuffer`](@ref) generated by `create_dss_buffer` function for field data
- `data`: field data
- `local_geometry`: local metric information defined at each node
- `weight`: local dss weights for horizontal space
- `perimeter`: perimeter iterator
- `localelems`: list of local elements to perform transformation operations on

Part of [`Spaces.weighted_dss2!`](@ref).
"""
function dss_transform2!(
    dss_buffer::DSSBuffer,
    data::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
    local_geometry::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
    weight::DataLayouts.IJFH,
    perimeter::AbstractPerimeter,
    localelems::Vector{Int},
)
    if !isempty(localelems)
        (; scalarfidx, vectorfidx, perimeter_data) = dss_buffer
        ∂ξ∂x = local_geometry.∂ξ∂x
        dss_transform2!(
            perimeter_data,
            data,
            ∂ξ∂x,
            weight,
            perimeter,
            scalarfidx,
            vectorfidx,
            localelems,
        )
    end
    return nothing
end
"""
    dss_untransform2!(
        dss_buffer::DSSBuffer,
        data::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
        local_geometry::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
        perimeter::AbstractPerimeter,
    )

Transforms the DSS'd local vectors back to Covariant12 vectors, and copies the DSS'd data from the
`perimeter_data` to `data`. This function calls the appropriate version of `dss_transform2!` function
based on the data layout of the input arguments.

Arguments:

- `dss_buffer`: [`DSSBuffer`](@ref) generated by `create_dss_buffer` function for field data
- `data`: field data
- `local_geometry`: local metric information defined at each node
- `perimeter`: perimeter iterator
- `localelems`: list of local elements to perform transformation operations on

Part of [`Spaces.weighted_dss2!`](@ref).
"""
function dss_untransform2!(
    dss_buffer::DSSBuffer,
    data::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
    local_geometry::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
    perimeter::AbstractPerimeter,
    localelems::Vector{Int},
)
    (; scalarfidx, vectorfidx, perimeter_data) = dss_buffer
    ∂x∂ξ = local_geometry.∂x∂ξ
    dss_untransform2!(
        perimeter_data,
        data,
        ∂x∂ξ,
        perimeter,
        scalarfidx,
        vectorfidx,
        localelems,
    )
    return nothing
end

"""
    function dss_transform2!(
        perimeter_data::DataLayouts.VIFH,
        data::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
        ∂ξ∂x::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
        weight::DataLayouts.IJFH,
        perimeter::AbstractPerimeter,
        scalarfidx::Vector{Int},
        vectorfidx::Vector{Int},
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
- `vectorfidx`: field index for Covariant12 vector fields in the data layout
- `localelems`: list of local elements to perform transformation operations on

Part of [`Spaces.weighted_dss2!`](@ref).
"""
function dss_transform2!(
    perimeter_data::DataLayouts.VIFH,
    data::DataLayouts.IJFH,
    ∂ξ∂x::DataLayouts.IJFH,
    weight::DataLayouts.IJFH,
    perimeter::AbstractPerimeter,
    scalarfidx::Vector{Int},
    vectorfidx::Vector{Int},
    localelems::Vector{Int},
)
    pdata = parent(data)
    pweight = parent(weight)
    p∂ξ∂x = parent(∂ξ∂x)
    pperimeter_data = parent(perimeter_data)
    (i11, i12, i21, i22) =
        length(p∂ξ∂x) / prod(size(∂ξ∂x)) == 4 ? (1, 3, 2, 4) : (1, 4, 2, 5)

    @inbounds for elem in localelems
        for (p, (ip, jp)) in enumerate(perimeter)

            for fidx in scalarfidx
                pperimeter_data[1, p, fidx, elem] =
                    pdata[ip, jp, fidx, elem] * pweight[ip, jp, 1, elem]
            end

            for fidx in vectorfidx
                pperimeter_data[1, p, fidx, elem] =
                    (
                        p∂ξ∂x[ip, jp, i11, elem] * pdata[ip, jp, fidx, elem] +
                        p∂ξ∂x[ip, jp, i21, elem] *
                        pdata[ip, jp, fidx + 1, elem]
                    ) * pweight[ip, jp, 1, elem]
                pperimeter_data[1, p, fidx + 1, elem] =
                    (
                        p∂ξ∂x[ip, jp, i12, elem] * pdata[ip, jp, fidx, elem] +
                        p∂ξ∂x[ip, jp, i22, elem] *
                        pdata[ip, jp, fidx + 1, elem]
                    ) * pweight[ip, jp, 1, elem]
            end

        end
    end
    return nothing
end

function dss_transform2!(
    perimeter_data::DataLayouts.VIFH,
    data::DataLayouts.VIJFH,
    ∂ξ∂x::DataLayouts.VIJFH,
    weight::DataLayouts.IJFH,
    perimeter::AbstractPerimeter,
    scalarfidx::Vector{Int},
    vectorfidx::Vector{Int},
    localelems::Vector{Int},
)
    Nv = size(data, 4)
    pdata = parent(data)
    pweight = parent(weight)
    p∂ξ∂x = parent(∂ξ∂x)
    pperimeter_data = parent(perimeter_data)
    (i11, i12, i21, i22) =
        length(p∂ξ∂x) / prod(size(∂ξ∂x)) == 4 ? (1, 3, 2, 4) : (1, 4, 2, 5)

    @inbounds for elem in localelems
        for (p, (ip, jp)) in enumerate(perimeter)

            for fidx in scalarfidx
                for level in 1:Nv
                    pperimeter_data[level, p, fidx, elem] =
                        pdata[level, ip, jp, fidx, elem] *
                        pweight[ip, jp, 1, elem]
                end
            end

            for fidx in vectorfidx
                for level in 1:Nv
                    pperimeter_data[level, p, fidx, elem] =
                        (
                            p∂ξ∂x[level, ip, jp, i11, elem] *
                            pdata[level, ip, jp, fidx, elem] +
                            p∂ξ∂x[level, ip, jp, i21, elem] *
                            pdata[level, ip, jp, fidx + 1, elem]
                        ) * pweight[ip, jp, 1, elem]
                    pperimeter_data[level, p, fidx + 1, elem] =
                        (
                            p∂ξ∂x[level, ip, jp, i12, elem] *
                            pdata[level, ip, jp, fidx, elem] +
                            p∂ξ∂x[level, ip, jp, i22, elem] *
                            pdata[level, ip, jp, fidx + 1, elem]
                        ) * pweight[ip, jp, 1, elem]
                end
            end
        end
    end
    return nothing
end
"""
    function dss_untransform2!(
        perimeter_data::DataLayouts.VIFH,
        data::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
        ∂x∂ξ::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
        perimeter::AbstractPerimeter,
        scalarfidx::Vector{Int},
        vectorfidx::Vector{Int},
    )

Transforms the DSS'd local vectors back to Covariant12 vectors, and copies the DSS'd data from the
`perimeter_data` to `data`.

Arguments:

- `perimeter_data`: contains the perimeter field data, represented on the physical axis, corresponding to the full field data in `data`
- `data`: field data
- `∂x∂ξ`: partial derivatives of the map from `ξ` to `x`: `∂x∂ξ[i,j]` is ∂xⁱ/∂ξʲ
- `perimeter`: perimeter iterator
- `scalarfidx`: field index for scalar fields in the data layout
- `vectorfidx`: field index for Covariant12 vector fields in the data layout

Part of [`Spaces.weighted_dss2!`](@ref).
"""
function dss_untransform2!(
    perimeter_data::DataLayouts.VIFH,
    data::DataLayouts.IJFH,
    ∂x∂ξ::DataLayouts.IJFH,
    perimeter::AbstractPerimeter,
    scalarfidx::Vector{Int},
    vectorfidx::Vector{Int},
    localelems::Vector{Int},
)
    nelems = size(data, 5)
    pdata = parent(data)
    p∂x∂ξ = parent(∂x∂ξ)
    pperimeter_data = parent(perimeter_data)
    (i11, i12, i21, i22) =
        length(p∂x∂ξ) / prod(size(∂x∂ξ)) == 4 ? (1, 3, 2, 4) : (1, 4, 2, 5)

    @inbounds for elem in localelems
        for (p, (ip, jp)) in enumerate(perimeter)
            for fidx in vectorfidx
                pdata[ip, jp, fidx, elem] =
                    p∂x∂ξ[ip, jp, i11, elem] *
                    pperimeter_data[1, p, fidx, elem] +
                    p∂x∂ξ[ip, jp, i21, elem] *
                    pperimeter_data[1, p, fidx + 1, elem]
                pdata[ip, jp, fidx + 1, elem] =
                    p∂x∂ξ[ip, jp, i12, elem] *
                    pperimeter_data[1, p, fidx, elem] +
                    p∂x∂ξ[ip, jp, i22, elem] *
                    pperimeter_data[1, p, fidx + 1, elem]
            end
            for fidx in scalarfidx
                pdata[ip, jp, fidx, elem] = pperimeter_data[1, p, fidx, elem]
            end
        end
    end
    return nothing
end

function dss_untransform2!(
    perimeter_data::DataLayouts.VIFH,
    data::DataLayouts.VIJFH,
    ∂x∂ξ::DataLayouts.VIJFH,
    perimeter::AbstractPerimeter,
    scalarfidx::Vector{Int},
    vectorfidx::Vector{Int},
    localelems::Vector{Int},
)
    (_, _, _, Nv, nelems) = size(data)
    pdata = parent(data)
    p∂x∂ξ = parent(∂x∂ξ)
    pperimeter_data = parent(perimeter_data)
    (i11, i12, i21, i22) =
        length(p∂x∂ξ) / prod(size(∂x∂ξ)) == 4 ? (1, 3, 2, 4) : (1, 4, 2, 5)

    @inbounds for elem in localelems
        for (p, (ip, jp)) in enumerate(perimeter)
            for fidx in vectorfidx
                for level in 1:Nv
                    pdata[level, ip, jp, fidx, elem] =
                        p∂x∂ξ[level, ip, jp, i11, elem] *
                        pperimeter_data[level, p, fidx, elem] +
                        p∂x∂ξ[level, ip, jp, i21, elem] *
                        pperimeter_data[level, p, fidx + 1, elem]
                    pdata[level, ip, jp, fidx + 1, elem] =
                        p∂x∂ξ[level, ip, jp, i12, elem] *
                        pperimeter_data[level, p, fidx, elem] +
                        p∂x∂ξ[level, ip, jp, i22, elem] *
                        pperimeter_data[level, p, fidx + 1, elem]
                end
            end
            for fidx in scalarfidx
                for level in 1:Nv
                    pdata[level, ip, jp, fidx, elem] =
                        pperimeter_data[level, p, fidx, elem]
                end
            end
        end
    end
    return nothing
end

"""
    function dss_local2!(
        perimeter_data::DataLayouts.VIFH,
        perimeter::AbstractPerimeter,
        topology::Topologies.AbstractTopology,
    )

Performs DSS on local vertices and faces.

Part of [`Spaces.weighted_dss2!`](@ref).
"""
function dss_local2!(
    perimeter_data::DataLayouts.VIFH,
    perimeter::AbstractPerimeter,
    topology::Topologies.AbstractTopology,
)
    (Np, _, _, Nv, _) = size(perimeter_data)
    nfacedof = div(Np - 4, 4)

    @inbounds for vertex in Topologies.local_vertices(topology)
        # for each level
        for level in 1:Nv
            # gather: compute sum over shared vertices
            sum_data = mapreduce(⊞, vertex) do (lidx, vert)
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
    function dss_local_ghost2!(
        perimeter_data::DataLayouts.VIFH,
        perimeter::AbstractPerimeter,
        topology::Topologies.AbstractTopology,
    )

Computes the "local" part of ghost vertex dss. (i.e. it computes the summation of all the shared local
vertices of a unique ghost vertex and stores the value in each of the local vertex locations in 
`perimeter_data`)

Part of [`Spaces.weighted_dss2!`](@ref).
"""
function dss_local_ghost2!(
    perimeter_data::DataLayouts.VIFH,
    perimeter::AbstractPerimeter,
    topology::Topologies.AbstractTopology,
)
    (Np, _, _, Nv, _) = size(perimeter_data)
    nfacedof = div(Np - 4, 4)
    zero_data = map(zero, slab(perimeter_data, 1, 1)[1])
    @inbounds for vertex in Topologies.ghost_vertices(topology)
        for level in 1:Nv
            # gather: compute sum over shared vertices
            sum_data = mapreduce(⊞, vertex) do (isghost, idx, vert)
                ip = Topologies.perimeter_vertex_node_index(vert)
                if !isghost
                    lidx = idx
                    perimeter_slab = slab(perimeter_data, level, lidx)
                    perimeter_slab[ip]
                else
                    zero_data
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
    return nothing
end
"""
    dss_ghost2!(
        perimeter_data::DataLayouts.VIFH,
        perimeter::AbstractPerimeter,
        topology::Topologies.AbstractTopology,
    )

Sets the value for all local vertices of each unique ghost vertex, in `perimeter_data`, to that of 
the representative ghost vertex.

Part of [`Spaces.weighted_dss2!`](@ref).
"""
function dss_ghost2!(
    perimeter_data::DataLayouts.VIFH,
    perimeter::AbstractPerimeter,
    topology::Topologies.AbstractTopology,
)
    Nv = size(perimeter_data, 4)
    Np = size(perimeter_data, 1)
    nfacedof = div(Np - 4, 4)
    perimeter_vertex_node_index = Topologies.perimeter_vertex_node_index
    perimeter_face_indices = Topologies.perimeter_face_indices
    (; repr_ghost_vertex) = topology
    @inbounds for (i, vertex) in enumerate(Topologies.ghost_vertices(topology))
        idxresult, lvertresult = repr_ghost_vertex[i]
        ipresult = perimeter_vertex_node_index(lvertresult)
        for level in 1:Nv
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
    return nothing
end

"""
    fill_send_buffer2!(dss_buffer::DSSBuffer)

Loads the send buffer from `perimeter_data`. For unique ghost vertices, only data from the
representative vertices which store result of "ghost local" DSS are loaded.

Part of [`Spaces.weighted_dss2!`](@ref).
"""
function fill_send_buffer2!(dss_buffer::DSSBuffer)
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
    load_from_recv_buffer2!(dss_buffer::DSSBuffer)

Adds data from the recv buffer to the corresponding location in `perimeter_data`.
For ghost vertices, this data is added only to the representative vertices. The values are 
then scattered to other local vertices corresponding to each unique ghost vertex in `dss_local_ghost`.

Part of [`Spaces.weighted_dss2!`](@ref).
"""
function load_from_recv_buffer2!(dss_buffer::DSSBuffer)
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

function dss_communicate2!(graph_context)
    ClimaComms.start(graph_context)
    ClimaComms.progress(graph_context)
    ClimaComms.finish(graph_context)
    return nothing
end
