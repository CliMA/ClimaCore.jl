import ..Topologies:
    DSSBuffer,
    create_dss_buffer,
    assert_same_eltype,
    dss!,
    dss_1d!,
    dss_transform!,
    dss_untransform!,
    dss_local!,
    dss_local_ghost!,
    dss_ghost!,
    fill_send_buffer!,
    load_from_recv_buffer!


perimeter(space::AbstractSpectralElementSpace) = Topologies.Perimeter2D(
    Quadratures.degrees_of_freedom(quadrature_style(space)),
)


"""
    create_dss_buffer(
        data::Union{DataLayouts.IJFH{S, Nij}, DataLayouts.VIJFH{S, Nij}},
        hspace::AbstractSpectralElementSpace,
    ) where {S, Nij}

Creates a [`DSSBuffer`](@ref) for the field data corresponding to `data`
"""
function create_dss_buffer(
    data::Union{DataLayouts.IJFH{S, Nij}, DataLayouts.VIJFH{S, Nij}},
    hspace::SpectralElementSpace2D,
) where {S, Nij}
    create_dss_buffer(
        data,
        topology(hspace),
        local_geometry_data(hspace),
        local_dss_weights(hspace),
    )
end

function create_dss_buffer(
    data::Union{DataLayouts.IFH{S, Nij}, DataLayouts.VIFH{S, Nij}},
    hspace::SpectralElementSpace1D,
) where {S, Nij}
    nothing
end

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

function weighted_dss_prepare!(
    data::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
    space::Union{
        Spaces.SpectralElementSpace2D,
        Spaces.ExtrudedFiniteDifferenceSpace,
    },
    dss_buffer::DSSBuffer,
)
    assert_same_eltype(data, dss_buffer)
    length(parent(data)) == 0 && return nothing
    device = ClimaComms.device(topology(space))
    hspace = horizontal_space(space)
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
    fill_send_buffer!(device, dss_buffer; synchronize = false)
    return nothing
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
function weighted_dss_start!(
    data::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
    space::Union{
        Spaces.SpectralElementSpace2D,
        Spaces.ExtrudedFiniteDifferenceSpace,
    },
    dss_buffer::DSSBuffer,
)
    device = ClimaComms.device(topology(space))
    weighted_dss_prepare!(data, space, dss_buffer)
    if device isa ClimaComms.CUDADevice
        CUDA.synchronize(; blocking = true)
    end
    ClimaComms.start(dss_buffer.graph_context)
    return nothing
end

weighted_dss_start!(data, space, dss_buffer::Nothing) = nothing

# TODO: deprecate
weighted_dss_start!(data, space, hspace, dss_buffer) =
    weighted_dss_start!(data, space, dss_buffer)


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
    hspace::SpectralElementSpace2D,
    dss_buffer::DSSBuffer,
)
    assert_same_eltype(data, dss_buffer)
    ClimaComms.finish(dss_buffer.graph_context)
    length(parent(data)) == 0 && return data
    device = ClimaComms.device(topology(hspace))
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


# TODO: deprecate


# for backward compatibility
function weighted_dss2! end
function weighted_dss_start2! end
function weighted_dss_internal2! end
function weighted_dss_ghost2! end
function dss2! end


dss2!(data, topology, quadrature_style) = dss!(data, topology)
