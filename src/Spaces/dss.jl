import ..Topologies:
    DSSBuffer,
    create_dss_buffer,
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
    create_dss_buffer(data, space)

Creates a [`DSSBuffer`](@ref) for the field data corresponding to `data`
"""
create_dss_buffer(data::DataLayouts.VIJHWithF, space) =
    isone(size(data, 3)) ? nothing :
    create_dss_buffer(
        data,
        topology(space),
        local_geometry_data(space),
        dss_weights(space),
    )

"""
    function weighted_dss!(data, space, dss_buffer)

Computes weighted dss of `data`. 

It comprises of the following steps:

1). [`Spaces.weighted_dss_start!`](@ref)

2). [`Spaces.weighted_dss_internal!`](@ref)

3). [`Spaces.weighted_dss_ghost!`](@ref)
"""
function weighted_dss!(data::DataLayouts.VIJHWithF, space, dss_buffer)
    weighted_dss_start!(data, space, dss_buffer)
    weighted_dss_internal!(data, space, dss_buffer)
    weighted_dss_ghost!(data, space, dss_buffer)
    call_post_op_callback() && post_op_callback(data, data, space, dss_buffer)
end

function weighted_dss_prepare!(data, space, dss_buffer)
    isnothing(dss_buffer) && return nothing
    device = ClimaComms.device(topology(space))
    hspace = horizontal_space(space)
    dss_transform!(
        device,
        dss_buffer,
        data,
        local_geometry_data(space),
        dss_weights(space),
        perimeter(hspace),
        dss_buffer.perimeter_elems,
    )
    dss_local_ghost!(
        device,
        dss_buffer.perimeter_data,
        perimeter(hspace),
        topology(hspace),
    )
    fill_send_buffer!(device, dss_buffer)
    return nothing
end

cuda_synchronize(device::ClimaComms.AbstractDevice; kwargs...) = nothing

"""
    weighted_dss_start!(data, space, dss_buffer)

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
function weighted_dss_start!(data, space, dss_buffer)
    isnothing(dss_buffer) && return nothing
    Quadratures.requires_dss(quadrature_style(space)) || return nothing
    sizeof(eltype(data)) > 0 || return nothing
    device = ClimaComms.device(topology(space))
    weighted_dss_prepare!(data, space, dss_buffer)
    cuda_synchronize(device; blocking = true)
    ClimaComms.start(dss_buffer.graph_context)
    return nothing
end

"""
    weighted_dss_internal!(data, space, dss_buffer)

1). Apply [`Spaces.dss_transform!`](@ref) on interior elements. Local elements are split into interior 
and perimeter elements to facilitate overlapping of communication with computation.

2). Probe communication

3). [`Spaces.dss_local!`](@ref) computes the weighted DSS on local vertices and faces.
"""
function weighted_dss_internal!(data, space, dss_buffer)
    Quadratures.requires_dss(quadrature_style(space)) || return nothing
    sizeof(eltype(data)) > 0 || return nothing
    hspace = horizontal_space(space)
    device = ClimaComms.device(topology(hspace))
    if hspace isa SpectralElementSpace1D
        dss_1d!(
            device,
            Base.broadcastable(data),
            topology(hspace),
            local_geometry_data(space),
            dss_weights(space),
        )
    else
        dss_transform!(
            device,
            dss_buffer,
            data,
            local_geometry_data(space),
            dss_weights(space),
            perimeter(hspace),
            dss_buffer.internal_elems,
        )
        dss_local!(
            device,
            dss_buffer.perimeter_data,
            perimeter(hspace),
            topology(hspace),
        )
        dss_untransform!(
            device,
            dss_buffer,
            data,
            local_geometry_data(space),
            perimeter(hspace),
            dss_buffer.internal_elems,
        )
    end
    return nothing
end

"""
    weighted_dss_ghost!(data, space, dss_buffer)

1). Finish communications.

2). Call [`Spaces.load_from_recv_buffer!`](@ref)
After the communication is complete, this adds data from the recv buffer to the corresponding location in 
`perimeter_data`. For ghost vertices, this data is added only to the representative vertices. The values are 
then scattered to other local vertices corresponding to each unique ghost vertex in `dss_local_ghost`.

3). Call [`Spaces.dss_untransform!`](@ref) on all local elements.
This transforms the DSS'd local vectors back to Covariant12 vectors, and copies the DSS'd data from the
`perimeter_data` to `data`.
"""
function weighted_dss_ghost!(data, space, dss_buffer)
    isnothing(dss_buffer) && return data
    Quadratures.requires_dss(quadrature_style(space)) || return data
    sizeof(eltype(data)) > 0 || return data
    ClimaComms.finish(dss_buffer.graph_context)
    hspace = horizontal_space(space)
    device = ClimaComms.device(topology(hspace))
    load_from_recv_buffer!(device, dss_buffer)
    dss_ghost!(
        device,
        dss_buffer.perimeter_data,
        perimeter(hspace),
        topology(hspace),
    )
    dss_untransform!(
        device,
        dss_buffer,
        data,
        local_geometry_data(space),
        perimeter(hspace),
        dss_buffer.perimeter_elems,
    )
    return data
end
