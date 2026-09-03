import ..Topologies:
    create_dss_buffer,
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

Create a [`Topologies.DSSBuffer`](@ref) for the field data `data` on `space`.
Return `nothing` if `space` is discontinuous (DG) or has no second horizontal
dimension (1D spectral element spaces use `dss_1d!` and need no buffer).
"""
create_dss_buffer(data::DataLayouts.VIJHWithF, space) =
    (!is_continuous(space) || isone(size(data, 3))) ? nothing :
    create_dss_buffer(
        data,
        topology(space),
        local_geometry_data(space),
        dss_weights(space),
    )

"""
    weighted_dss!(data, space, dss_buffer)

Compute the weighted direct stiffness summation (DSS) of `data` in place: values at
nodes shared between elements are replaced by their weighted average. On a
discontinuous (DG) space this is a no-op.

It consists of the following steps:

 1. [`Spaces.weighted_dss_start!`](@ref),
 2. [`Spaces.weighted_dss_internal!`](@ref),
 3. [`Spaces.weighted_dss_ghost!`](@ref).
"""
function weighted_dss!(data::DataLayouts.VIJHWithF, space, dss_buffer)
    weighted_dss_start!(data, space, dss_buffer)
    weighted_dss_internal!(data, space, dss_buffer)
    weighted_dss_ghost!(data, space, dss_buffer)
    call_post_op_callback() && post_op_callback(data, data, space, dss_buffer)
end

function weighted_dss_prepare!(data, space, dss_buffer)
    isnothing(dss_buffer) && return nothing
    is_continuous(space) || return nothing
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

Start the weighted DSS of `data`: prepare the perimeter data of the elements on the
process boundary and begin communication with neighboring processes. Returns
`nothing`.

It consists of the following steps:

 1. Apply [`Spaces.dss_transform!`](@ref) on perimeter elements. This weights, and
    transforms to the physical basis if needed, the vector fields; scalar fields are
    weighted. The result is stored in `dss_buffer.perimeter_data`.
 2. Apply [`Spaces.dss_local_ghost!`](@ref), which computes the partial weighted DSS
    on ghost vertices using only the information from local vertices.
 3. Apply [`Spaces.fill_send_buffer!`](@ref), which loads the send buffer from
    `perimeter_data`. For unique ghost vertices, only the representative vertices,
    which hold the result of the "ghost local" DSS, are loaded.
 4. Start the DSS communication with neighboring processes.
"""
function weighted_dss_start!(data, space, dss_buffer)
    isnothing(dss_buffer) && return nothing
    is_continuous(space) || return nothing
    sizeof(eltype(data)) > 0 || return nothing
    device = ClimaComms.device(topology(space))
    weighted_dss_prepare!(data, space, dss_buffer)
    cuda_synchronize(device; blocking = true)
    ClimaComms.start(dss_buffer.graph_context)
    return nothing
end

"""
    weighted_dss_internal!(data, space, dss_buffer)

Perform the part of the weighted DSS of `data` that needs no communication, while
the communication started by [`Spaces.weighted_dss_start!`](@ref) is in flight.
Returns `nothing`.

It consists of the following steps:

 1. Apply [`Spaces.dss_transform!`](@ref) on interior elements. Local elements are
    split into interior and perimeter elements so that communication overlaps with
    computation.
 2. Apply [`Spaces.dss_local!`](@ref), which computes the weighted DSS on local
    vertices and faces.
 3. Apply [`Spaces.dss_untransform!`](@ref) on interior elements.

On a 1D spectral element space, the whole DSS is performed here with `dss_1d!`.
"""
function weighted_dss_internal!(data, space, dss_buffer)
    is_continuous(space) || return nothing
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

Finish the weighted DSS of `data` started by [`Spaces.weighted_dss_start!`](@ref).
Returns `data`.

It consists of the following steps:

 1. Finish the communication.
 2. Apply [`Spaces.load_from_recv_buffer!`](@ref), which adds the data in the
    receive buffer to the corresponding locations in `perimeter_data`. For ghost
    vertices, the data is added only to the representative vertices; the values
    are then scattered to the other local vertices of each unique ghost vertex by
    `dss_ghost!`.
 3. Apply [`Spaces.dss_untransform!`](@ref) on perimeter elements, which transforms
    the summed vectors back to their original basis and copies the summed data from
    `perimeter_data` to `data`.
"""
function weighted_dss_ghost!(data, space, dss_buffer)
    isnothing(dss_buffer) && return data
    is_continuous(space) || return data
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
