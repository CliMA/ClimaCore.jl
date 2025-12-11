"""
    ClimaCoreConservativeRegriddingExt

This extension provides a set of functions for performing conservative regridding
to and/or from ClimaCore spaces. It uses the `ConservativeRegridding.jl` package,
which must be loaded to trigger this extension.
"""
module ClimaCoreConservativeRegriddingExt

using ConservativeRegridding, ClimaCore
import ConservativeRegridding: Regridder, regrid!
import ClimaCore: Spaces, Meshes, Quadratures, Fields, RecursiveApply, Spaces, Remapping

"""
    get_element_vertices(space::SpectralElementSpace2D)

Returns a vector of vectors, each containing the coordinates of the vertices
of an element. The vertices are in clockwise order for each element, and the
first coordinate pair is repeated at the end.

Also performs a check for zero area polygons, and throws an error if any are found.

This is the format expected by ConservativeRegridding.jl to construct a
Regridder object.
"""
function Remapping.get_element_vertices(space)
    # Get the indices of the vertices of the elements, in clockwise order for each element
    Nh = Meshes.nelements(space.grid.topology.mesh)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    vertex_inds = [
        CartesianIndex(i, j, 1, 1, e) # f and v are 1 for SpectralElementSpace2D
        for e in 1:Nh
        for (i, j) in [(1, 1), (1, Nq), (Nq, Nq), (Nq, 1), (1, 1)]
    ] # repeat the first coordinate pair at the end

    # Get the lat and lon at each vertex index
    coords = Fields.coordinate_field(space)
    vertex_coords = [
        (Fields.field_values(coords.lat)[ind], Fields.field_values(coords.long)[ind])
        for ind in vertex_inds
    ]

    # Put each polygon into a vector, with the first coordinate pair repeated at the end
    vertices = collect(Iterators.partition(vertex_coords, 5))

    # Check for zero area polygons (all latitude or longitude values are the same)
    for polygon in vertices
        if allequal(first.(polygon)) || allequal(last.(polygon))
            @error "Zero area polygon found in vertices" polygon
        end
    end
    return vertices
end

### These functions are used to facilitate storing a single value per element on a field
### rather than one value per node. Note that this will not be our long-term solution.
"""
    integrate_each_element(field)

Integrate the field over each element of the space.
Returns a vector with length equal to the number of elements in the space,
containing the integrated value over the nodes of each element.
"""
function Remapping.integrate_each_element(field)
    space = axes(field)
    weighted_values =
        RecursiveApply.rmul.(
            Spaces.weighted_jacobian(space),
            Fields.todata(field),
        )

    Nh = Meshes.nelements(space.grid.topology.mesh)
    integral_each_element = zeros(Float64, Nh)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    for e in 1:Nh # loop over each element
        for i in 1:Nq
            for j in 1:Nq
                integral_each_element[e] += weighted_values[CartesianIndex(i, j, 1, 1, e)]
            end
        end
    end
    return integral_each_element
end

"""
    get_value_per_element!(value_per_element, field, ones_field)

Get one value per element of a field by integrating over the nodes of
each element and dividing by the area of the element. The result is stored in
`value_per_element`, which is expected to be a Float-valued vector of length equal
to the number of elements in the space.

Here `ones_field` is a field on the same space as `field` with all
values set to 1.
"""
function Remapping.get_value_per_element!(
    value_per_element,
    field,
    ones_field,
)
    integral_each_element = Remapping.integrate_each_element(field)
    area_each_element = Remapping.integrate_each_element(ones_field)
    value_per_element .= integral_each_element ./ area_each_element
    return nothing
end

"""
    set_value_per_element!(field, value_per_element)

Set the values of a field with the provided values in each element.
Each node within an element will have the same value.

The input vector is expected to be of length equal to the number of elements in
the space.
"""
function Remapping.set_value_per_element!(field, value_per_element)
    space = axes(field)
    Nh = Meshes.nelements(space.grid.topology.mesh)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))

    @assert length(value_per_element) == Nh "Length of value_per_element must be equal to the number of elements in the space"

    # Set the value in each node of each element to the value per element
    for e in 1:Nh
        for i in 1:Nq
            for j in 1:Nq
                Fields.field_values(field)[CartesianIndex(i, j, 1, 1, e)] =
                    value_per_element[e]
            end
        end
    end
    return field
end

"""
    Regridder(dst_space, src_space; kwargs...)

Create a regridder between two ClimaCore Spaces.
This works by finding the vertices of the elements of the source and
destination spaces, and then computing the areas of their intersections.

This is currently only defined for 2D spaces, but could be extended to
3D spaces.

This function is intended to be used with the finite volume approximation
of the ClimaCore spectral element space.
"""
function ConservativeRegridding.Regridder(
    dst_space::Spaces.SpectralElementSpace2D,
    src_space::Spaces.SpectralElementSpace2D;
    kwargs...,
)
    dst_vertices = Remapping.get_element_vertices(dst_space)
    src_vertices = Remapping.get_element_vertices(src_space)
    return ConservativeRegridding.Regridder(dst_vertices, src_vertices; kwargs...)
end
ConservativeRegridding.Regridder(
    dst_field::Fields.Field,
    src_field::Fields.Field;
    kwargs...,
) = ConservativeRegridding.Regridder(axes(dst_field), axes(src_field); kwargs...)

"""
    regrid!(dst_field, regridder_tuple, src_field)

Perform conservative regridding from `src_field` to `dst_field` using a
Regridder object and pre-allocated buffers.

The `regridder_tuple` should be a NamedTuple containing:
- `regridder`: The ConservativeRegridding.Regridder object
- `value_per_element_src`: Pre-allocated buffer for source values
- `value_per_element_dst`: Pre-allocated buffer for destination values
- `ones_src`: Pre-allocated field of ones on the source space
```
"""
function ConservativeRegridding.regrid!(
    dst_field::Fields.Field,
    regridder_tuple::NamedTuple,
    src_field::Fields.Field,
)
    @assert eltype(dst_field) isa Number && eltype(src_field) isa Number "Regridding is only supported for scalar fields"
    @assert eltype(dst_field) == eltype(src_field) "Source and destination fields must have the same element type"

    # Use pre-allocated buffers from the regridder tuple
    (; value_per_element_src, value_per_element_dst, ones_src, regridder) = regridder_tuple

    # Get one value per element in the source field, equal to the quadrature-weighted average of the
    # values at nodes of the element
    Remapping.get_value_per_element!(value_per_element_src, src_field, ones_src)

    # Perform the regridding
    ConservativeRegridding.regrid!(value_per_element_dst, regridder, value_per_element_src)

    # Now that we have our regridded vector, put it onto a field on the second space
    Remapping.set_value_per_element!(dst_field, value_per_element_dst)
    return nothing
end

"""
    regrid!(dst_field, regridder, src_field)

Perform conservative regridding from `src_field` to `dst_field` using a
Regridder object.

This is a convenience function that allocates the buffers for you.
Note that this is not efficient for repeated regridding with the same regridder,
but it may be helpful for one-off regriddings or testing/debugging.
"""
function ConservativeRegridding.regrid!(
    dst_field::Fields.Field,
    regridder::ConservativeRegridding.Regridder,
    src_field::Fields.Field,
)
    # Allocate space for the buffers
    value_per_element_src =
        zeros(Float64, Meshes.nelements(axes(src_field).grid.topology.mesh))
    value_per_element_dst =
        zeros(Float64, Meshes.nelements(axes(dst_field).grid.topology.mesh))
    ones_src = ones(axes(src_field))
    regridder_tuple = (;
        regridder,
        value_per_element_src,
        value_per_element_dst,
        ones_src,
    )

    # Perform the regridding
    ConservativeRegridding.regrid!(dst_field, regridder_tuple, src_field)
    return nothing
end

end
