"""
    vertical_indices(space, zcoords)

Return the vertical index of the element that contains `zcoords`.

`zcoords` is interpreted as "reference z coordinates".
"""
function vertical_indices(space, zcoords)
    vert_topology = Spaces.vertical_topology(space)
    vert_mesh = vert_topology.mesh
    return Meshes.containing_element.(Ref(vert_mesh), zcoords)
end

"""
    vertical_reference_coordinates(space, zcoords)

Return the reference coordinates of the element that contains `zcoords`.

Reference coordinates (ξ) typically go from -1 to 1, but for center spaces this function
remaps in such a way that they can directly be used for linear interpolation (so, if ξ is
negative, it is remapped to (0,1), if positive to (-1, 0)). This is best used alongside with
`vertical_bounding_indices`.

`zcoords` is interpreted as "reference z coordinates".
"""
function vertical_reference_coordinates(space, zcoords)
    vert_topology = Spaces.vertical_topology(space)
    vert_mesh = vert_topology.mesh

    is_cell_center =
        space isa Spaces.CenterExtrudedFiniteDifferenceSpace ||
        space isa Spaces.CenterFiniteDifferenceSpace

    ξ3s = map(zcoords) do zcoord
        velem = Meshes.containing_element(vert_mesh, zcoord)
        ξ3, = Meshes.reference_coordinates(vert_mesh, velem, zcoord)
        # For cell centered spaces, shift ξ3 so that we can use it for linear interpolation
        is_cell_center && (ξ3 = ξ3 < 0 ? ξ3 + 1 : ξ3 - 1)
        return ξ3
    end

    return ξ3s
end


"""
    vertical_bounding_indices(space, zcoords)

Return the vertical element indices needed to perform linear interpolation of `zcoords`.

For centered-valued fields, if `zcoord` is in the top (bottom) half of a top (bottom)
element in a column, no interpolation is performed and the value at the cell center is
returned. Effectively, this means that the interpolation is first-order accurate across the
column, but zeroth-order accurate close to the boundaries.
"""
function vertical_bounding_indices end

function vertical_bounding_indices(
    space::Union{
        Spaces.FaceExtrudedFiniteDifferenceSpace,
        Spaces.FaceFiniteDifferenceSpace,
    },
    zcoords,
)
    vert_topology = Spaces.vertical_topology(space)
    vert_mesh = vert_topology.mesh
    velems = Meshes.containing_element.(Ref(vert_mesh), zcoords)
    return map(v -> (v, v + 1), velems)
end

function vertical_bounding_indices(
    space::Union{
        Spaces.CenterExtrudedFiniteDifferenceSpace,
        Spaces.CenterFiniteDifferenceSpace,
    },
    zcoords,
)
    vert_topology = Spaces.vertical_topology(space)
    vert_mesh = vert_topology.mesh
    Nz = Spaces.nlevels(space)

    is_periodic = Topologies.isperiodic(vert_topology)

    vert_indices = map(zcoords) do zcoord
        velem = Meshes.containing_element(vert_mesh, zcoord)
        ξ3, = Meshes.reference_coordinates(vert_mesh, velem, zcoord)
        if ξ3 < 0
            v_lo = is_periodic ? mod1(velem - 1, Nz) : max(velem - 1, 1)
            v_hi = velem
        else
            v_lo = velem
            v_hi = is_periodic ? mod1(velem + 1, Nz) : min(velem + 1, Nz)
        end
        return v_lo, v_hi
    end

    return vert_indices
end


"""
    vertical_interpolation_weights(space, zcoords)

Compute the interpolation weights to vertically interpolate the `zcoords` in the given `space`.

This assumes a linear interpolation, where the first weight is to be multiplied with the "lower"
element, and the second weight with the "higher" element in a stack.

That is, this function returns `A`, `B` such that `f(zcoord) = A f_lo + B f_hi`, where `f_lo` and
`f_hi` are the values on the neighboring elements of `zcoord`.
"""
function vertical_interpolation_weights(space, zcoords)
    ξs = vertical_reference_coordinates(space, zcoords)
    return map(ξ -> ((1 - ξ) / 2, (1 + ξ) / 2), ξs)
end
