"""
    DataMask

Marks points in a discretized domain as active or inactive.
"""
abstract type DataMask end

"""
    NoMask()

A [`DataMask`](@ref) that marks every point in a discretized domain as active.
"""
struct NoMask <: DataMask end

"""
    IJHMask(data)

A [`DataMask`](@ref) that marks the columns of a [`VIJFH`](@ref) or
[`VIJHF`](@ref) layout as active or inactive, using the following cached values:
 - `is_active`, a layout similar to `level(data, 1)` representing a boolean mask
 - `N`, an array that contains the total number of active columns
 - `i_map`, an array that contains the `i`-index of each active column
 - `j_map`, an array that contains the `j`-index of each active column
 - `h_map`, an array that contains the `h`-index of each active column
"""
struct IJHMask{D, A} <: DataMask
    is_active::D
    N::A
    i_map::A
    j_map::A
    h_map::A
end

Adapt.@adapt_structure IJHMask

function IJHMask(data::VIJHWithF)
    is_active = map(Returns(true), level(data, 1))
    N = similar(parent(data), Int, 1)
    i_map = similar(parent(data), Int, length(is_active))
    mask = IJHMask(is_active, N, i_map, similar(i_map), similar(i_map))
    set_mask_maps!(mask)
    return mask
end

"""
    set_mask_maps!(mask)

Update the maps in an [`IJHMask`](@ref) based on the values in `mask.is_active`.
This allocates memory when using GPUs, so it should only be called infrequently.
"""
function set_mask_maps!(mask::IJHMask)
    using_arrays = parent(mask.is_active) isa Array
    is_active = using_arrays ? mask.is_active : rebuild(mask.is_active, Array)
    i_map = using_arrays ? mask.i_map : Array(mask.i_map)
    j_map = using_arrays ? mask.j_map : Array(mask.j_map)
    h_map = using_arrays ? mask.h_map : Array(mask.h_map)
    n = 1
    @inbounds for index in CartesianIndices(is_active)
        is_active[index] || continue
        i_map[n] = index[2]
        j_map[n] = index[3]
        h_map[n] = index[4]
        n += 1
    end
    fill!(mask.N, n - 1)
    if !using_arrays
        copyto!(mask.i_map, i_map)
        copyto!(mask.j_map, j_map)
        copyto!(mask.h_map, h_map)
    end
    return mask
end

"""
    should_compute(mask, index)

Check whether a [`DataMask`](@ref) marks the point at some index as active.
"""
@propagate_inbounds should_compute(::NoMask, _) = true

# IJHMask supports linear/Cartesian column indices and Cartesian point indices.
@propagate_inbounds should_compute(mask::IJHMask, index::Integer) =
    mask.is_active[index]
@propagate_inbounds should_compute(mask::IJHMask, index::CartesianIndex{3}) =
    mask.is_active[1, index[1], index[2], index[3]]
@propagate_inbounds should_compute(mask::IJHMask, index::CartesianIndex{4}) =
    mask.is_active[1, index[2], index[3], index[4]]
