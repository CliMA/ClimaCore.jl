"""
    DataMask

Abstract type for masks that mark points in a discretized domain as active or
inactive. Subtypes: [`NoMask`](@ref), [`IJHMask`](@ref) and
[`LevelAlignedOrder`](@ref).
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
[`VIJHF`](@ref) layout as active or inactive. The constructor marks every column as
active; modify `is_active` and call [`set_mask_maps!`](@ref) to change the mask.

# Fields

  - `is_active`: A layout similar to `level(data, 1)` that holds the boolean mask.
  - `N`: A one-element array that holds the total number of active columns.
  - `i_map`: An array that holds the `i`-index of each active column.
  - `j_map`: An array that holds the `j`-index of each active column.
  - `h_map`: An array that holds the `h`-index of each active column.
"""
struct IJHMask{D, A} <: DataMask
    is_active::D
    N::A
    i_map::A
    j_map::A
    h_map::A
end

Adapt.@adapt_structure IJHMask

"""
    is_active(mask::IJHMask)

The boolean layout that marks each column of the masked data as active (`true`) or
inactive (`false`); it has the shape of `level(data, 1)`. Modify it in place and call
[`set_mask_maps!`](@ref) to apply the change.
"""
is_active(mask::IJHMask) = mask.is_active

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

Update `mask.N`, `mask.i_map`, `mask.j_map`, and `mask.h_map` in an
[`IJHMask`](@ref) based on the values in `mask.is_active`, and return `mask`. This
allocates memory when using GPUs, so it should only be called infrequently.
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

Return whether a [`DataMask`](@ref) marks the point at `index` as active.
"""
@propagate_inbounds should_compute(::NoMask, _) = true

# IJHMask supports linear/Cartesian column indices and Cartesian point indices.
@propagate_inbounds should_compute(mask::IJHMask, index::Integer) =
    mask.is_active[index]
@propagate_inbounds should_compute(mask::IJHMask, index::CartesianIndex{3}) =
    mask.is_active[1, index[1], index[2], index[3]]
@propagate_inbounds should_compute(mask::IJHMask, index::CartesianIndex{4}) =
    mask.is_active[1, index[2], index[3], index[4]]

struct ActiveColumnIndices{M, V} <: AbstractVector{CartesianIndex{3}}
    mask::M
    indices::V
end
ActiveColumnIndices(mask) =
    ActiveColumnIndices(mask, Base.OneTo(Int(@inbounds mask.N[1])))
@inline function active_index(inds::ActiveColumnIndices, real_n)
    (; i_map, j_map, h_map) = inds.mask
    @inbounds CartesianIndex(i_map[real_n], j_map[real_n], h_map[real_n])
end
Adapt.@adapt_structure ActiveColumnIndices

# The level count is a type parameter, so that the divrem below strength
# reduces to multiplies and shifts instead of compiling into an integer
# division, which is emulated on GPUs.
struct ActivePointIndices{Nv, M, V} <: AbstractVector{CartesianIndex{4}}
    mask::M
    indices::V
end
ActivePointIndices{Nv}(mask, indices) where {Nv} =
    ActivePointIndices{Nv, typeof(mask), typeof(indices)}(mask, indices)
ActivePointIndices{Nv}(mask) where {Nv} =
    ActivePointIndices{Nv}(mask, Base.OneTo(Nv * Int(@inbounds mask.N[1])))
@inline function active_index(inds::ActivePointIndices{Nv}, real_n) where {Nv}
    (; i_map, j_map, h_map) = inds.mask
    (n_zero, v_zero) = divrem(real_n - one(real_n), Nv % typeof(real_n))
    (v, col) = (v_zero + 1, n_zero + 1)
    @inbounds CartesianIndex(v, i_map[col], j_map[col], h_map[col])
end
Adapt.adapt_structure(to, inds::ActivePointIndices{Nv}) where {Nv} =
    ActivePointIndices{Nv}(Adapt.adapt(to, inds.mask), Adapt.adapt(to, inds.indices))

const ActiveIndices = Union{ActiveColumnIndices, ActivePointIndices}
Base.size(inds::ActiveIndices) = (length(inds.indices),)
Base.@propagate_inbounds Base.getindex(inds::ActiveIndices, n::Int) =
    active_index(inds, inds.indices[n])

"""
    LevelAlignedOrder()

A [`DataMask`](@ref) that marks every point active, as [`NoMask`](@ref) does,
and orders the points of a `VIJFH` or `VIJHF` layout horizontally instead of
vertically: consecutive threads take cells of one level, so a warp covers two
spectral elements at one height rather than 32 heights of one column.

The arithmetic is unchanged and no point is skipped, so a loop under this order
writes the same values as one under `NoMask`. What changes is which points share
a warp, which is what a point body with a per-point early-out is paid in: a
criterion that holds at a height holds across a level-aligned warp and is spread
across 32 vertical ones. The reads become strided rather than contiguous, so this
is only worth asking for on a loop whose body is far from its memory roofline.

Only `view` slices have an order to change; a column of a level-aligned loop is
not contiguous, so [`column`](@ref) slices are rejected.
"""
struct LevelAlignedOrder <: DataMask end

@propagate_inbounds should_compute(::LevelAlignedOrder, _) = true

# The cell grid in iteration order, (Ni, Nj, Nh, Nv), presented as the
# (v, i, j, h) indices the layout is indexed by. `indices` is the subset of
# positions the calling subscope was given, as in ActiveIndices.
struct LevelAlignedIndices{D, V} <: AbstractVector{CartesianIndex{4}}
    dims::D
    indices::V
end
Adapt.@adapt_structure LevelAlignedIndices

@noinline throw_unorderable_size(size) = throw(
    ArgumentError(
        "LevelAlignedOrder needs a (v, i, j, h) point layout, got a size of $size",
    ),
)

@inline function LevelAlignedIndices(arg)
    dims = size(arg)
    length(dims) == 4 || throw_unorderable_size(dims)
    (Nv, Ni, Nj, Nh) = dims
    return LevelAlignedIndices((Ni, Nj, Nh, Nv), Base.OneTo(Nv * Ni * Nj * Nh))
end

Base.size(inds::LevelAlignedIndices) = (length(inds.indices),)
Base.@propagate_inbounds function Base.getindex(inds::LevelAlignedIndices, n::Int)
    index = cartesian_index_from_linear(inds.dims, inds.indices[n])
    return CartesianIndex(index[4], index[1], index[2], index[3])
end
