const IndexableData = Union{DataLayout, MaybeFusedDataLayoutBroadcast}
const PointIndex = Union{Integer, CartesianIndex}

# Allow linear indexing if parent(data)[1:length(data)] has one value per point.
Base.IndexStyle(::Type{D}) where {D <: DataLayout} =
    ndims(D) <= 1 ? IndexLinear() :
    ncomponents(D) <= 1 || unrolled_all(==(1), inferred_size(D)[f_dim(D):end]) ?
    IndexStyle(parent_type(D)) : IndexCartesian()

Base.IndexStyle(bc::MaybeFusedDataLayoutBroadcast) = IndexStyle(layout_args(bc)...)

# Allow linear indexing if all DataLayouts in an expression have the same shape.
# Add DataLayout-only methods to avoid ambiguities with AbstractArray methods.
for T in (:IndexableData, :DataLayout)
    @eval function Base.IndexStyle(arg1::$T, arg2::$T, args::$T...)
        non_point_args = unrolled_filter(!iszero ∘ ndims, (arg1, arg2, args...))
        unrolled_allequal(layout_type, non_point_args) || return IndexCartesian()
        unrolled_allequal(shape_params, non_point_args) || return IndexCartesian()
        return unrolled_mapreduce(IndexStyle, IndexStyle, (arg1, arg2, args...))
    end

    @eval Base.eachindex(arg::$T, args::$T...) =
        eachindex(IndexStyle(arg, args...), arg, args...)
    @eval Base.eachindex(::IndexLinear, arg::$T, args::$T...) =
        unrolled_allequal(length, (arg, args...)) ? Base.OneTo(length(arg)) :
        throw(DimensionMismatch("Inputs to eachindex must have the same length"))
    @eval Base.eachindex(::IndexCartesian, arg::$T, args::$T...) =
        unrolled_allequal(size, (arg, args...)) ? CartesianIndices(size(arg)) :
        throw(DimensionMismatch("Inputs to eachindex must have the same size"))
end

"""
    each_slice_index(op, args...)

Generalization of `eachindex` for the slice operators [`level`](@ref),
[`slab`](@ref), [`column`](@ref), and `view` (for creating single-point slices).
The result is always an iterator of Cartesian indices, whose scalar offsets are
simple enough for SIMD optimization (a `view` at a linear index wraps its parent
in a 1-dimensional `ReshapedArray`, which blocks SIMD in pointwise loops).
"""
@inline each_slice_index(op::O, args...) where {O} =
    unrolled_allequal(Base.Fix1(each_slice_index, op), args) ?
    each_slice_index(op, first(args)) :
    throw(DimensionMismatch("Inputs to each_slice_index must have consistent dimensions"))

@inline each_slice_index(::typeof(view), arg) = CartesianIndices(size(arg))
@inline each_slice_index(::typeof(level), arg) = CartesianIndices((nlevels(arg),))
@inline each_slice_index(::typeof(slab), arg) =
    CartesianIndices((nlevels(arg), nelems(arg)))
@inline each_slice_index(::typeof(column), arg) =
    CartesianIndices((vijh_params(arg).Ni, vijh_params(arg).Nj, nelems(arg)))

# Preserve linear indices into broadcast arguments: Base's newindex fallback
# reinterprets an integer as a CartesianIndex along the first dimension, which
# is incorrect for multidimensional arguments. IndexStyle only permits linear
# indices when all nonzero-dimensional layouts share a shape, so only
# 0-dimensional data needs conversion (its single point read by every index).
@inline Broadcast.newindex(arg::IndexableData, index::Integer) =
    iszero(ndims(arg)) ? CartesianIndex() : index

# Override checkbounds for LazyDataLayouts to prevent unnecessary BoundsErrors.
@inline Base.checkbounds(bc::LazyDataLayout, index::Integer) =
    1 <= index <= length(bc) || Base.throw_boundserror(bc, (index,))
@inline Base.checkbounds(bc::LazyDataLayout, ::CartesianIndex{0}) = checkbounds(bc, 1)

# Like single-point broadcasts, single-point layouts (e.g. level slices with one
# level) are identified by their length, since they keep their dimensions.
@inline is_single_point(data, index) = isone(length(data)) && index == CartesianIndex()

# Avoid unnecessary indexing arithmetic whenever possible. Base's default array
# access methods use Cartesian-to-linear index conversions, without any constant
# propagation of array dimensions. Even worse, Base's default SubArray access
# methods use linear-to-Cartesian index conversions, calling div/rem at runtime.
# This function ensures that Cartesian-to-linear conversion is constant-folded,
# and it uses a linear index to access the parent of a constant-stride SubArray.
@propagate_inbounds function array_and_index_args(data, index)
    array = parent(data)
    is_single_point(data, index) && return (array, ())
    F = f_dim(data)
    (isnothing(F) || !has_inferred_size(data)) && return (array, (index, Val(F)))
    dims = inferred_size(data)
    stride = prod(dims[1:(F - 1)])
    if IndexStyle(data) == IndexLinear()
        array_index = index isa Integer ? index : linear_index(dims, Tuple(index))
        return (array, (array_index, stride))
    elseif unrolled_all(==(1), dims[F:end]) && is_constant_stride_view(array, Val(F))
        parent_array_offset = (first(parentindices(array)[F]) - 1) * stride
        array_index = linear_index(dims[1:(F - 1)], Tuple(index)[1:(F - 1)])
        return (parent(array), (parent_array_offset + array_index, stride))
    end
    return (array, (index, Val(F)))
end

# Constant-folded Cartesian-to-linear index conversion for array of size `dims`.
@inline function linear_index(dims, indices)
    dim_index_pairs = unrolled_map(tuple, dims, indices)
    (offset, _) =
        unrolled_reduce(dim_index_pairs; init = (0, 1)) do (offset, stride), (dim, index)
            (offset + (index - 1) * stride, stride * dim)
        end
    return offset + 1
end

# Check for a SubArray built by struct_field_view whose parent is IndexLinear
# and whose indices are full Slice ranges except for a UnitRange at position F.
# Such an array can be accessed using constant-stride linear indices, bypassing
# Base's inefficient linear-to-Cartesian index conversion.
@inline is_constant_stride_view(_, _) = false
@inline is_constant_stride_view(array::SubArray, ::Val{F}) where {F} =
    IndexStyle(parent(array)) == IndexLinear() &&
    unrolled_all(Base.Fix2(isa, Base.Slice), parentindices(array)[1:(F - 1)]) &&
    unrolled_all(Base.Fix2(isa, Base.Slice), parentindices(array)[(F + 1):end])

@propagate_inbounds safe_index(data, index) =
    IndexStyle(data) == IndexCartesian() && index isa Integer ?
    CartesianIndices(data)[index] : index

# Always convert to the element type of a DataLayout when modifying its values.
@propagate_inbounds function Base.setindex!(data::DataLayout, value, index::PointIndex)
    (array, index_args) = array_and_index_args(data, safe_index(data, index))
    return set_struct!(array, convert(eltype(data), value), index_args...)
end

@propagate_inbounds function Base.getindex(data::DataLayout, index::PointIndex)
    (array, index_args) = array_and_index_args(data, safe_index(data, index))
    return get_struct(array, eltype(data), index_args...)
end

# Represent every single-point DataLayout view using a zero-dimensional DataF.
@propagate_inbounds Base.view(data::DataLayout, index::PointIndex) =
    is_single_point(data, index) ? data :
    DataF{eltype(data), typeof(DataScope(data))}(
        view_struct(parent(data), eltype(data), safe_index(data, index), Val(f_dim(data))),
    )

# Use Broadcast.newindex to match the behavior of getindex for LazyDataLayouts.
@propagate_inbounds Base.view(bc::MaybeFusedDataLayoutBroadcast, index::PointIndex) =
    modify_args(bc) do arg
        Base.@_propagate_inbounds_meta
        view(arg, Broadcast.newindex(arg, index))
    end

# A single-point slice of multidimensional data keeps its number of dimensions,
# so single-point broadcasts are identified by their length instead of ndims.
@inline Base.view(bc::MaybeFusedDataLayoutBroadcast, ::CartesianIndex{0}) =
    isone(length(bc)) ? bc : Base.throw_boundserror(bc, (CartesianIndex(),))

@propagate_inbounds Base.setindex!(data::DataLayout, value, indices::PointIndex...) =
    setindex!(data, value, CartesianIndex(indices...))
@propagate_inbounds Base.getindex(data::DataLayout, indices::PointIndex...) =
    getindex(data, CartesianIndex(indices...))
@propagate_inbounds Base.view(arg::IndexableData, indices::PointIndex...) =
    view(arg, CartesianIndex(indices...))

# Reduce latency by only constructing slice views when necessary.
@propagate_inbounds function level(arg::IndexableData, v)
    (; Nv) = vijh_params(arg)
    Nv == 1 || return level_view(arg, v)
    @boundscheck v == 1 || throw(ArgumentError("DataLayout has only one level"))
    return arg
end
@propagate_inbounds function slab(arg::IndexableData, v, h)
    (; Nv, Nh) = vijh_params(arg)
    Nv == Nh == 1 || return slab_view(arg, v, h)
    @boundscheck v == h == 1 || throw(ArgumentError("DataLayout has only one slab"))
    return arg
end
@propagate_inbounds function column(arg::IndexableData, i, j, h)
    (; Ni, Nj, Nh) = vijh_params(arg)
    Ni == Nj == Nh == 1 || return column_view(arg, i, j, h)
    @boundscheck i == j == h == 1 || throw(ArgumentError("DataLayout has only one column"))
    return arg
end

# Convenience methods for data with a single vertical level or a single
# horizontal dimension, matching the corresponding methods for spaces.
@propagate_inbounds slab(arg::IndexableData, h) = slab(arg, 1, h)
@propagate_inbounds column(arg::IndexableData, i, h) = column(arg, i, 1, h)
