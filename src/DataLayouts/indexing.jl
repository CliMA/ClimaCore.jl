const IndexableData = Union{DataLayout, MaybeFusedDataLayoutBroadcast}
const PointIndex = Union{Integer, CartesianIndex}

# Check for a SubArray built by struct_field_view whose parent is IndexLinear
# and whose indices are full Slice ranges except at position F. The index at F
# must select contiguous entries (a unit range), since the constant-stride
# accessors read consecutive parent fields; a StepRange there would silently
# read the wrong entries.
@inline is_constant_stride_view_type(::Type{A}, ::Val{F}) where {A, F} = false
@inline function is_constant_stride_view_type(
    ::Type{<:SubArray{<:Any, <:Any, P, I}},
    ::Val{F},
) where {P, I, F}
    isnothing(F) && return false
    Base.IndexStyle(P) == Base.IndexLinear() || return false
    params = fieldtypes(I)
    (1 <= F <= length(params)) || return false
    params[F] <: AbstractUnitRange || return false
    return unrolled_all(
        ((i, param),) -> i == F || param <: Base.Slice,
        unrolled_map(tuple, ntuple(identity, Val(length(params))), params),
    )
end

# Allow linear indexing if parent(data)[1:length(data)] has one value per point.
Base.IndexStyle(::Type{D}) where {D <: DataLayout} =
    ndims(D) <= 1 ? IndexLinear() :
    (is_constant_stride_view_type(parent_type(D), Val(f_dim(D))) && ncomponents(D) <= 1) ?
    IndexLinear() :
    ncomponents(D) <= 1 || unrolled_all(==(1), inferred_size(D)[f_dim(D):end]) ?
    IndexStyle(parent_type(D)) : IndexCartesian()

@inline is_non_point_arg(arg) = !iszero(ndims(arg))

# Whether all non-point layouts share a shape, so that a linear point index
# denotes the same point in every one of them.
# The non-point args are collected into a tuple with unrolled_flatmap, which
# keeps every intermediate type concrete. Folding the checks into a reduction
# over a reference arg would instead accumulate a union over the distinct
# layout types among the args, which exceeds inference's union-splitting
# limits for heterogeneous broadcast expressions and makes the shape checks
# dynamic.
# Layouts are collected recursively, so that a singleton layout nested inside
# a broadcast expression is seen by the shape checks: a nested broadcast's own
# merged shape_params would hide it, and a linear index does not project onto
# singleton dimensions, so reading such a layout at a linear index would go
# out of range.
@inline get_non_point_arg_tuple(arg) = is_non_point_arg(arg) ? (arg,) : ()
@inline get_non_point_arg_tuple(bc::MaybeFusedDataLayoutBroadcast) =
    unrolled_flatmap(get_non_point_arg_tuple, layout_args(bc))

@inline function equal_layout_shapes(args)
    non_point_args = unrolled_flatmap(get_non_point_arg_tuple, args)
    return unrolled_allequal(layout_type, non_point_args) &&
           unrolled_allequal(shape_params, non_point_args)
end

# A broadcast expression permits linear point indices whenever its non-point
# layouts share a shape: each argument then converts a linear index as its own
# layout requires. Layouts with different shapes require Cartesian indices,
# which Broadcast.newindex projects onto singleton dimensions.
@inline Base.IndexStyle(bc::MaybeFusedDataLayoutBroadcast) =
    equal_layout_shapes(layout_args(bc)) ? IndexLinear() : IndexCartesian()

# Allow linear indexing if all DataLayouts in an expression have the same shape.
# Add DataLayout-only methods to avoid ambiguities with AbstractArray methods.
for T in (:IndexableData, :DataLayout)
    @eval @inline Base.IndexStyle(arg1::$T, arg2::$T, args::$T...) =
        equal_layout_shapes((arg1, arg2, args...)) ?
        unrolled_mapreduce(IndexStyle, IndexStyle, (arg1, arg2, args...)) : IndexCartesian()

    @eval @inline Base.eachindex(arg::$T, args::$T...) =
        eachindex(IndexStyle(arg, args...), arg, args...)
    @eval @inline Base.eachindex(::IndexLinear, arg::$T, args::$T...) =
        unrolled_allequal(length, (arg, args...)) ? Base.OneTo(length(arg)) :
        throw(DimensionMismatch("Inputs to eachindex must have the same length"))
    @eval @inline Base.eachindex(::IndexCartesian, arg::$T, args::$T...) =
        unrolled_allequal(size, (arg, args...)) ? CartesianIndices(size(arg)) :
        throw(DimensionMismatch("Inputs to eachindex must have the same size"))
end

"""
    each_slice_index(op, arg)

Generalization of `eachindex` for the slice operators [`level`](@ref),
[`slab`](@ref), [`column`](@ref), and `view` (for creating single-point slices).
The result is always an iterator of Cartesian indices, whose scalar offsets are
simple enough for SIMD optimization (a `view` at a linear index wraps its parent
in a 1-dimensional `ReshapedArray`, which blocks SIMD in pointwise loops).
"""
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
# CartesianIndex arguments keep Base's newindex, which projects them onto
# singleton dimensions.
@inline Broadcast.newindex(arg::IndexableData, index::Integer) =
    iszero(ndims(arg)) ? CartesianIndex() : index

# Override checkbounds for LazyDataLayouts to prevent unnecessary BoundsErrors.
@inline Base.checkbounds(bc::LazyDataLayout, index::Integer) =
    1 <= index <= length(bc) || Base.throw_boundserror(bc, (index,))
@inline Base.checkbounds(bc::LazyDataLayout, ::CartesianIndex{0}) = checkbounds(bc, 1)

# Avoid unnecessary indexing arithmetic whenever possible. Base's default array
# access methods use Cartesian-to-linear index conversions, without any constant
# propagation of array dimensions. Even worse, Base's default SubArray access
# methods use linear-to-Cartesian index conversions, calling div/rem at runtime.
# This function ensures that Cartesian-to-linear conversion is constant-folded,
# and it uses a linear index to access the parent of a constant-stride SubArray.
@inline field_offset(array::SubArray, ::Val{F}) where {F} =
    first(parentindices(array)[F]) - 1

# Constant-folded Cartesian-to-linear index conversion for array of size `dims`.
# The init value is passed positionally instead of as a keyword argument
# because kwcalls of unrolled_reduce do not always specialize during GPU
# compilation of wide broadcast expressions, which makes them dynamic.
@inline function linear_index(dims, indices)
    dim_index_pairs = unrolled_map(tuple, dims, indices)
    (offset, _) =
        unrolled_reduce(dim_index_pairs, (0, 1)) do (offset, stride), (dim, index)
            (offset + (index - 1) * stride, stride * dim)
        end
    return offset + 1
end

# Propagate all Cartesian indices that are specified by the user, without any
# Cartesian-to-linear conversion. Avoid linear-to-Cartesian conversion unless it
# is necessary, using a single integer division to access any constant-stride
# parent array. Avoid the need to reshape single-point views, converting empty
# Cartesian indices into full-rank indices for multidimensional parent arrays.
@propagate_inbounds function array_and_index_args(data, index)
    array = parent(data)
    F = f_dim(data)
    (index == CartesianIndex() && isone(length(data))) &&
        return (array, isone(ndims(array)) ? () : (first(CartesianIndices(data)), Val(F)))
    (index isa CartesianIndex || isnothing(F)) && return (array, (index, Val(F)))
    IndexStyle(data) == IndexCartesian() &&
        return (array, (CartesianIndices(data)[index], Val(F)))
    stride = prod(size(data)[1:(F - 1)])
    IndexStyle(array) == IndexLinear() && return (array, (index, stride))
    index_for_dims_after_F, offset_for_dims_before_F = divrem(index - 1, stride)
    parent_Nf = size(parent(array), F)
    parent_f = field_offset(array, Val(F))
    num_strides_in_parent = index_for_dims_after_F * parent_Nf + parent_f
    parent_index = num_strides_in_parent * stride + offset_for_dims_before_F + 1
    return (parent(array), (parent_index, stride))
end

# Always convert to the element type of a DataLayout when modifying its values.
@propagate_inbounds function Base.setindex!(data::DataLayout, value, index::PointIndex)
    (array, index_args) = array_and_index_args(data, index)
    return set_struct!(array, convert(eltype(data), value), index_args...)
end

@propagate_inbounds function Base.getindex(data::DataLayout, index::PointIndex)
    (array, index_args) = array_and_index_args(data, index)
    return get_struct(array, eltype(data), index_args...)
end

# Represent every single-point DataLayout view using a zero-dimensional DataF.
@propagate_inbounds function Base.view(data::DataLayout, index::PointIndex)
    (array, index_args) = array_and_index_args(data, index)
    return DataF{eltype(data), typeof(DataScope(data))}(
        view_struct(array, eltype(data), index_args...),
    )
end

# Use Broadcast.newindex to match the behavior of getindex for LazyDataLayouts.
#
# The argument loop is expanded by a generated function rather than routed
# through modify_args, whose two extra closure layers cost 10% more compilation
# memory on the vector hyperdiffusion benchmark (point views are taken once per
# node of a broadcast expression per slice loop over it).
@generated Base.view(bc::LazyDataLayout, index::PointIndex) = quote
    Base.@_propagate_inbounds_meta
    args = getfield(bc, :args)
    return Broadcast.Broadcasted(
        getfield(bc, :style),
        getfield(bc, :f),
        Base.Cartesian.@ntuple(
            $(length(bc.parameters[4].parameters)),
            n -> let arg = getfield(args, n)
                arg isa MaybeLazyDataLayout ?
                view(arg, Broadcast.newindex(arg, index)) : arg
            end,
        ),
    )
end
@propagate_inbounds Base.view(bc::FusedMultiBroadcast, index::PointIndex) =
    FusedMultiBroadcast(
        unrolled_map_with_inbounds(bc.pairs) do (dest, arg)
            Base.@_propagate_inbounds_meta
            Pair(
                view(dest, Broadcast.newindex(dest, index)),
                arg isa MaybeLazyDataLayout ?
                view(arg, Broadcast.newindex(arg, index)) : arg,
            )
        end,
    )

# A single-point slice of multidimensional data keeps its number of dimensions,
# so single-point broadcasts are identified by their length instead of ndims.
# One method per broadcast type, avoiding ambiguity with the PointIndex methods.
@inline Base.view(bc::LazyDataLayout, ::CartesianIndex{0}) =
    isone(length(bc)) ? bc : Base.throw_boundserror(bc, (CartesianIndex(),))
@inline Base.view(bc::FusedMultiBroadcast, ::CartesianIndex{0}) =
    isone(length(bc)) ? bc : Base.throw_boundserror(bc, (CartesianIndex(),))

@propagate_inbounds Base.setindex!(data::DataLayout, value, indices::PointIndex...) =
    setindex!(data, value, CartesianIndex(indices...))
@propagate_inbounds Base.getindex(data::DataLayout, indices::PointIndex...) =
    getindex(data, CartesianIndex(indices...))
@propagate_inbounds Base.view(arg::IndexableData, indices::PointIndex...) =
    view(arg, CartesianIndex(indices...))

# Reduce latency by only constructing slice views when necessary. A slice with
# exactly one point is returned as a zero-dimensional view, so that every way
# of slicing down to one point produces the same representation (see the note
# on single-point views above); a single-point broadcast slice would otherwise
# keep multidimensional style and axes, incompatible with zero-dimensional
# slices of materialized data.
@propagate_inbounds function level(arg::IndexableData, v)
    (; Nv, Ni, Nj, Nh) = vijh_params(arg)
    Ni == Nj == Nh == 1 && return view(arg, CartesianIndex(v, 1, 1, 1))
    Nv == 1 || return level_view(arg, v)
    @boundscheck v == 1 || throw(ArgumentError("DataLayout has only one level"))
    return arg
end
@propagate_inbounds function slab(arg::IndexableData, v, h)
    (; Nv, Ni, Nj, Nh) = vijh_params(arg)
    Ni == Nj == 1 && return view(arg, CartesianIndex(v, 1, 1, h))
    Nv == Nh == 1 || return slab_view(arg, v, h)
    @boundscheck v == h == 1 || throw(ArgumentError("DataLayout has only one slab"))
    return arg
end
@propagate_inbounds function column(arg::IndexableData, i, j, h)
    (; Nv, Ni, Nj, Nh) = vijh_params(arg)
    Nv == 1 && return view(arg, CartesianIndex(1, i, j, h))
    Ni == Nj == Nh == 1 || return column_view(arg, i, j, h)
    @boundscheck i == j == h == 1 || throw(ArgumentError("DataLayout has only one column"))
    return arg
end

# Convenience methods for data with a single vertical level or a single
# horizontal dimension, matching the corresponding methods for spaces.
@propagate_inbounds slab(arg::IndexableData, h) = slab(arg, 1, h)
@propagate_inbounds column(arg::IndexableData, i, h) = column(arg, i, 1, h)
