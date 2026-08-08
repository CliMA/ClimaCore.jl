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
@inline function equal_layout_shapes(args::Tuple)
    non_point_args = unrolled_filter(is_non_point_arg, args)
    return unrolled_allequal(layout_type, non_point_args) &&
           unrolled_allequal(shape_params, non_point_args)
end

# A broadcast expression permits linear point indices whenever its non-point
# layouts share a shape: each argument then converts a linear index as its own
# layout requires. Layouts with different shapes require Cartesian indices,
# which Broadcast.newindex projects onto singleton dimensions.
@inline Base.IndexStyle(bc::MaybeFusedDataLayoutBroadcast) =
    equal_layout_shapes(layout_args(bc)) ? IndexLinear() : IndexCartesian()

@inline index_style_layouts(::Tuple{}) = IndexLinear()
@inline index_style_layouts(args::Tuple{Any}) = IndexStyle(first(args))
@inline index_style_layouts(args::Tuple) =
    equal_layout_shapes(args) ?
    unrolled_mapreduce(IndexStyle, IndexStyle, args) : IndexCartesian()

# Allow linear indexing if all DataLayouts in an expression have the same shape.
# Add DataLayout-only methods to avoid ambiguities with AbstractArray methods.
for T in (:IndexableData, :DataLayout)
    @eval @inline Base.IndexStyle(arg1::$T, arg2::$T, args::$T...) =
        index_style_layouts((arg1, arg2, args...))

    @eval @inline Base.eachindex(arg::$T, args::$T...) =
        eachindex(IndexStyle(arg, args...), arg, args...)
    @eval @inline Base.eachindex(::IndexLinear, arg::$T, args::$T...) =
        unrolled_allequal(length, (arg, args...)) ? Base.OneTo(length(arg)) :
        throw(DimensionMismatch("Inputs to eachindex must have the same length"))
    @eval @inline Base.eachindex(::IndexCartesian, arg::$T, args::$T...) =
        unrolled_allequal(size, (arg, args...)) ? CartesianIndices(size(arg)) :
        throw(DimensionMismatch("Inputs to eachindex must have the same size"))
end

@inline slice_axes(op, arg) = axes(each_slice_index(op, arg))

"""
    each_slice_index(op, args...)

Generalization of `eachindex` for the slice operators [`level`](@ref),
[`slab`](@ref), [`column`](@ref), and `view` (for creating single-point slices).
The result is always an iterator of Cartesian indices, whose scalar offsets are
simple enough for SIMD optimization (a `view` at a linear index wraps its parent
in a 1-dimensional `ReshapedArray`, which blocks SIMD in pointwise loops).

The arguments' axes are combined with `stable_combine_axes`, which expands
singleton and 0-dimensional axes like broadcasting does and never throws, since
this function is called from GPU kernels, where an error path either fails to
compile or traps with an unrelated CUDA error. Arguments whose axes are
genuinely incompatible are rejected on the host by [`foreach_slice`](@ref)
before any kernel is launched.
"""
@inline each_slice_index(op::O, args...) where {O} = CartesianIndices(
    unrolled_reduce(
        stable_combine_axes,
        unrolled_map(Base.Fix1(slice_axes, op), args),
    ),
)

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

# Like single-point broadcasts, single-point layouts (e.g. level slices with one
# level) are identified by their length, since they keep their dimensions.
@inline is_single_point(data, index) = isone(length(data)) && index == CartesianIndex()

# Avoid unnecessary indexing arithmetic whenever possible. Base's default array
# access methods use Cartesian-to-linear index conversions, without any constant
# propagation of array dimensions. Even worse, Base's default SubArray access
# methods use linear-to-Cartesian index conversions, calling div/rem at runtime.
# This function ensures that Cartesian-to-linear conversion is constant-folded,
# and it uses a linear index to access the parent of a constant-stride SubArray.
@inline field_offset(array::SubArray, ::Val{F}) where {F} =
    first(parentindices(array)[F]) - 1

@propagate_inbounds function array_and_index_args(data::DataLayout, index::Integer)
    array = parent(data)
    is_single_point(data, index) && return (array, ())
    F = f_dim(data)
    isnothing(F) && return (array, (CartesianIndices(data)[index], Val(F)))
    # The strides below are computed from runtime sizes rather than from
    # inferred_size, so that layouts with dynamic extents (every field has a
    # dynamic Nh) can use the linear fast paths; LLVM constant-folds the
    # products whenever the sizes are statically known. Converting the index
    # with CartesianIndices is only needed when a multi-component layout is
    # given a linear point index, since its parent cannot be accessed linearly.
    if is_constant_stride_view_type(typeof(array), Val(F))
        stride = prod(size(data)[1:(F - 1)])
        f0 = field_offset(array, Val(F))
        Nf_parent = size(parent(array), F)
        h0 = (index - 1) ÷ stride
        p0 = (index - 1) % stride
        parent_idx = p0 + f0 * stride + h0 * (stride * Nf_parent) + 1
        return (parent(array), (parent_idx, stride))
    elseif IndexStyle(data) == IndexLinear()
        stride = prod(size(data)[1:(F - 1)])
        return (array, (index, stride))
    else
        return (array, (CartesianIndices(data)[index], Val(F)))
    end
end

@propagate_inbounds function array_and_index_args(data::DataLayout, index::CartesianIndex)
    array = parent(data)
    is_single_point(data, index) && return (array, ())
    F = f_dim(data)
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
        view_point_struct(data, safe_index(data, index)),
    )

@propagate_inbounds view_point_struct(data, index) =
    view_struct(parent(data), eltype(data), index, Val(f_dim(data)))

# A point view at a linear index is built from the same array and index
# arguments as getindex and setindex!, so that linearly indexable data gets a
# strided view of its parent's linear indices (at most one integer division,
# on constant-stride field views) instead of a linear-to-Cartesian index
# conversion (one division per dimension); point loops construct a view of
# every argument at every point, so this conversion would otherwise dominate
# the index arithmetic of GPU broadcast kernels. Data that requires Cartesian
# indices is converted by array_and_index_args.
@propagate_inbounds function view_point_struct(data, index::Integer)
    (array, index_args) = array_and_index_args(data, index)
    return view_struct(array, eltype(data), index_args...)
end

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
