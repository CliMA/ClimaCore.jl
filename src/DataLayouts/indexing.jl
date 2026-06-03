# Allow linear indexing if parent(data)[1:length(data)] has one value per point.
Base.IndexStyle(::Type{D}) where {D <: DataLayout} =
    ndims(D) <= 1 ? IndexLinear() :
    ncomponents(D) <= 1 || all_ones(inferred_size(D)[f_dim(D):end]...) ?
    IndexStyle(parent_type(D)) : IndexCartesian()

Base.IndexStyle(bc::LazyDataLayout) = IndexStyle(layout_args(bc)...)
Base.IndexStyle(bc::FusedMultiBroadcast) = IndexStyle(unrolled_map(first, bc.pairs)...)

const IndexableData = Union{DataLayout, LazyDataLayout, FusedMultiBroadcast}

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

# Override checkbounds for LazyDataLayouts to prevent unnecessary BoundsErrors.
@inline Base.checkbounds(bc::LazyDataLayout, index::Integer) =
    1 <= index <= length(bc) || Base.throw_boundserror(bc, (index,))
@inline Base.checkbounds(bc::LazyDataLayout, ::CartesianIndex{0}) = checkbounds(bc, 1)

is_invalid_linear(data, index) = index isa Integer && IndexStyle(data) isa IndexCartesian

const PointIndex = Union{Integer, CartesianIndex}

# Always convert to the expected element type when modifying a DataLayout.
@propagate_inbounds Base.setindex!(data::DataLayout, value) =
    isone(length(data)) ? set_struct!(parent(data), convert(eltype(data), value)) :
    throw(ArgumentError("setindex! requires an index for data with multiple points"))
@propagate_inbounds Base.setindex!(data::DataLayout, value, index::PointIndex) =
    is_invalid_linear(data, index) ? setindex!(data, value, CartesianIndices(data)[index]) :
    set_struct!(parent(data), convert(eltype(data), value), index, Val(f_dim(data)))

@propagate_inbounds Base.getindex(data::DataLayout) =
    isone(length(data)) ? get_struct(parent(data), eltype(data)) :
    throw(ArgumentError("getindex requires an index for data with multiple points"))
@propagate_inbounds Base.getindex(data::DataLayout, index::PointIndex) =
    is_invalid_linear(data, index) ? getindex(data, CartesianIndices(data)[index]) :
    get_struct(parent(data), eltype(data), index, Val(f_dim(data)))

@inline Base.view(data::DataLayout) =
    isone(length(data)) ? data :
    throw(ArgumentError("view requires an index for data with multiple points"))
@propagate_inbounds function Base.view(data::DataLayout, index::PointIndex)
    is_invalid_linear(data, index) && return view(data, CartesianIndices(data)[index])
    size_param_names = unrolled_filter(!=(:F), keys(shape_params(data)))
    size_param_pairs = unrolled_map(Base.Fix2(Pair, 1), size_param_names)
    array = view_struct(parent(data), eltype(data), index, Val(f_dim(data)))
    return rebuild(data, array; size_param_pairs...)
end

# Combine multiple integers into a CartesianIndex. Add DataLayout/LazyDataLayout
# getindex methods to avoid ambiguities with AbstractArray/Broadcasted methods.
@propagate_inbounds Base.setindex!(
    data::DataLayout,
    value,
    index1::Integer,
    index2::Integer,
    indices::Integer...,
) = setindex!(data, value, CartesianIndex(index1, index2, indices...))
for T in (:IndexableData, :DataLayout, :LazyDataLayout)
    @eval @propagate_inbounds Base.getindex(
        arg::$T,
        index1::Integer,
        index2::Integer,
        indices::Integer...,
    ) = getindex(arg, CartesianIndex(index1, index2, indices...))
end
@propagate_inbounds Base.view(
    arg::IndexableData,
    index1::Integer,
    index2::Integer,
    indices::Integer...,
) = view(arg, CartesianIndex(index1, index2, indices...))

all_ones(params...) = params isa Tuple{Vararg{Integer}} && unrolled_all(isone, params)

# Only construct slice views when necessary.
@propagate_inbounds level(arg::IndexableData, v) =
    all_ones(vijh_params(arg).Nv) ? arg : level_view(arg, v)
@propagate_inbounds slab(arg::IndexableData, v, h) =
    all_ones(vijh_params(arg).Nv, vijh_params(arg).Nh) ? arg : slab_view(arg, v, h)
@propagate_inbounds column(arg::IndexableData, i, j, h) =
    all_ones(vijh_params(arg).Ni, vijh_params(arg).Nj, vijh_params(arg).Nh) ? arg :
    column_view(arg, i, j, h)

@inline slice_index_limits(::typeof(level), arg) = (nlevels(arg),)
@inline slice_index_limits(::typeof(slab), arg) = (nlevels(arg), nelems(arg))
@inline slice_index_limits(::typeof(column), arg) =
    (vijh_params(arg).Ni, vijh_params(arg).Nj, nelems(arg))

"""
    each_slice_index(op, args...)

Generalization of `eachindex` for the slice operators [`level`](@ref),
[`slab`](@ref), [`column`](@ref), and `view` (for creating single-point slices).
The result is a `CartesianIndices` iterator when `op` is set to `level`, `slab`,
or `column`, and it is equivalent to `eachindex` when `op` is set to `view`.
"""
@inline each_slice_index(::typeof(view), args...) = eachindex(args...)
@inline each_slice_index(op::O, args...) where {O} =
    unrolled_allequal(Base.Fix1(slice_index_limits, op), args) ?
    CartesianIndices(slice_index_limits(op, first(args))) :
    throw(DimensionMismatch("Inputs to each_slice_index must have consistent dimensions"))
