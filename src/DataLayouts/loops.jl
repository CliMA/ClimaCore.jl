@inline is_valid_slice_mask(::NoMask, _) = true
@inline is_valid_slice_mask(::IJHMask, ::typeof(column)) = true
@inline is_valid_slice_mask(::IJHMask, ::typeof(view)) = true
@inline is_valid_slice_mask(::IJHMask, _) = false

@inline each_maskable_slice_index(_, op::O, args...) where {O} =
    each_slice_index(op, args...)
@inline each_maskable_slice_index(mask::IJHMask, ::typeof(view), args...) =
    eachindex(IndexStyle(mask.is_active, args...), args...)

@inline function subscope_slice_indices(subscope, scope, mask, op::O, args...) where {O}
    is_valid_slice_mask(mask, op) || throw(ArgumentError(invalid_mask_string(mask, op)))
    full_scope_indices = each_maskable_slice_index(mask, op, args...)
    indices = @inbounds subscope_indices(subscope, scope, full_scope_indices)
    mask == NoMask() && return indices
    return Iterators.filter(index -> (@inbounds should_compute(mask, index)), indices)
end
@generated invalid_mask_string(::M, ::O) where {M, O} =
    "$M cannot be applied to $(O.instance) slices"

@inline function inferred_slice_length(op::O, arg) where {O}
    index = one(eltype(each_slice_index(op, arg)))
    slice_type = return_type(op, typeof((arg, Tuple(index)...)))
    has_inferred_size(slice_type) && return prod(inferred_size(slice_type))
    throw(ArgumentError("Size of view from slice operator must be inferrable"))
end

"""
    slice_subscope(scope, op, args...)

[`DataScope`](@ref) that [`foreach_slice`](@ref) assigns to slices of the given
arguments when parallelizing over `scope`. By default, this is the smallest
subset of `scope` that does not require any thread to process more than one
point from the largest slice returned by `op`. When no such subset is available,
the largest subset is used in order to minimize the number of points per thread.
"""
@inline function slice_subscope(scope, op::O, args...) where {O}
    subscope = partition(scope)
    subscope == ThisThread() && return subscope
    max_slice_points = unrolled_maximum(Base.Fix1(inferred_slice_length, op), args)
    max_slice_points > num_threads(partition(subscope)) && return subscope
    return slice_subscope(op, subscope, args...)
end

"""
    foreach_slice([scope], op, f, args...; [mask])

Generalization of `eachslice`/`mapslices` that applies `f` to slices of every
[`DataLayout`](@ref) or similarly indexable argument, where the slice operator
`op` can be any of the following:
 - [`level`](@ref), but only when [`nelems`](@ref) is statically inferrable
 - [`slab`](@ref) or [`column`](@ref)
 - `view` (for single-point slices)

Each slice is assigned to a [`slice_subscope`](@ref) of `scope`, which by
default is the largest available [`DataScope`](@ref) that can access every
argument. A [`DataMask`](@ref), which by default is set to [`NoMask`](@ref), may
also be used to skip over a particular subset of slices.
"""
@inline foreach_slice(op::O, f::F, args...; mask = NoMask()) where {O, F} =
    foreach_slice(DataScope(args...), op, f, args...; mask)

# Change the scope to ThisThread when given only one thread.
@inline foreach_slice(scope::DataScope, op::O, f::F, args...; mask) where {O, F} =
    isone(num_threads(scope)) ? foreach_slice(ThisThread(), op, f, args...; mask) :
    parallelize_over(scope) do
        subscope = slice_subscope(scope, op, args...)
        for index in subscope_slice_indices(subscope, scope, mask, op, args...)
            slices = unrolled_map(args) do arg
                @inbounds reassign(op(arg, Tuple(index)...), subscope)
            end
            f(slices...)
        end
    end

@inline foreach_slice(::ThisThread, op::O, f::F, args...; mask) where {O, F} =
    for index in subscope_slice_indices(ThisThread(), ThisThread(), mask, op, args...)
        slices = unrolled_map(arg -> (@inbounds op(arg, Tuple(index)...)), args)
        f(slices...)
    end

"""
    foreach_point(f, args...; [mask])

Run [`foreach_slice`](@ref) with `view` as the slice operator.
"""
@inline foreach_point(f::F, args...; kwargs...) where {F} =
    foreach_slice(view, f, args...; kwargs...)

for op in (:level, :slab, :column)
    @eval begin
        """
            foreach_$($op)(f, args...; [mask])

        Run [`foreach_slice`](@ref) with [`$($op)`](@ref) as the slice operator.
        """
        @inline $(Symbol(:foreach_, op))(f::F, args...; kwargs...) where {F} =
            foreach_slice($op, f, args...; kwargs...)
    end
end

"""
    reduce_points([scope], op, arg; [mask], [init])

Generalization of `reduce` that uses `op` to combine values stored in a
[`DataLayout`](@ref) or similarly indexable argument.

This combines all values in the given argument that are assigned to `scope`,
which by default is the largest available [`DataScope`](@ref) that can access
the argument. A [`DataMask`](@ref), which by default is set to [`NoMask`](@ref),
may also be used to skip over a particular subset of points. If the `mask`
disables every point, or if there are no points in `arg` to begin with, the
`init` value must be specified.
"""
@inline reduce_points(op::O, arg; mask = NoMask(), init...) where {O} =
    reduce_points(DataScope(arg), op, arg; mask, init...)

# Change the scope to ThisThread when given only one thread or a small argument.
# Otherwise, reduce each thread's values, then reduce the results in one thread.
@inline function reduce_points(scope::DataScope, op::O, arg; kwargs...) where {O}
    (isone(num_threads(scope)) || length(arg) <= num_threads(scope)) &&
        return reduce_points(ThisThread(), op, reassign(arg, ThisThread()); kwargs...)
    T = return_type(op, NTuple{2, eltype(arg)})
    results = scoped_array(scope, T, num_threads(scope))
    parallelize_over(scope) do
        @inbounds results[thread_rank(scope)] =
            reduce_points(ThisThread(), op, arg; kwargs...)
    end
    return reduce(op, results)
end

@inline function reduce_points(::ThisThread, op::O, arg; mask, init...) where {O}
    point_indices = subscope_slice_indices(ThisThread(), DataScope(arg), mask, view, arg)
    return mapreduce(index -> (@inbounds arg[index]), op, point_indices; init...)
end

"""
    column_reduce!(op, dest, arg; [mask], [flip], [init])

Use [`foreach_column`](@ref) to pass each column of `arg` to `reduce`, storing
the results in corresponding columns of `dest`. Setting `flip` to `true` changes
the order of reduction from left-associative (default) to right-associative.
"""
@inline column_reduce!(op::O, dest, arg; mask = NoMask(), flip = false, init...) where {O} =
    foreach_column(dest, arg; mask) do dest_column, arg_column
        maybe_reverse = flip ? reverse : identity
        fill!(dest_column, reduce(op, maybe_reverse(arg_column); init...))
    end
# TODO: Extend this to column_accumulate!, column_stencil!, and slab_convolve!

function Base.fill!(dest::DataLayout, value; kwargs...)
    foreach_point(dest_point -> (@inbounds dest_point[] = value), dest; kwargs...)
    call_post_op_callback() && post_op_callback(dest, dest, value; kwargs...)
    return dest
end

# Replicate the scalar broadcast method from Base's copyto! for AbstractArrays.
# Add a StaticArrayStyle{0} method to resolve an ambiguity with StaticArrays.
for S in (:(<:Broadcast.AbstractArrayStyle{0}), :(<:StaticArrays.StaticArrayStyle{0}))
    @eval function Base.copyto!(dest::DataLayout, bc::Broadcast.Broadcasted{$S}; kwargs...)
        (bc.f == identity && isone(length(bc.args)) && Broadcast.isflat(bc)) &&
            return fill!(dest, first(bc.args); kwargs...)
        foreach_point(dest_point -> (@inbounds dest_point[] = bc[]), dest; kwargs...)
        call_post_op_callback() && post_op_callback(dest, dest, bc; kwargs...)
        return dest
    end
end

# Handle single-element tuples in DataLayout broadcasts the same way as Refs.
# For multi-element tuples, fall back to Base's default copyto! implementation.
@inline function Base.copyto!(
    dest::DataLayout,
    bc::Broadcast.Broadcasted{Broadcast.Style{Tuple}};
    kwargs...,
)
    is_length_one(arg::Tuple) = isone(length(arg))
    is_length_one(bc::Broadcast.Broadcasted) = unrolled_all(is_length_one, bc.args)
    style = is_length_one(bc) ? Broadcast.DefaultArrayStyle{0}() : nothing
    return copyto!(dest, convert(Broadcast.Broadcasted{typeof(style)}, bc); kwargs...)
end

function Base.copyto!(dest::DataLayout, arg::MaybeLazyDataLayout; kwargs...)
    foreach_point(dest, arg; kwargs...) do dest_point, arg_point
        @inbounds dest_point[] = arg_point[]
    end
    call_post_op_callback() && post_op_callback(dest, dest, arg; kwargs...)
    return dest
end

function Base.copyto!(bc::FusedMultiBroadcast; kwargs...)
    foreach_point(bc; kwargs...) do bc_point
        unrolled_foreach(bc_point.pairs) do (dest_point, arg_point)
            @inbounds dest_point[] = arg_point[]
        end
    end
    call_post_op_callback() && post_op_callback(bc, bc; kwargs...)
    return bc
end

@inline Base.copy(arg::MaybeLazyDataLayout; kwargs...) =
    copyto!(similar(arg), arg; kwargs...)

# Add axes to LazyDataLayouts and AutoBroadcaster wrappers to DataLayouts before
# reducing them. Remove all AutoBroadcaster wrappers after obtaining the result.
function Base.reduce(op::O, arg::MaybeLazyDataLayout; kwargs...) where {O}
    reducible = arg isa LazyDataLayout ? Broadcast.instantiate : Broadcast.broadcastable
    result = drop_auto_broadcasters(reduce_points(op, reducible(arg); kwargs...))
    call_post_op_callback() && post_op_callback(result, op, arg; kwargs...)
    return result
end

# Combine arguments for map!, map, and mapreduce into LazyDataLayouts.
@inline Base.map!(
    f::F,
    dest::DataLayout,
    args::MaybeLazyDataLayout...;
    kwargs...,
) where {F} = copyto!(dest, Broadcast.broadcasted(f, args...); kwargs...)
@inline Base.map(
    f::F,
    arg::MaybeLazyDataLayout,
    args::MaybeLazyDataLayout...;
    kwargs...,
) where {F} = copy(Broadcast.broadcasted(f, arg, args...); kwargs...)
@inline Base.mapreduce(
    f::F,
    op::O,
    arg::MaybeLazyDataLayout,
    args::MaybeLazyDataLayout...;
    kwargs...,
) where {F, O} = reduce(op, Broadcast.broadcasted(f, arg, args...); kwargs...)

# Avoid constructing a LazyDataLayout if the broadcast operation does nothing.
@inline Base.mapreduce(
    ::typeof(identity),
    op::O,
    arg::MaybeLazyDataLayout;
    kwargs...,
) where {O} = reduce(op, arg; kwargs...)

# Optimize simple, unmasked equality checks by deferring to parent arrays.
@inline Base.:(==)(arg1::DataLayout, arg2::DataLayout; mask = NoMask()) =
    eltype(arg1) == eltype(arg2) &&
    layout_type(arg1) == layout_type(arg2) &&
    shape_params(arg1) == shape_params(arg2) &&
    mask == NoMask() ? parent(arg1) == parent(arg2) :
    mapreduce(==, &, arg1, arg2; mask, init = true)
@inline Base.:(==)(arg1::MaybeLazyDataLayout, arg2::MaybeLazyDataLayout; mask = NoMask()) =
    mapreduce(==, &, arg1, arg2; mask, init = true)
