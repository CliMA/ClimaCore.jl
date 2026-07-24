macro simd_if(condition_expr, loop_expr)
    esc(:($condition_expr ? $(:(@simd $loop_expr)) : $loop_expr))
end

# Whether a point loop over these indices should run under @simd. This requires
# an indexable iterator, and it only pays off when LLVM can vectorize across
# many contiguous points, like in the flattened CartesianIndices iterations of
# CPU loops. GPU scopes override this for their strided index subsets: each GPU
# thread only iterates a few points, and @simd's loop restructuring just adds
# branches and index arithmetic to every kernel.
@inline simd_over_indices(indices) = indices isa AbstractArray

@inline is_valid_slice_mask(::NoMask, _) = true
@inline is_valid_slice_mask(::IJHMask, ::typeof(view)) = true
@inline is_valid_slice_mask(::IJHMask, ::typeof(column)) = true
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

# A view slice contains one point by definition, so its size is known without
# constructing a slice. The generic method cannot handle point slices of fused
# broadcasts, whose 0-dimensional DataF args and dimension-preserving
# Broadcasted args have inconsistent values of inferred_size.
@inline num_slice_points(::typeof(view), arg) = 1
@inline function num_slice_points(op::O, arg) where {O}
    isempty(each_slice_index(op, arg)) && return 0
    first_slice = @inbounds op(arg, Tuple(first(each_slice_index(op, arg)))...)
    has_inferred_size(first_slice) && return prod(inferred_size(first_slice))
    throw(ArgumentError("Size of slice operator result must be inferrable"))
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
    max_slice_points = unrolled_maximum(Base.Fix1(num_slice_points, op), args)
    max_slice_points > num_threads(partition(subscope)) && return subscope
    return slice_subscope(subscope, op, args...)
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

# Change the scope to ThisThread when given only one thread, which compiles the
# simplest possible loop.
@inline foreach_slice(scope::DataScope, op::O, f::F, args...; mask) where {O, F} =
    isone(num_threads(scope)) ? foreach_slice(ThisThread(), op, f, args...; mask) :
    parallelize_over(() -> slice_loop(scope, op, f, mask, args...), scope)

# Run the loop without parallelize_over when the scope is ThisThread. Since each
# slice is reassigned to its subscope, nested loops in f dispatch here
# statically. This avoids both the runtime thread-count check and the
# parallelize_over closure, which the compiler does not always remove, causing
# an allocation at every slice of the outer loop.
@inline foreach_slice(::ThisThread, op::O, f::F, args...; mask) where {O, F} =
    slice_loop(ThisThread(), op, f, mask, args...)

# Point loops need @simd and an inlined call to f for vectorization, since LLVM
# cannot vectorize across a flattened CartesianIndices iterator unless @simd
# splits it into an outer loop and a unit-stride inner loop, and Julia's inliner
# gives up on point closures over large broadcast expressions, which forces
# every argument slice to be materialized at each point. Without @simd and
# @inline, pointwise loops are several times slower than ordinary Array
# broadcasts. Lazy iterators such as Iterators.filter or Iterators.map do not
# support @simd, and operations on non-point slices typically do too much work
# per slice for vectorization to be worthwhile.
@inline function slice_loop(scope, op::O, f::F, mask, args...) where {O, F}
    subscope = slice_subscope(scope, op, args...)
    indices = subscope_slice_indices(subscope, scope, mask, op, args...)
    @simd_if (op == view && simd_over_indices(indices)) for index in indices
        slices = unrolled_map(args) do arg
            @inbounds reassign(op(arg, Tuple(index)...), subscope)
        end
        @inline f(slices...)
    end
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

# Reduce all unmasked points by folding over their indices, which are nonempty
# since the launcher assigns every thread at least one point. Use safe_mapreduce
# instead of Base's pairwise mapreduce to avoid the empty-collection error path,
# whose string cannot be compiled in GPU kernels. Masked reductions require an
# init, and mapreduce's empty path is only reached without one. Masked indices
# are equivalent to what the slice index machinery uses for single-point views.
@inline reduce_points(::ThisThread, op::O, arg; mask, init...) where {O} =
    if mask == NoMask()
        indices = @inbounds subscope_indices(ThisThread(), DataScope(arg), eachindex(arg))
        safe_mapreduce(index -> (@inbounds arg[index]), op, indices; init...)
    else
        indices = subscope_slice_indices(ThisThread(), DataScope(arg), mask, view, arg)
        mapreduce(index -> (@inbounds arg[index]), op, indices; init...)
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

# Convert the value before the fill! loop. Even though setindex! converts at
# every point, the compiler does not hoist the conversion, and filling a Float64
# layout with an Int is measurably slower if the conversion isn't done first.
# The converted value is passed to GPU kernels as parent array entries because
# Int128 and UInt128 fields in kernel arguments crash LLVM's NVPTX backend prior
# to LLVM 20 (llvm/llvm-project#49221). 128-bit integers are only safe in
# registers, like the ones bitcast_struct uses to reconstruct the value.
function Base.fill!(dest::DataLayout, value; kwargs...)
    B = eltype(parent(dest))
    converted_value = convert(eltype(dest), value)
    entries = bitcast_struct(NTuple{num_basetypes(B, eltype(dest)), B}, converted_value)
    foreach_point(dest; kwargs...) do dest_point
        @inbounds dest_point[] = bitcast_struct(eltype(dest_point), entries)
    end
    call_post_op_callback() && post_op_callback(dest, dest, value; kwargs...)
    return dest
end

# Replicate Base's scalar broadcast copyto!, where data .= value becomes fill!,
# and any other scalar broadcast becomes a pointwise loop. Since materialize!
# attaches dest's axes, but foreach_point strips dest of its axes, the scalar
# broadcast must also have its axes dropped, mirroring how Base's instantiate
# drops scalar broadcast axes. The StaticArrayStyle{0} and AbstractBlockStyle{0}
# methods avoid ambiguities with StaticArrays and BlockArrays.
for S in (
    :(<:Broadcast.AbstractArrayStyle{0}),
    :(<:StaticArrays.StaticArrayStyle{0}),
    :(<:BlockArrays.AbstractBlockStyle{0}),
)
    @eval @inline Base.copyto!(dest::DataLayout, bc::Broadcast.Broadcasted{$S}; kwargs...) =
        if bc.f === identity && isone(length(bc.args)) && Broadcast.isflat(bc)
            @inbounds arg = first(bc.args)
            @inbounds fill!(dest, arg isa Tuple ? first(arg) : arg[]; kwargs...)
        else
            bc_without_axes = Broadcast.Broadcasted(bc.f, bc.args)
            foreach_point(dest; kwargs...) do dest_point
                @inbounds dest_point[] = first(bc_without_axes)
            end
            call_post_op_callback() && post_op_callback(dest, dest, bc; kwargs...)
            dest
        end
end

@inline is_scalar_or_length_one(arg) = true
@inline is_scalar_or_length_one(arg::Tuple) = isone(length(arg))
@inline is_scalar_or_length_one(bc::Broadcast.Broadcasted) =
    unrolled_all(is_scalar_or_length_one, bc.args)

# Handle single-element tuples in DataLayout broadcasts the same way as Refs.
# For multi-element tuples, fall back to Base's default copyto! implementation.
@inline function Base.copyto!(
    dest::DataLayout,
    bc::Broadcast.Broadcasted{Broadcast.Style{Tuple}};
    kwargs...,
)
    style_type = is_scalar_or_length_one(bc) ? Broadcast.DefaultArrayStyle{0} : Nothing
    return copyto!(dest, convert(Broadcast.Broadcasted{style_type}, bc); kwargs...)
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

# Optimize unmasked equality checks for similar layouts with the same packed
# (un-padded) element types by deferring to their parent arrays. Padded values
# should not be compared in this way, since equality must not depend on padding.
@inline Base.:(==)(arg1::DataLayout, arg2::DataLayout; mask = NoMask()) =
    size(arg1) == size(arg2) && (
        mask == NoMask() &&
        eltype(arg1) == eltype(arg2) &&
        (Base.ispacked(eltype(arg1)) && Base.ispacked(eltype(arg2))) &&
        (layout_type(arg1) == layout_type(arg2) && f_dim(arg1) == f_dim(arg2)) ?
        parent(arg1) == parent(arg2) : mapreduce(==, &, arg1, arg2; mask, init = true)
    )
@inline Base.:(==)(arg1::MaybeLazyDataLayout, arg2::MaybeLazyDataLayout; mask = NoMask()) =
    size(arg1) == size(arg2) && mapreduce(==, &, arg1, arg2; mask, init = true)
