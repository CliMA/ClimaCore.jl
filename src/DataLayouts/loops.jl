macro simd_if(condition_expr, loop_expr)
    esc(:($condition_expr ? $(:(@simd $loop_expr)) : $loop_expr))
end

# Whether a point loop over these indices should run under @simd. This requires
# an indexable iterator, and it only pays off when LLVM can vectorize across
# many contiguous points, like in the flattened CartesianIndices iterations of
# CPU loops. GPU scopes override this for their strided index subsets: each GPU
# thread only iterates a few points, and @simd's loop restructuring just adds
# branches and index arithmetic to every kernel.
# The dispatch is on the index type rather than on the loop's scope, which would
# read as the more direct question. Taking a scope here regressed GPU kernels
# when it was tried, and the scope is a weaker discriminator than it looks:
# ThisThread is the scope of point loops on both hosts and devices.
@inline simd_over_indices(indices) = indices isa AbstractArray

@inline is_valid_slice_mask(::NoMask, _) = true
@inline is_valid_slice_mask(::IJHMask, ::typeof(view)) = true
@inline is_valid_slice_mask(::IJHMask, ::typeof(column)) = true
@inline is_valid_slice_mask(::IJHMask, _) = false

# The scope is passed so that GPU scopes can override the indices of unmasked
# point loops with eachindex, whose linear indices avoid the integer divisions
# required for linear-to-Cartesian conversion. CPU scopes keep the Cartesian
# each_slice_index for point loops: a view at a linear index wraps its parent in
# a 1-dimensional ReshapedArray that blocks SIMD (see each_slice_index), which
# would make single-component CPU broadcasts slow, while GPU threads iterate too
# few points per thread for SIMD to matter. Although CPU scopes only need one
# argument, GPU scopes must check every argument for linear indexing support.
@inline each_maskable_slice_index(_, _, op::O, args...) where {O} =
    each_slice_index(op, first(args))
@inline each_maskable_slice_index(_, mask::IJHMask, ::typeof(column), args...) =
    ActiveColumnIndices(mask)
@inline each_maskable_slice_index(_, mask::IJHMask, ::typeof(view), args...) =
    ActivePointIndices{size(first(args), 1)}(mask)

# Every valid mask and slice operator combination has indexable slice indices:
# NoMask uses the full index ranges, and IJHMask (which only supports column
# and view slices) uses compacted active indices, so no combination requires
# filtering the indices with a per-slice mask lookup.
@noinline throw_invalid_slice_mask(mask, op::O) where {O} =
    throw(ArgumentError(invalid_mask_string(mask, op)))
@generated invalid_mask_string(
    ::M,
    ::O,
) where {M, O} = "$M cannot be applied to $(O.instance) slices"

@inline function maskable_slice_indices(scope, mask, op::O, args...) where {O}
    is_valid_slice_mask(mask, op) || throw_invalid_slice_mask(mask, op)
    return each_maskable_slice_index(scope, mask, op, args...)
end

@inline subscope_slice_indices(subscope, scope, mask, op::O, args...) where {O} =
    @inbounds subscope_indices(
        subscope,
        scope,
        maskable_slice_indices(scope, mask, op, args...),
    )

# Number of points in each slice that op generates from arg, measured from the
# slice via inferred_size so slice_subscope stays statically inferrable; a
# non-inferrable size is treated as unbounded, stopping slice_subscope's descent
# at the largest scope. A view slice contains one point by definition (point
# slices of fused broadcasts have inconsistent inferred_size across args).
@inline num_slice_points(::typeof(view), arg) = 1
@inline function num_slice_points(op::O, arg) where {O}
    slice = @inbounds op(arg, Tuple(first(each_slice_index(op, arg)))...)
    return has_inferred_size(slice) ? prod(inferred_size(slice)) : typemax(Int)
end

"""
    slice_subscope(scope, op, args...)

[`DataScope`](@ref) that [`foreach_slice`](@ref) assigns to slices of the given
arguments when parallelizing over `scope`. By default, this is the smallest
scope that does not require any thread to process more than one point from the
largest slice returned by `op`, out of `scope` itself and its subsets. When no
such scope is available, the largest subset is used in order to minimize the
number of points per thread.

`scope` itself is only used when its thread count is a compile-time constant
(see [`static_num_threads`](@ref)), since `scoped_slice_loop` gives every
slice a scope with a statically known number of threads.
"""
@inline function slice_subscope(scope, op::O, args...) where {O}
    partition(scope) == ThisThread() && return ThisThread()
    max_slice_points = unrolled_maximum(Base.Fix1(num_slice_points, op), args)
    return points_subscope(scope, max_slice_points)
end

# Recursing on the precomputed point count (rather than on op and args) keeps
# the descent from re-materializing the argument tuple and re-slicing every
# argument at each scope level, which allocates once per kernel launch on hosts.
@inline function points_subscope(scope, max_slice_points::Int)
    subscope = partition(scope)
    subscope == ThisThread() && return subscope
    max_slice_points > num_threads(partition(subscope)) &&
        return fits_in_scope(scope, subscope, max_slice_points) ? scope : subscope
    return points_subscope(subscope, max_slice_points)
end

# Whether a slice is too wide for every subset of scope but not for scope
# itself (re-slicing a slab inside another slab loop's body). A subset would
# give it half the threads it has points and double the shared memory that
# scoped_static_array reserves for every buffer the loop allocates.
@inline function fits_in_scope(scope, subscope, max_slice_points)
    scope_threads = static_num_threads(scope)
    subscope_threads = static_num_threads(subscope)
    isnothing(scope_threads) && return false
    isnothing(subscope_threads) && return false
    return subscope_threads < max_slice_points <= scope_threads
end

# The single definition of the slice-loop keyword defaults; every entry point
# slurps its keyword arguments and unpacks them here.
@inline slice_loop_flags(; mask = NoMask(), enumerate = Val(false)) =
    (mask, enumerate)

"""
    foreach_slice(op, f, args...; [mask], [enumerate])

Generalization of `eachslice`/`mapslices` that applies `f` to slices of every
[`DataLayout`](@ref) or similarly indexable argument, where the slice operator
`op` can be any of the following:

  - [`level`](@ref), but only when [`nelems`](@ref) is statically inferrable
  - [`slab`](@ref) or [`column`](@ref)
  - `view` (for single-point slices)

Each slice is assigned to a [`slice_subscope`](@ref) of `scope`, by default the
largest available [`DataScope`](@ref) that can access every argument, and a
[`DataMask`](@ref) may be used to skip a particular subset of slices.

Statements in `f` execute in order at the slice's scope: same-shaped pointwise
broadcasts assign every point to the same thread in each statement, and
spectral operators publish values that cross threads through synchronized
buffers, so each statement may read the results of the statements before it.

By default, `f` is called as `f(slices...)`. Setting `enumerate` to `Val(true)`
makes this `f(index, slices...)`, like in a loop over `Base.enumerate(arg)`.
"""
@inline function foreach_slice(op::O, f::F, args...; kwargs...) where {O, F}
    (mask, enumerate) = slice_loop_flags(; kwargs...)
    unrolled_allequal(Base.Fix1(each_slice_index, op), args) ||
        throw(DimensionMismatch("Inputs to foreach_slice must have compatible dimensions"))
    scope = DataScope(args...)
    # Go straight onto the loop barrier for scopes without setup: every layer
    # of keyword-argument forwarding costs a Core.kwcall method and a hidden
    # body method per loop, each re-optimized with the whole loop inlined.
    return needs_loop_setup(scope) ?
           foreach_slice(scope, op, f, args...; mask, enumerate) :
           scoped_slice_loop(
        slice_subscope(scope, op, args...), scope, op, f, mask, enumerate, args...,
    )
end

for (name, op, ref) in (
    (:foreach_point, :view, "`view`"),
    (:foreach_level, :level, "[`level`](@ref)"),
    (:foreach_slab, :slab, "[`slab`](@ref)"),
    (:foreach_column, :column, "[`column`](@ref)"),
)
    # The body of foreach_slice is replicated instead of forwarded to, since a
    # forwarding layer costs about as much inference as the loop it forwards to.
    @eval begin
        """
            $($name)(f, args...; [mask], [enumerate])

        Run [`foreach_slice`](@ref) with $($ref) as the slice operator.
        """
        @inline function $name(f::F, args...; kwargs...) where {F}
            (mask, enumerate) = slice_loop_flags(; kwargs...)
            unrolled_allequal(Base.Fix1(each_slice_index, $op), args) ||
                throw(
                    DimensionMismatch(
                        "Inputs to foreach_slice must have compatible dimensions",
                    ),
                )
            scope = DataScope(args...)
            return needs_loop_setup(scope) ?
                   foreach_slice(scope, $op, f, args...; mask, enumerate) :
                   scoped_slice_loop(
                slice_subscope(scope, $op, args...),
                scope,
                $op,
                f,
                mask,
                enumerate,
                args...,
            )
        end
    end
end

# Whether looping over a DataScope requires work before and after the loop,
# done by the scope's own foreach_slice method; extend alongside every method.
@inline needs_loop_setup(::DataScope) = false

# A thread pool has to be resolved before looping over it, and given back afterward.
@inline needs_loop_setup(::ThisThreadPool) = true
@inline function foreach_slice(
    scope::ThisThreadPool,
    op::O,
    f::F,
    args...;
    kwargs...,
) where {O, F}
    pool_thread_info() == (0, 0) ||
        return foreach_pool_slice(num_threads(scope), scope, op, f, args...; kwargs...)
    threads = resolve_pool_threads()
    try
        return foreach_pool_slice(threads, scope, op, f, args...; kwargs...)
    finally
        release_pool_threads()
    end
end

# Loop over the threads a pool loop resolved to. The count is passed as an integer rather
# than as a resolved scope, because a scope whose type is only known at run time becomes a
# union at every call below it, which uses up inference budget that the point loops need.
@inline foreach_pool_slice(
    threads::Int,
    scope::ThisThreadPool,
    op::O,
    f::F,
    args...;
    mask,
    enumerate,
) where {O, F} =
    isone(threads) ? foreach_slice(ThisThread(), op, f, args...; mask, enumerate) :
    parallelize_over(
        () -> scoped_slice_loop(
            slice_subscope(scope, op, args...), scope, op, f, mask, enumerate, args...,
        ),
        scope,
    )

@inline function foreach_slice(
    scope::DataScope, op::O, f::F, args...; kwargs...,
) where {O, F}
    (mask, enumerate) = slice_loop_flags(; kwargs...)
    return scoped_slice_loop(
        slice_subscope(scope, op, args...), scope, op, f, mask, enumerate, args...,
    )
end

# The loop body lives behind a function barrier so that the subscope is a
# dispatch argument: a run-time slice size would then split a Union of scopes
# once instead of forcing correlated unions of all the argument slices at the
# call to f. The mask and enumerate flags are positional: another keyword layer
# would add a Core.kwcall method and a hidden body method per loop, each
# re-optimized with the whole loop inlined.
#
# Point loops need @simd and an inlined call to f for vectorization (LLVM
# cannot split a flattened CartesianIndices iterator on its own, and the
# inliner gives up on point closures over large broadcasts); without them they
# are several times slower than ordinary Array broadcasts. Non-point slices do
# too much work per slice for vectorization to matter.
#
# NOTE: a function barrier here is not an option: a non-inlined call boxes the
# broadcast expression at least once per slab, allocating inside the tendency
# (68 kB per hyperdiffusion tendency for slab loops alone, 21 MB with point
# loops too).
@inline function scoped_slice_loop(
    subscope, scope, op::O, f::F, mask, enumerate, args...,
) where {O, F}
    indices = subscope_slice_indices(subscope, scope, mask, op, args...)
    scoped_args = reassign_every_arg(subscope, args...)
    @simd_if (op == view && simd_over_indices(indices)) for i in 1:length(indices)
        index = @inbounds indices[i]
        slices = @inbounds slice_every_arg(op, index, scoped_args...)
        @inline enumerate isa Val{true} ? f(index, slices...) : f(slices...)
    end
end

# Every argument, assigned to the subscope that processes its slices; hoisted
# out of the loop (which it commutes with), since reassigning inside the loop
# would rebuild each broadcast expression at every iteration.
@inline reassign_every_arg(subscope, args...) =
    unrolled_tuple_map(Base.Fix2(reassign, subscope), args)

# One slice of every argument. A generated function rather than an unrolled_map
# over a closure (two method instances per slice loop per argument-type
# combination); level/slab/column take a splatted Cartesian index, view takes
# the index itself, and the singleton operator leaves one inferred branch.
@generated slice_every_arg(op::O, index, args::Vararg{Any, N}) where {O, N} = quote
    Base.@_propagate_inbounds_meta
    return Base.Cartesian.@ntuple $N n -> $(
        O === typeof(view) ? :(view(getfield(args, n), index)) :
        :(op(getfield(args, n), Tuple(index)...))
    )
end

# Alternative to scoped_slice_loop that generates ordinary, unfused for loops,
# with every slice processed by the full scope instead of a slice_subscope.
@inline function unfused_slice_loop(scope, op::O, f::F, args...; kwargs...) where {O, F}
    (mask, enumerate) = slice_loop_flags(; kwargs...)
    for index in subscope_slice_indices(scope, scope, mask, op, args...)
        slices = map(arg -> (@inbounds op(arg, Tuple(index)...)), args)
        enumerate isa Val{true} ? f(index, slices...) : f(slices...)
    end
end

"""
    reduce_points(op, arg; [mask], [init])

Generalization of `reduce` that uses `op` to combine all values of a
[`DataLayout`](@ref) or similarly indexable argument assigned to `scope`, by
default the largest available [`DataScope`](@ref) that can access the argument.
A [`DataMask`](@ref) may be used to skip a particular subset of points. The
`init` value must be specified if the `mask` disables every point, if there are
no points in `arg` to begin with, or when reducing with any mask on a GPU,
where a mask can leave whole blocks without active points.
"""
@inline reduce_points(op::O, arg; mask = NoMask(), init...) where {O} =
    reduce_points(DataScope(arg), op, arg; mask, init...)

@inline function reduce_points(scope::ThisThreadPool, op::O, arg; kwargs...) where {O}
    # Reduce on this thread when the pool has one thread, or when there are too few points
    # to divide up, without resolving the pool or claiming any of its threads.
    (isone(default_pool_size()) || length(arg) <= default_pool_size()) &&
        return reduce_points(ThisThread(), op, reassign(arg, ThisThread()); kwargs...)
    T = return_type(op, NTuple{2, eltype(arg)})
    if pool_thread_info() != (0, 0)
        # A reduction nested in another loop gets a fresh results array, since the task's
        # reduction buffer may still be read by an outer reduction that is applying op.
        results = Array{T}(undef, num_threads(scope))
        return reduce_pool_points(results, scope, op, arg; kwargs...)
    end
    threads = resolve_pool_threads()
    try
        isone(threads) &&
            return reduce_points(ThisThread(), op, reassign(arg, ThisThread()); kwargs...)
        results = scoped_array(scope, T, threads; buffer = true)
        return reduce_pool_points(results, scope, op, arg; kwargs...)
    finally
        release_pool_threads()
    end
end

# Reduce over the threads of the current pool loop, storing each thread's share of the
# reduction in results. The results array is allocated by the caller, so that this method
# stays concretely typed whether it is given a view of the task's reduction buffer or a
# fresh array.
@inline function reduce_pool_points(
    results,
    scope::ThisThreadPool,
    op::O,
    arg;
    kwargs...,
) where {O}
    parallelize_over(scope) do
        @inbounds results[thread_rank(scope)] =
            reduce_points(ThisThread(), op, arg; kwargs...)
    end
    return reduce(op, results)
end

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

# All indices a reduction over arg combines, before division among a
# DataScope's threads. Unmasked reductions use eachindex: CPU reductions
# vectorize an order of magnitude better over linear than Cartesian indices.
@inline reduced_point_indices(arg, mask) =
    mask == NoMask() ? eachindex(arg) :
    maskable_slice_indices(DataScope(arg), mask, view, arg)

# Reduce all points assigned to this thread with safe_mapreduce (pairwise
# splitting for logarithmic roundoff, @simd for vectorization); Base's pairwise
# mapreduce has an empty-collection error path that cannot compile in GPU
# kernels.
#
# A thread can be assigned no points (slice_subscope rounds a slice up to a
# power of two, or a mask leaves fewer active points than threads). Such a
# thread folds a duplicate of the first point's value; every scope that can
# leave a thread empty excludes it from its own fold (see num_reduced_threads
# in the CUDA extension). A mask can also leave a whole kernel block empty,
# which the grid-level fold cannot exclude, so masked kernel reductions
# require an init value (enforced in the CUDA extension). With no points at
# all, only an init value can define the result, so empty reductions still
# require one.
@inline function reduce_points(::ThisThread, op::O, arg; mask, init...) where {O}
    all_indices = reduced_point_indices(arg, mask)
    indices = @inbounds subscope_indices(ThisThread(), DataScope(arg), all_indices)
    value(index) = @inbounds arg[index]
    isempty(init) &&
        isempty(indices) &&
        !isempty(all_indices) &&
        return value(@inbounds all_indices[firstindex(all_indices)])
    return safe_mapreduce(value, op, indices; init...)
end

"""
    column_reduce!(op, dest, arg; [mask], [flip], [init])

Use [`foreach_column`](@ref) to combine the levels of each column of `arg` with
`op`, storing the results in corresponding columns of `dest`. Setting `flip` to
`Val(true)` changes the order of reduction from left-associative (default) to
right-associative, and `init` seeds the fold when it is given.
"""
@inline column_reduce!(
    op::O,
    dest,
    arg;
    mask = NoMask(),
    flip = Val(false),
    init...,
) where {O} =
    foreach_column(dest, arg; mask) do dest_column, arg_column
        maybe_reverse = flip isa Val{true} ? reverse : identity
        fill!(dest_column, reduce(op, maybe_reverse(arg_column); init...))
    end

# TODO: Extend this to column_accumulate!

# Convert the value before the fill! loop. Even though setindex! converts at
# every point, the compiler does not hoist the conversion, and filling a Float64
# layout with an Int is measurably slower if the conversion isn't done first.
# The converted value is passed to GPU kernels as parent array entries because
# Int128 and UInt128 fields in kernel arguments crash LLVM's NVPTX backend prior
# to LLVM 20 (llvm/llvm-project#49221). 128-bit integers are only safe in
# registers, like the ones bitcast_struct uses to reconstruct the value.
@inline function Base.fill!(dest::DataLayout, value; kwargs...)
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
            bc_without_axes = Broadcast.Broadcasted(bc.style, bc.f, bc.args)
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

@inline function Base.copyto!(dest::DataLayout, arg::MaybeLazyDataLayout; kwargs...)
    foreach_point(dest, arg; kwargs...) do dest_point, arg_point
        @inbounds dest_point[] = arg_point[]
    end
    call_post_op_callback() && post_op_callback(dest, dest, arg; kwargs...)
    return dest
end

@inline function Base.copyto!(bc::FusedMultiBroadcast; kwargs...)
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
@inline function Base.reduce(op::O, arg::MaybeLazyDataLayout; kwargs...) where {O}
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
# Parent arrays with mismatched dimensions are flattened before being compared.
@inline Base.:(==)(arg1::DataLayout, arg2::DataLayout; mask = NoMask()) =
    size(arg1) == size(arg2) && (
        mask != NoMask() ||
        !(eltype(arg1) == eltype(arg2) && Base.ispacked(eltype(arg1))) ||
        !(layout_type(arg1) == layout_type(arg2) && f_dim(arg1) == f_dim(arg2)) ?
        mapreduce(all ∘ ==, &, arg1, arg2; mask, init = true) :
        ndims(parent(arg1)) == ndims(parent(arg2)) ? parent(arg1) == parent(arg2) :
        stable_view(parent(arg1), :) == stable_view(parent(arg2), :)
    )
@inline Base.:(==)(arg1::MaybeLazyDataLayout, arg2::MaybeLazyDataLayout; kwargs...) =
    size(arg1) == size(arg2) &&
    mapreduce(all ∘ ==, &, arg1, arg2; init = true, kwargs...)
