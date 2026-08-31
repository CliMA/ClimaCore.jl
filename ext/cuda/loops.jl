# Only run a fused foreach_slice kernel when every slice has an inferrable size.
has_inferred_slice_size(op::O, arg) where {O} =
    Val{true} == Utilities.return_type(Tuple{typeof(arg)}) do arg
        slice = op(arg, Tuple(first(DataLayouts.each_slice_index(op, arg)))...)
        Val(DataLayouts.has_inferred_size(slice))
    end

DataLayouts.needs_loop_setup(::ThisHost) = true
DataLayouts.foreach_slice(scope::ThisHost, op::O, f::F, args...; kwargs...) where {O, F} =
    if !unrolled_all(Base.Fix1(has_inferred_slice_size, op), args)
        DataLayouts.unfused_slice_loop(scope, op, f, args...; kwargs...)
    else
        check_device_assumptions()
        kernel_kwargs = values(kwargs) # capture kwargs as NamedTuple (Pairs isn't isbitstype)
        kernel_function(args...) = DataLayouts.foreach_slice(
            ThisKernel(),
            op,
            f,
            args...;
            kernel_kwargs...,
        )

        # A rank can own no elements; there are then no slices to launch
        # over, and num_slice_points cannot slice an empty layout.
        isempty(DataLayouts.each_slice_index(op, first(args))) && return

        # partition(::ThisKernel) descends directly to sub-blocks, so
        # slice_subscope never returns ThisBlock itself.
        subscope = DataLayouts.slice_subscope(ThisKernel(), op, args...)
        if subscope isa ThisSubBlock
            # Each sub-block gets one slice. The block size is capped at the
            # value that sizes the sub-blocks' shared memory (see
            # scoped_static_array) and constrained to whole sub-blocks: a
            # partially populated sub-block skips its missing threads' points
            # (see the whole subscopes invariant in subscope_launch_threads).
            max_slices = length(DataLayouts.each_slice_index(op, first(args)))
            subblock_threads = Int(DataLayouts.num_threads(subscope))
            max_block_threads = DataLayouts.subscope_launch_threads(
                subscope,
                max_subblock_launch_threads(subscope),
            )
            slices_per_block = max_block_threads ÷ subblock_threads
            (; threads, blocks) = launch_configuration(
                kernel_function, args, max_block_threads,
                cld(max_slices, slices_per_block); granularity = subblock_threads,
            )
        else
            # Extra threads run empty loops, so max_points isn't a strict limit.
            max_points = maximum(length, args)
            (; threads, blocks) =
                launch_configuration(kernel_function, args, max_points; strict = false)
        end
        auto_launch!(kernel_function, args; threads_s = threads, blocks_s = blocks)
    end

# Only save a reduction result to an array from one thread per reduction scope.
is_first_thread_in(scope) = isone(DataLayouts.thread_rank(scope))

# The grid-synchronization counter is cached per task under its own key
# (task_reduction_buffer is keyed by array type, so the counter would alias any
# Int32 results buffer). Reductions read their result back before returning,
# so consecutive reductions on a task never share a live counter.
function reduction_sync_counter()
    storage = current_task().storage
    if !isnothing(storage)
        counter =
            get(storage::IdDict{Any, Any}, :climacore_reduction_sync_counter, nothing)
        isnothing(counter) || return counter::CUDA.CuVector{Int32}
    end
    counter = CUDA.CuVector{Int32}(undef, 1)
    task_local_storage(:climacore_reduction_sync_counter, counter)
    return counter
end

# Reduce each block's values, then finish in the last block (the one whose
# atomic increment of the grid-synchronization counter returns num_blocks - 1),
# so a reduction is one kernel launch. The last block's first warp folds the
# block results in registers with warp shuffles, which needs no shared memory
# and supports arbitrarily wide element types.
function DataLayouts.reduce_points(scope::ThisHost, op::O, arg; mask, init...) where {O}
    check_device_assumptions()
    # A mask can leave whole blocks without active points, and such a block's
    # placeholder result cannot be excluded from the grid-level fold below, so
    # only a neutral init value makes masked reductions well-defined.
    isempty(init) && !(mask isa DataLayouts.NoMask) &&
        throw(
            ArgumentError("masked GPU reductions require an `init` value"),
        )
    kernel_kwargs = (; mask, init...) # capture kwargs as NamedTuple (Pairs isn't isbitstype)
    function kernel_function(results, finished_blocks, arg, num_blocks)
        result = DataLayouts.reduce_points(ThisBlock(), op, arg; kernel_kwargs...)
        block_idx = DataLayouts.subscope_rank(ThisBlock(), ThisKernel())
        if is_first_thread_in(ThisBlock())
            @inbounds results[block_idx] = result
        end

        # Make this block's result visible to the whole device before its
        # counter increment, so that the last block observes every result.
        CUDA.threadfence()
        is_last = DataLayouts.scoped_static_array(ThisBlock(), Bool, 1)
        if is_first_thread_in(ThisBlock())
            old = CUDA.atomic_add!(pointer(finished_blocks, 1), Int32(1))
            @inbounds is_last[1] = (old == num_blocks - Int32(1))
        end
        DataLayouts.synchronize(ThisBlock())

        if @inbounds is_last[1]
            # Pair the fence above: order the loads of the other blocks'
            # results after this block's counter increment.
            CUDA.threadfence()
            if isone(DataLayouts.subscope_rank(ThisWarp(), ThisBlock()))
                lane_idx = Int(DataLayouts.thread_rank(ThisWarp()))
                num_lanes = THREADS_PER_WARP
                # Every lane of the warp must reach the shuffles below, so
                # lanes without their own block result load a duplicate of the
                # first one, which num_active excludes from the fold.
                local_val = @inbounds results[min(lane_idx, Int(num_blocks))]
                i = lane_idx + num_lanes
                while i <= num_blocks
                    local_val = op(local_val, @inbounds results[i])
                    i += num_lanes
                end
                num_active = min(Int(num_blocks), num_lanes)
                final_val = shuffle_reduce(ThisWarp(), op, local_val, num_active)
                if is_first_thread_in(ThisWarp())
                    @inbounds results[1] = final_val
                end
            end
        end
        return nothing
    end

    T = return_type(op, NTuple{2, eltype(arg)})
    results = DataLayouts.scoped_array(scope, T, 0; buffer = true)
    finished_blocks = reduction_sync_counter()
    (; threads, blocks) = launch_configuration(
        kernel_function,
        (results, finished_blocks, arg, Int32(1)),
        length(arg),
    )
    results = DataLayouts.scoped_array(scope, T, blocks; buffer = true)

    fill!(finished_blocks, Int32(0))
    auto_launch!(
        kernel_function,
        (results, finished_blocks, arg, Int32(blocks));
        threads_s = threads,
        blocks_s = blocks,
    )
    return CUDA.@allowscalar @inbounds results[1]
end

# Number of threads in a scope assigned at least one point of arg (fewer than
# all active threads when slice_subscope rounded the slice up, or a mask left
# fewer active points than threads). Threads take strided subsets
# rank:n:num_points, so the threads with points are a prefix of the scope,
# which the num_values guard in shuffle_reduce requires.
@inline function num_reduced_threads(scope, arg, mask)
    num_points = length(DataLayouts.reduced_point_indices(arg, mask))
    first_rank =
        Int(DataLayouts.thread_rank(DataLayouts.DataScope(arg))) -
        Int(DataLayouts.thread_rank(scope))
    return clamp(num_points - first_rank, 0, Int(num_active_threads(scope)))
end

# Reduce a warp or sub-warp with warp shuffles, limited to the threads that
# were assigned points (other threads' results are undefined or duplicates).
# For multi-warp scopes, reduce each warp, then the results in the first warp.
DataLayouts.reduce_points(
    scope::ThisCooperativeGroup, op::O, arg; mask, init...,
) where {O} =
    if scope != ThisBlock() && DataLayouts.num_threads(scope) <= THREADS_PER_WARP
        thread_result =
            DataLayouts.reduce_points(DataLayouts.ThisThread(), op, arg; mask, init...)
        shuffle_reduce(scope, op, thread_result, num_reduced_threads(scope, arg, mask))
    else
        num_warps = DataLayouts.num_subscopes(ThisWarp(), scope)
        max_results = scope == ThisBlock() ? MAX_WARPS_PER_BLOCK : num_warps
        warp_index = DataLayouts.subscope_rank(ThisWarp(), scope)
        warp_result = DataLayouts.reduce_points(ThisWarp(), op, arg; mask, init...)
        results = DataLayouts.scoped_static_array(scope, typeof(warp_result), max_results)
        if is_first_thread_in(ThisWarp())
            @inbounds results[warp_index] = warp_result
        end
        DataLayouts.synchronize(scope)
        # Empty warps hold duplicates rather than partial results, so only
        # warps that reduced points take part in the fold; at least one entry
        # is always folded so a scope with no points still returns its first
        # warp's init-seeded value.
        num_results = max(cld(num_reduced_threads(scope, arg, mask), THREADS_PER_WARP), 1)
        if !isone(num_results)
            if isone(warp_index)
                # Every lane of the first warp must reach the shuffles in
                # shuffle_reduce, so surplus lanes load a duplicate of the last
                # result instead of an entry that no warp ever wrote.
                lane_index = min(Int(DataLayouts.thread_rank(ThisWarp())), num_results)
                @inbounds warp_result = results[lane_index]
                final_result = shuffle_reduce(ThisWarp(), op, warp_result, num_results)
                if is_first_thread_in(ThisWarp())
                    @inbounds results[1] = final_result
                end
            end
            DataLayouts.synchronize(scope)
        end
        @inbounds results[1]
    end

# Use the scope type to generate the number of pairwise reductions, log2(N), in
# the compiler, without needing to rely on constant propagation in GPU kernels.
@generated num_reductions(::ThisSubBlock{N}) where {N} =
    8 * sizeof(N) - leading_zeros(N) - 1

# Binary-tree reduction over first num_values threads: all active lanes (nonzero
# lower bits in the mask) exchange data, but ranks above num_values are ignored.
function shuffle_reduce(scope, op::O, value, num_values) where {O}
    num_offsets = num_reductions(scope)
    num_inactive = THREADS_PER_WARP - num_active_threads(ThisWarp())
    thread_index = DataLayouts.thread_rank(scope)
    for offset in
        ntuple(Base.Fix1(>>, DataLayouts.num_threads(scope)) ∘ Int32, Val(num_offsets))
        # shfl_recurse fails if `offset` is not explicitly converted to an Int32
        shuffled_value =
            CUDA.shfl_xor_sync(CUDA.FULL_MASK >> num_inactive, value, offset)
        if thread_index <= num_values && xor(thread_index - 1, offset) + 1 <= num_values
            value = op(value, shuffled_value)
        end
    end
    return value
end

# Extend CUDA's warp shuffle intrinsics to support AutoBroadcasters and Tensors, recursively
# shuffling each value that appears in a multi-component reduction.
CUDA.shfl_recurse(
    op::O,
    x::T,
) where {O, T <: Union{ClimaCore.Geometry.Tensor, Utilities.AutoBroadcaster}} = map(op, x)
