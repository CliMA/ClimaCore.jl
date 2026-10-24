has_inferred_slice_size(op::O, arg) where {O} =
    Val{true} == Utilities.return_type(Tuple{typeof(arg)}) do arg
        slice = op(arg, Tuple(first(DataLayouts.each_slice_index(op, arg)))...)
        Val(DataLayouts.has_inferred_size(slice))
    end

DataLayouts.needs_loop_setup(::ThisHost) = true

# Only run a fused foreach_slice kernel when every slice has an inferrable size.
# Also, avoid launching fused loops on MPI ranks that do not own any elements.
@inline DataLayouts._foreach_slice(
    scope::ThisHost,
    op::O,
    f::F,
    mask,
    enumerate,
    args...,
) where {O, F} =
    if !unrolled_all(Base.Fix1(has_inferred_slice_size, op), args)
        DataLayouts.unfused_slice_loop(scope, op, f, args...; mask, enumerate)
    elseif !isempty(DataLayouts.each_slice_index(op, first(args)))
        check_device_assumptions()
        subscope = DataLayouts.slice_subscope(ThisKernel(), op, args...)
        function kernel_function(compact_kernel_args...)
            kernel_args = Grids.toggle_compact_args(compact_kernel_args...)
            DataLayouts.scoped_slice_loop(
                subscope,
                ThisKernel(),
                op,
                f,
                mask,
                enumerate,
                kernel_args...,
            )
        end

        # partition(::ThisKernel) descends directly to sub-blocks, so
        # slice_subscope never returns ThisBlock itself.
        if subscope isa ThisSubBlock
            # Each sub-block gets one slice. The block size is capped at the
            # value that sizes the sub-blocks' shared memory (see
            # scoped_static_array) and constrained to whole sub-blocks: a
            # partially populated sub-block skips its missing threads' points
            # (see the whole subscopes invariant in subscope_launch_threads).
            max_slices = length(DataLayouts.each_slice_index(op, first(args)))
            check_index_count(max_slices)
            subblock_threads = Int(DataLayouts.num_threads(subscope))
            max_block_threads = DataLayouts.subscope_launch_threads(
                subscope,
                max_subblock_launch_threads(subscope),
            )
            max_blocks = cld(max_slices, max_block_threads ÷ subblock_threads)
            config =
                kernel -> launch_configuration(
                    kernel, max_block_threads, max_blocks;
                    granularity = subblock_threads,
                )
        else
            # Extra threads run empty loops, so max_points isn't a strict limit.
            max_points = maximum(length, args)
            check_index_count(max_points)
            config = kernel -> launch_configuration(kernel, max_points; strict = false)
        end
        auto_launch!(kernel_function, Grids.toggle_compact_args(args...); config)
    end

# Only save a reduction result to an array from one thread per reduction scope.
is_first_thread_in(scope) = isone(DataLayouts.thread_rank(scope))

# The grid-synchronization counter is cached per task under its own key
# (task_reduction_buffer is keyed by array type, so the counter would alias any
# UInt32 results buffer). Reductions read their result back before returning,
# so consecutive reductions on a task never share a live counter.
function reduction_sync_counter()
    storage = current_task().storage
    if !isnothing(storage)
        counter =
            get(storage::IdDict{Any, Any}, :climacore_reduction_sync_counter, nothing)
        isnothing(counter) || return counter::CUDA.CuVector{UInt32}
    end
    counter = CUDA.CuVector{UInt32}(undef, 1)
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
            old = CUDA.atomic_add!(pointer(finished_blocks, 1), UInt32(1))
            @inbounds is_last[1] = (old == num_blocks - UInt32(1))
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
                final_val = parallel_reduce(ThisWarp(), op, local_val, num_active)
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
    # The launch needs results sized by the block count, so the kernel is compiled before it.
    kernel = CUDA.@cuda always_inline = true launch = false kernel_function(
        results, finished_blocks, arg, UInt32(1),
    )
    (; threads, blocks) = launch_configuration(kernel, length(arg))
    results = DataLayouts.scoped_array(scope, T, blocks; buffer = true)

    fill!(finished_blocks, UInt32(0))
    auto_launch!(
        kernel_function,
        (results, finished_blocks, arg, UInt32(blocks));
        threads_s = threads,
        blocks_s = blocks,
    )
    return CUDA.@allowscalar @inbounds results[1]
end

# Construct a scope with at least one thread for each warp in the given scope.
@inline warps_to_threads(::ThisBlock) = ThisWarp()
@inline warps_to_threads(::ThisSubBlock{N}) where {N} =
    ThisSubBlock{cld(N, THREADS_PER_WARP)}()

# Reduce each thread's points, then combine the results of the threads that were
# assigned points (other threads' results are undefined or duplicates). Not all
# active threads are assigned points when slice_subscope rounds the slice up, or
# when applying a mask leaves fewer active points than available threads.
function DataLayouts.reduce_points(
    scope::ThisCooperativeGroup, op::O, arg; mask, init...,
) where {O}
    first_rank_offset =
        Int(DataLayouts.thread_rank(DataLayouts.DataScope(arg))) -
        Int(DataLayouts.thread_rank(scope))
    num_points = length(DataLayouts.reduced_point_indices(arg, mask))
    num_partial_results =
        clamp(num_points - first_rank_offset, 0, Int(num_active_threads(scope)))
    partial_result =
        DataLayouts.reduce_points(DataLayouts.ThisThread(), op, arg; mask, init...)
    return parallel_reduce(scope, op, partial_result, num_partial_results)
end

# Combination of the first num_values threads' values, returned to every thread.
# Multi-warp scopes reduce each warp, and then the warp results, which are read
# from shared memory into the threads of a subscope that has one thread per warp
# in the full scope.
@inline parallel_reduce(scope::ThisCooperativeGroup, op::O, value, num_values) where {O} =
    if scope != ThisBlock() && shuffles_in_warp(scope)
        result_in_first_thread = shuffle_reduce(scope, op, value, num_values)
        shuffle_point_value(scope, result_in_first_thread, 1)
    else
        warp_index = Int(DataLayouts.subscope_rank(ThisWarp(), scope))
        num_warp_values =
            clamp(num_values - (warp_index - 1) * THREADS_PER_WARP, 0, THREADS_PER_WARP)
        warp_result = shuffle_reduce(ThisWarp(), op, value, num_warp_values)

        subscope = warps_to_threads(scope)
        max_num_warps = DataLayouts.static_num_threads(subscope)
        warp_results =
            DataLayouts.scoped_static_array(scope, typeof(warp_result), max_num_warps)
        is_first_thread_in(ThisWarp()) && (@inbounds warp_results[warp_index] = warp_result)
        DataLayouts.synchronize(scope)

        num_warps = max(cld(num_values, THREADS_PER_WARP), 1)
        lane_index = min(Int(DataLayouts.thread_rank(subscope)), num_warps)
        lane_value = @inbounds warp_results[lane_index]
        DataLayouts.synchronize(scope)
        result_in_first_thread = shuffle_reduce(subscope, op, lane_value, num_warps)
        shuffle_point_value(subscope, result_in_first_thread, 1)
    end

# Scan a cooperative group's points in chunks of one point per thread, using
# each chunk's result to initialize the next one. Threads without a point in the
# last chunk read a copy of the last point in arg, which the scans below ignore.
@inline DataLayouts.scan_points(
    scope::ThisCooperativeGroup, f::F, op::O, arg, reverse, init,
) where {F, O} =
    if !DataLayouts.is_associative(op)
        Base.@invoke DataLayouts.scan_points(scope::Any, f, op, arg, reverse, init)
    else
        n = length(arg)
        num_threads = Int(DataLayouts.num_threads(scope))

        @inline function scan_points_uncoarsened(init, chunk_index)
            index = (chunk_index - 1) * num_threads + Int(DataLayouts.thread_rank(scope))
            num_values = min(num_threads, n - (chunk_index - 1) * num_threads)
            (partial_result, final_result) =
                parallel_scan(scope, op, @inbounds(arg[min(index, n)]), num_values, reverse)
            index <= n &&
                f(index, init isa NoInit ? partial_result : op(init, partial_result))
            return init isa NoInit ? final_result : op(init, final_result)
        end

        num_chunks = cld(n, num_threads)
        result = scan_points_uncoarsened(init, reverse ? num_chunks : 1)
        for iter in 2:num_chunks
            result = scan_points_uncoarsened(result, reverse ? num_chunks + 1 - iter : iter)
        end
        result
    end

# Inclusive scan of the first num_values threads' values in order of rank (or in
# reverse order), and the combination of all of their values. Multi-warp scopes
# scan each warp, and then the warp results, which are read from shared memory
# into the threads of a subscope that has one thread per warp in the full scope.
@inline parallel_scan(
    scope::ThisCooperativeGroup, op::O, value, num_values, reverse,
) where {O} =
    if scope != ThisBlock() && shuffles_in_warp(scope)
        partial_result = shuffle_scan(scope, op, value, num_values, reverse)
        final_result_index = reverse ? 1 : max(num_values, 1)
        final_result = shuffle_point_value(scope, partial_result, final_result_index)
        (partial_result, final_result)
    else
        warp_index = Int(DataLayouts.subscope_rank(ThisWarp(), scope))
        num_warp_values =
            clamp(num_values - (warp_index - 1) * THREADS_PER_WARP, 0, THREADS_PER_WARP)
        partial_warp_result = shuffle_scan(ThisWarp(), op, value, num_warp_values, reverse)
        warp_result_index = reverse ? 1 : max(num_warp_values, 1)
        warp_result =
            shuffle_point_value(ThisWarp(), partial_warp_result, warp_result_index)

        subscope = warps_to_threads(scope)
        max_num_warps = DataLayouts.static_num_threads(subscope)
        warp_results =
            DataLayouts.scoped_static_array(scope, typeof(warp_result), max_num_warps)
        is_first_thread_in(ThisWarp()) && (@inbounds warp_results[warp_index] = warp_result)
        DataLayouts.synchronize(scope)

        num_warps = max(cld(num_values, THREADS_PER_WARP), 1)
        lane_index = min(Int(DataLayouts.thread_rank(subscope)), num_warps)
        lane_value =
            @inbounds warp_results[reverse ? num_warps + 1 - lane_index : lane_index]
        DataLayouts.synchronize(scope)
        prefix = shuffle_scan(subscope, op, lane_value, num_warps, false)

        partial_result_prefix_index = reverse ? num_warps - warp_index : warp_index - 1
        partial_result_prefix =
            shuffle_point_value(subscope, prefix, max(partial_result_prefix_index, 1))
        partial_result =
            iszero(partial_result_prefix_index) ? partial_warp_result :
            op(partial_result_prefix, partial_warp_result)
        final_result = shuffle_point_value(subscope, prefix, num_warps)
        (partial_result, final_result)
    end
