function DataLayouts.foreach_slice(
    ::ThisHost,
    op::O,
    f::F,
    args...;
    max_waves = nothing,
    kwargs...,
) where {O, F}
    check_device_assumptions()

    # Capture the kwargs as a NamedTuple, whose names are type parameters. The
    # Pairs structure of kwargs stores its names in a Tuple of Symbols, which
    # cannot be passed to a kernel because Symbols are not bitstypes.
    kernel_kwargs = values(kwargs)
    kernel_function(args...) =
        DataLayouts.foreach_slice(ThisKernel(), op, f, args...; kernel_kwargs...)

    (; threads, blocks) =
        if DataLayouts.slice_subscope(ThisKernel(), op, args...) == ThisBlock()
            max_slice_points = maximum(Base.Fix1(DataLayouts.num_slice_points, op), args)
            max_slices = length(DataLayouts.each_slice_index(op, first(args)))
            launch_configuration(
                kernel_function,
                args,
                max_slice_points,
                max_slices;
                max_waves,
            )
        else
            # Extra threads run empty loops, so max_points isn't a strict limit.
            max_points = maximum(length, args)
            launch_configuration(
                kernel_function,
                args,
                max_points;
                max_waves,
                strict = false,
            )
        end
    auto_launch!(kernel_function, args; threads_s = threads, blocks_s = blocks)
end

# Only save a reduction result to an array from one thread per reduction scope.
is_first_thread_in(scope) = isone(DataLayouts.thread_rank(scope))

# Reduce each block's values, then reduce the results in a single-block kernel.
function DataLayouts.reduce_points(::ThisHost, op::O, arg; kwargs...) where {O}
    check_device_assumptions()

    kernel_kwargs = values(kwargs)
    function kernel_function(results, finished_blocks, arg, num_blocks)
        result = DataLayouts.reduce_points(ThisBlock(), op, arg; kernel_kwargs...)
        
        block_idx = DataLayouts.partition_rank(ThisKernel())
        if is_first_thread_in(ThisBlock())
            @inbounds results[block_idx] = result
        end
        
        # Ensure all writes to `results` are globally visible
        CUDA.threadfence()
        
        # Determine if this is the last block
        is_last = DataLayouts.scoped_static_array(ThisBlock(), Bool, 1)
        if is_first_thread_in(ThisBlock())
            old = CUDA.atomic_add!(pointer(finished_blocks, 1), Int32(1))
            @inbounds is_last[1] = (old == (num_blocks - 1))
        end
        DataLayouts.synchronize(ThisBlock())
        
        if @inbounds is_last[1]
            warp_idx = DataLayouts.subscope_rank(ThisWarp(), ThisBlock())
            if isone(warp_idx)
                lane_idx = DataLayouts.thread_rank(ThisWarp())
                num_lanes = THREADS_PER_WARP
                
                if lane_idx <= num_blocks
                    local_val = @inbounds results[lane_idx]
                    i = lane_idx + num_lanes
                    while i <= num_blocks
                        local_val = op(local_val, @inbounds results[i])
                        i += num_lanes
                    end
                    
                    num_active = min(num_blocks, num_lanes)
                    final_val = shuffle_reduce(ThisWarp(), op, local_val, num_active)
                    
                    if is_first_thread_in(ThisWarp())
                        @inbounds results[1] = final_val
                    end
                end
            end
        end
        return nothing
    end

    T = return_type(op, NTuple{2, eltype(arg)})
    results = DataLayouts.scoped_array(ThisHost(), T, 0; buffer = true)
    
    # We must call auto_launch! with 0 blocks first to get launch_configuration
    (; threads, blocks) = launch_configuration(kernel_function, (results, CUDA.zeros(Int32, 1), arg, 1), length(arg))
    
    results = DataLayouts.scoped_array(ThisHost(), T, blocks; buffer = true)
    finished_blocks = DataLayouts.scoped_array(ThisHost(), Int32, 1; buffer = true)
    CUDA.fill!(finished_blocks, Int32(0))
    
    auto_launch!(kernel_function, (results, finished_blocks, arg, Int32(blocks)); threads_s = threads, blocks_s = blocks)
    
    return CUDA.@allowscalar @inbounds results[1]
end

# Reduce a warp or sub-warp with warp shuffles, limited to active threads since
# inactive threads have undefined results. For multi-warp scopes, first reduce
# each warp, then reduce the results in the first warp.
DataLayouts.reduce_points(scope::ThisCooperativeGroup, op::O, arg; kwargs...) where {O} =
    if scope != ThisBlock() && DataLayouts.num_threads(scope) <= THREADS_PER_WARP
        thread_result =
            DataLayouts.reduce_points(DataLayouts.ThisThread(), op, arg; kwargs...)
        shuffle_reduce(scope, op, thread_result, num_active_threads(scope))
    else
        num_results = DataLayouts.num_subscopes(ThisWarp(), scope)
        max_results = scope == ThisBlock() ? MAX_WARPS_PER_BLOCK : num_results
        warp_index = DataLayouts.subscope_rank(ThisWarp(), scope)
        warp_result = DataLayouts.reduce_points(ThisWarp(), op, arg; kwargs...)
        results = DataLayouts.scoped_static_array(scope, typeof(warp_result), max_results)
        if is_first_thread_in(ThisWarp())
            @inbounds results[warp_index] = warp_result
        end
        DataLayouts.synchronize(scope)
        if !isone(num_results)
            if isone(warp_index)
                @inbounds warp_result = results[DataLayouts.thread_rank(ThisWarp())]
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
    for offset in ntuple(Base.Fix1(>>, DataLayouts.num_threads(scope)), Val(num_offsets))
        shuffled_value = CUDA.shfl_xor_sync(CUDA.FULL_MASK >> num_inactive, value, offset)
        if thread_index <= num_values && xor(thread_index - 1, offset) + 1 <= num_values
            value = op(value, shuffled_value)
        end
    end
    return value
end

# Extend CUDA's warp shuffle intrinsics to support AutoBroadcasters, recursively
# shuffling each value that appears in a multi-component reduction.
CUDA.shfl_recurse(op::O, x::Utilities.AutoBroadcaster) where {O} =
    Utilities.AutoBroadcaster(UnrolledUtilities.unrolled_map(op, Utilities.unwrap(x)))
