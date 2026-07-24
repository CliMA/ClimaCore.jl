function DataLayouts.foreach_slice(::ThisHost, op::O, f::F, args...; kwargs...) where {O, F}
    check_device_assumptions()
    # Capture the kwargs as a NamedTuple, whose names are type parameters. The
    # Pairs structure of kwargs stores its names in a Tuple of Symbols, which
    # cannot be passed to a kernel because Symbols are not bitstypes.
    nt_kwargs = values(kwargs)
    kernel(args...) = DataLayouts.foreach_slice(ThisKernel(), op, f, args...; nt_kwargs...)
    if DataLayouts.slice_subscope(ThisKernel(), op, args...) == ThisBlock()
        max_slice_points = maximum(Base.Fix1(DataLayouts.num_slice_points, op), args)
        threads = min(threads_via_occupancy(kernel, args), max_slice_points)
        blocks = length(DataLayouts.each_slice_index(op, first(args)))
    else
        (; threads, blocks) = config_via_occupancy(kernel, maximum(length, args), args)
    end
    blocks = min(max_resident_blocks(threads), blocks)
    auto_launch!(kernel, args; threads_s = threads, blocks_s = blocks)
    return nothing
end

# Only save a reduction result to an array from one thread per reduction scope.
is_first_thread_in(scope) = isone(DataLayouts.thread_rank(scope))

# Reduce each block's values, then reduce the results in a single-block kernel.
function DataLayouts.reduce_points(::ThisHost, op::O, arg; kwargs...) where {O}
    check_device_assumptions()
    nt_kwargs = values(kwargs)
    function kernel(results, arg)
        result = DataLayouts.reduce_points(ThisBlock(), op, arg; nt_kwargs...)
        if is_first_thread_in(ThisBlock())
            @inbounds results[DataLayouts.partition_rank(ThisKernel())] = result
        end
        return nothing
    end
    T = return_type(op, NTuple{2, eltype(arg)})
    empty_results = DataLayouts.scoped_array(ThisHost(), T, 0)
    # Launch at most one thread per point, so every thread's strided range of
    # indices is nonempty. Threads without values would need warp-shuffle
    # placeholders, which reductions without init values (like min) do not have.
    max_threads = threads_via_occupancy(kernel, (empty_results, arg))
    threads = min(length(arg), max_threads)
    blocks = max(fld(length(arg), threads), 1)
    num_results = min(max_resident_blocks(threads), blocks)
    results = similar(empty_results, num_results)
    auto_launch!(kernel, (results, arg); threads_s = threads, blocks_s = num_results)
    if !isone(num_results)
        threads = min(threads_via_occupancy(kernel, (results, results)), num_results)
        auto_launch!(kernel, (results, results); threads_s = threads, blocks_s = 1)
    end
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
