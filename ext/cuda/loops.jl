function DataLayouts.foreach_slice(::ThisHost, op::O, f::F, args...; kwargs...) where {O, F}
    kernel(args...) = DataLayouts.foreach_slice(ThisKernel(), op, f, args...; kwargs...)
    if DataLayouts.slice_subscope(ThisKernel(), op, args...) == ThisBlock()
        max_slice_points = maximum(Base.Fix1(DataLayouts.inferred_slice_length, op), args)
        threads = min(threads_via_occupancy(kernel, args), max_slice_points)
        blocks = length(DataLayouts.each_slice_index(op, first(args)))
    else
        (; threads, blocks) = config_via_occupancy(kernel, maximum(length, args), args)
    end
    blocks = min(max_resident_blocks(threads), blocks)
    @cuda always_inline = true threads = threads blocks = blocks kernel(args...)
    return nothing
end

# Only save a reduction result to an array from one thread per reduction scope.
is_first_thread_in(scope) = isone(DataLayouts.thread_rank(scope))

# Reduce each block's values, then reduce the results in a single-block kernel.
function DataLayouts.reduce_points(::ThisHost, op::O, arg; kwargs...) where {O}
    function kernel(results, arg)
        result = DataLayouts.reduce_points(ThisBlock(), op, arg; kwargs...)
        if is_first_thread_in(ThisBlock())
            @inbounds results[DataLayouts.partition_rank(ThisKernel())] = result
        end
        return nothing
    end
    T = return_type(op, NTuple{2, eltype(arg)})
    empty_results = DataLayouts.scoped_array(ThisHost(), T, 0)
    (; threads, blocks) = config_via_occupancy(kernel, length(arg), (empty_results, arg))
    num_results = min(max_resident_blocks(threads), blocks)
    results = similar(empty_results, num_results)
    @cuda always_inline = true threads = threads blocks = blocks kernel(results, arg)
    if !isone(num_results)
        threads = min(threads_via_occupancy(kernel, (results, results)), num_results)
        @cuda always_inline = true threads = threads blocks = 1 kernel(results, results)
    end
    return CUDA.@allowscalar @inbounds results[1]
end

# Reduce each warp's values, then reduce the results in the first warp.
function DataLayouts.reduce_points(::ThisBlock, op::O, arg; kwargs...) where {O}
    result = DataLayouts.reduce_points(ThisWarp(), op, arg; kwargs...)
    T = typeof(result)
    results = DataLayouts.scoped_static_array(ThisBlock(), T, MAX_WARPS_PER_BLOCK)
    if is_first_thread_in(ThisWarp())
        @inbounds results[DataLayouts.partition_rank(ThisBlock())] = result
    end
    DataLayouts.synchronize(ThisBlock())
    num_results = DataLayouts.num_partitions(ThisBlock())
    if !isone(num_results)
        result_index = DataLayouts.thread_rank(ThisWarp())
        if isone(DataLayouts.partition_rank(ThisBlock())) && result_index <= num_results
            @inbounds result = results[result_index]
            result = shuffle_reduce(ThisWarp(), op, result, num_results)
            if is_first_thread_in(ThisWarp())
                @inbounds results[1] = result
            end
        end
        DataLayouts.synchronize(ThisBlock())
    end
    return @inbounds results[1]
end

DataLayouts.reduce_points(scope::ThisSubBlock, op::O, arg; kwargs...) where {O} =
    DataLayouts.num_threads(scope) <= THREADS_PER_WARP ?
    shuffle_reduce(scope, op, DataLayouts.reduce_points(ThisThread(), op, arg; kwargs...)) :
    DataLayouts.reduce_points(
        ThisWarp(),
        op,
        DataLayouts.reassign(arg, ThisWarp());
        kwargs...,
    )

# Use warp shuffles to perform binary tree reductions when num_threads is small.
function shuffle_reduce(scope, op::O, value, num_threads = THREADS_PER_WARP) where {O}
    DataLayouts.is_subscope(scope, ThisWarp()) ||
        throw(ArgumentError(DataLayouts.invalid_subscope_string(scope, ThisWarp())))
    num_threads <= THREADS_PER_WARP ||
        throw(ArgumentError("Number of threads is too large for warp shuffle"))
    warp_thread_mask = CUDA.FULL_MASK << (THREADS_PER_WARP - num_threads)
    n = DataLayouts.num_threads(scope)
    log2_n = 8 * sizeof(n) - Base.ctlz_int(n) - 1
    for offset in ntuple(Base.Fix1(>>, n), Val(log2_n)) # n ÷ 2, n ÷ 4, ..., 1
        if num_threads < THREADS_PER_WARP
            num_threads < xor(DataLayouts.thread_rank(ThisWarp()), offset) && continue
        end
        value = op(value, CUDA.shfl_xor_sync(warp_thread_mask, value, offset))
    end
    return value
end
