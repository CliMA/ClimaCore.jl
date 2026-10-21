has_inferred_slice_size(op::O, arg) where {O} =
    Val{true} == Utilities.return_type(Tuple{typeof(arg)}) do arg
        index = first(DataLayouts.each_slice_index(op, arg))
        slice = DataLayouts.slice_arg(op, arg, Tuple(index)...)
        Val(DataLayouts.has_inferred_size(slice))
    end

DataLayouts.needs_loop_setup(::ThisHost) = true

# A fused loop's kernel needs inferrable slices, and its arguments must fit in the parameter
# memory that CUDA.jl allows: 4 KiB, or 32 KiB for PTX 8.1+ on Volta and newer devices.
# The size is inferred, so the device is only queried for arguments larger than 4 KiB.
function DataLayouts.fits_slice_loop(
    ::ThisHost,
    op::O,
    f::F,
    mask::M,
    args...,
) where {O, F, M}
    unrolled_all(Base.Fix1(has_inferred_slice_size, op), args) || return false
    T = Core.Compiler.return_type(kernel_arguments, Tuple{F, M, map(typeof, args)...})
    isconcretetype(T) || return false
    # The margin covers the KernelState argument and alignment padding.
    return sizeof(T) + 64 <= 4096 || sizeof(T) + 64 <= max_kernel_parameter_bytes()
end
# The arguments of a fused loop's kernel, including the mask that its function captures.
kernel_arguments(f, mask, args...) =
    CUDA.cudaconvert((f, mask, Grids.toggle_compact_args(args...)...))
function max_kernel_parameter_bytes()
    compiler = isdefined(CUDA, :compiler_config) ? CUDA : CUDA.CUDACore # moved in CUDA 6
    ptx = compiler.compiler_config(CUDA.device()).params.ptx
    return CUDA.capability(CUDA.device()) >= v"7.0" && ptx >= v"8.1" ? 32764 : 4096
end

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
                final_val = scan_threads(ThisWarp(), op, local_val, num_active, false)[2]
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

# Number of threads in a scope assigned at least one point of arg (fewer than
# all active threads when slice_subscope rounded the slice up, or a mask left
# fewer active points than threads). Threads take strided subsets
# rank:n:num_points, so the threads with points are a prefix of the scope,
# which the num_values guard in scan_threads requires.
@inline function num_reduced_threads(scope, arg, mask)
    num_points = length(DataLayouts.reduced_point_indices(arg, mask))
    first_rank =
        Int(DataLayouts.thread_rank(DataLayouts.DataScope(arg))) -
        Int(DataLayouts.thread_rank(scope))
    return clamp(num_points - first_rank, 0, Int(num_active_threads(scope)))
end

# Reduce each thread's points, then combine the results of the threads that were assigned
# points (other threads' results are undefined or duplicates) like a scan over them.
DataLayouts.reduce_points(
    scope::ThisCooperativeGroup, op::O, arg; mask, init...,
) where {O} = scan_threads(
    scope,
    op,
    DataLayouts.reduce_points(DataLayouts.ThisThread(), op, arg; mask, init...),
    num_reduced_threads(scope, arg, mask),
    false,
)[2]

# Use the scope type to generate the number of pairwise reductions, log2(N), in
# the compiler, without needing to rely on constant propagation in GPU kernels.
@generated num_reductions(::ThisSubBlock{N}) where {N} =
    8 * sizeof(N) - leading_zeros(N) - 1

# Scan a cooperative group's points in rounds of one point per thread, carrying the
# combination of each round's values into the next. Threads without a point in the last
# round read a duplicate of the last point, which the scans below exclude.
@inline function DataLayouts.scan_points(
    store!::S,
    scope::ThisCooperativeGroup,
    op::DataLayouts.AssociativeOperator,
    arg,
    init,
    flip,
) where {S}
    n = length(arg)
    num_rounds = cld(n, Int(DataLayouts.num_threads(scope)))
    carry = scan_round(store!, scope, op, arg, init, flip ? num_rounds : 1, flip)
    for round in 2:num_rounds
        carry = scan_round(
            store!, scope, op, arg, carry, flip ? num_rounds + 1 - round : round, flip,
        )
    end
    return carry
end

@inline function scan_round(store!::S, scope, op::O, arg, carry, round, flip) where {S, O}
    (n, num_threads) = (length(arg), Int(DataLayouts.num_threads(scope)))
    index = (round - 1) * num_threads + Int(DataLayouts.thread_rank(scope))
    num_values = min(num_threads, n - (round - 1) * num_threads)
    (scan, total) = scan_threads(scope, op, @inbounds(arg[min(index, n)]), num_values, flip)
    index <= n && store!(index, carry isa NoInit ? scan : op(carry, scan))
    return carry isa NoInit ? total : op(carry, total)
end

# Inclusive scan of the first num_values threads' values in order of rank (reverse order when
# flip), along with the combination of all of their values (the first value if there are
# none). Multi-warp scopes scan each warp, and then the warp totals, which every warp reads
# from shared memory into the lanes of a sub-warp with one lane per warp of the scope.
@inline function scan_threads(
    scope::ThisCooperativeGroup,
    op::O,
    value,
    num_values,
    flip,
) where {O}
    if scope != ThisBlock() && shuffles_in_warp(scope)
        scan = shuffle_scan(scope, op, value, num_values, flip)
        return (scan, shuffle_point_value(scope, scan, flip ? 1 : max(num_values, 1)))
    end
    warp_index = Int(DataLayouts.subscope_rank(ThisWarp(), scope))
    num_warp_values =
        clamp(num_values - (warp_index - 1) * THREADS_PER_WARP, 0, THREADS_PER_WARP)
    warp_scan = shuffle_scan(ThisWarp(), op, value, num_warp_values, flip)
    warp_total =
        shuffle_point_value(ThisWarp(), warp_scan, flip ? 1 : max(num_warp_values, 1))
    totals_scope = warp_totals_scope(scope)
    totals = DataLayouts.scoped_static_array(
        scope,
        typeof(warp_total),
        DataLayouts.static_num_threads(totals_scope),
    )
    if is_first_thread_in(ThisWarp())
        @inbounds totals[warp_index] = warp_total
    end
    DataLayouts.synchronize(scope)
    num_warps = max(cld(num_values, THREADS_PER_WARP), 1)
    lane_index = min(Int(DataLayouts.thread_rank(totals_scope)), num_warps)
    total = @inbounds totals[flip ? num_warps + 1 - lane_index : lane_index]
    DataLayouts.synchronize(scope)
    totals_scan = shuffle_scan(totals_scope, op, total, num_warps, false)
    warp_position = flip ? num_warps + 1 - warp_index : warp_index
    prefix = shuffle_point_value(totals_scope, totals_scan, max(warp_position - 1, 1))
    total = shuffle_point_value(totals_scope, totals_scan, num_warps)
    return (warp_position > 1 ? op(prefix, warp_scan) : warp_scan, total)
end
@inline warp_totals_scope(::ThisBlock) = ThisWarp()
@inline warp_totals_scope(::ThisSubBlock{N}) where {N} =
    ThisSubBlock{N ÷ THREADS_PER_WARP}()

# Kogge-Stone scan of a sub-warp, in which each thread combines its value with the one
# shuffled from the thread at every power-of-two offset below (above, when flip) its rank;
# a thread past num_values never sends a value to a thread at or before num_values.
@inline function shuffle_scan(scope, op::O, value, num_values, flip) where {O}
    rank = DataLayouts.thread_rank(scope)
    for offset in ntuple(n -> one(UInt32) << (n - 1), Val(num_reductions(scope)))
        shuffled_value = shuffle_shift(scope, value, offset, flip)
        if flip ? rank + offset <= num_values : rank > offset
            value = op(shuffled_value, value)
        end
    end
    return value
end
