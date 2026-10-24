# Only run a fused foreach_slice kernel when every slice has an inferrable size.
has_inferred_slice_size(op::O, arg) where {O} =
    Val{true} == Utilities.return_type(Tuple{typeof(arg)}) do arg
        slice = op(arg, Tuple(first(DataLayouts.each_slice_index(op, arg)))...)
        Val(DataLayouts.has_inferred_size(slice))
    end

DataLayouts.needs_loop_setup(::ThisHost) = true

@inline DataLayouts._foreach_slice(
    scope::ThisHost{L},
    op::O,
    f::F,
    mask,
    enumerate,
    args...,
) where {L, O, F} =
    if !unrolled_all(Base.Fix1(has_inferred_slice_size, op), args)
        DataLayouts.unfused_slice_loop(scope, op, f, args...; mask, enumerate)
    else
        check_device_assumptions()

        # A rank can own no elements; there are then no slices to launch
        # over, and num_slice_points cannot slice an empty layout.
        isempty(DataLayouts.each_slice_index(op, first(args))) && return

        # partition(::ThisKernel) descends directly to sub-blocks, so
        # slice_subscope never returns ThisBlock itself.
        subscope = slice_subscope_at_level(
            DataLayouts.slice_subscope(ThisKernel(), op, args...),
            Val(L),
            op,
            f,
            args...,
        )
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
        compact_args = Grids.toggle_compact_args(args...)
        max_slices = length(DataLayouts.each_slice_index(op, first(args)))
        check_slice_count(max_slices)
        if subscope isa ThisSubBlock
            # Each sub-block gets one slice, and a block holds whole sub-blocks (the
            # granularity; see the whole subscopes invariant in DataLayouts.subscope_indices):
            # one wide sub-block (see max_subblock_launch_threads), or as many narrow ones
            # as launch_configuration fits with their shared memory, the surplus sub-blocks
            # of the last block running empty loops.
            subblock_threads = Int(DataLayouts.num_threads(subscope))
            max_block_threads = max_subblock_launch_threads(subscope)
            shmem = subblock_shmem(subscope)
            config =
                kernel ->
                    max_block_threads == subblock_threads ?
                    launch_configuration(
                        kernel, subblock_threads, max_slices;
                        granularity = subblock_threads,
                        shmem,
                    ) :
                    launch_configuration(
                        kernel, max_slices * subblock_threads;
                        strict = false,
                        granularity = subblock_threads,
                        shmem,
                    )
            launch_at_level(kernel_function, compact_args, config, shmem)
        else
            # One thread per slice; extra threads run empty loops, so the count of slices
            # isn't a strict limit.
            config = kernel -> launch_configuration(kernel, max_slices; strict = false)
            auto_launch!(kernel_function, compact_args; config)
        end
    end

# The subscope of every slice of a loop at a level of shared memory (see ThisHost): one
# thread at :none, a sub-block whose threads have the level's bytes at a fixed level, and
# at :auto a sub-block at :medium, unless the buffers of the function's operators are known
# from the slices (see DataLayouts.required_shared_memory) and need more, in which case :max
# if its bytes hold them, or one thread per slice beyond :max. So a broadcast never fails to
# compile for lack of a level.
slice_subscope_at_level(subscope, _, op, f, args...) = subscope
slice_subscope_at_level(::ThisSubBlock, ::Val{:none}, op, f, args...) =
    DataLayouts.ThisThread()
slice_subscope_at_level(::ThisSubBlock{N}, level::Val, op, f, args...) where {N} =
    ThisSubBlock{N, shmem_level_bytes(level, Val(N))}()
function slice_subscope_at_level(
    ::ThisSubBlock{N},
    ::Val{:auto},
    op::O,
    f::F,
    args...,
) where {N, O, F}
    medium = ThisSubBlock{N, shmem_level_bytes(Val(:medium), Val(N))}()
    bytes = slices_required_shared_memory(medium, op, f, args...)
    (isnothing(bytes) || bytes <= DataLayouts.total_buffer_limit(medium)) && return medium
    max = ThisSubBlock{N, shmem_level_bytes(Val(:max), Val(N))}()
    bytes <= DataLayouts.total_buffer_limit(max) && return max
    return DataLayouts.ThisThread()
end

# Bytes of shared memory that f needs on every slice (or nothing if they are unknown), found
# by inference alone (like has_inferred_slice_size), since they only depend on the types of
# the slices; a result that inference cannot fold is an error, not a fallback. The slices
# are reassigned after they are taken: slicing a host array rebuilds its layout with the
# host's scope (see DataLayouts.rebuild), which would undo the reassignment.
slices_required_shared_memory(subscope, op::O, f::F, args...) where {O, F} =
    val_value(
        Utilities.return_type(Tuple{typeof(args)}) do args
            index = first(DataLayouts.each_slice_index(op, first(args)))
            slices = DataLayouts.slice_every_arg(op, index, args...)
            scoped_slices = unrolled_map(Base.Fix2(DataLayouts.reassign, subscope), slices)
            Val(DataLayouts.required_shared_memory(f, scoped_slices...))
        end,
    )
val_value(::Type{Val{x}}) where {x} = x

# Every region of a sub-block slice loop is given the bytes of its level.
subblock_shmem(::ThisSubBlock{N, B}) where {N, B} = DynamicShmem(N * B, N)

# A buffer that exceeds the level fails to compile (see check_shmem_level); the marker is the
# name of the undefined function that the check calls, read from the invalid IR error's
# metadata, and the largest one is reported.
function launch_at_level(kernel_function, compact_args, config, shmem)
    try
        auto_launch!(kernel_function, compact_args; config, shmem)
    catch err
        err isa CUDA_COMPILER.GPUCompiler.InvalidIRError || rethrow()
        markers = [
            meta for (_, _, meta) in err.errors if
            meta isa String && startswith(meta, SHMEM_LEVEL_ERROR)
        ]
        isempty(markers) && rethrow()
        (bytes, threads, per_thread) = maximum(shmem_level_error_values, markers)
        throw(
            ArgumentError(
                "foreach_slice needs $bytes bytes of shared memory for slices of $threads \
                 threads ($(cld(bytes, threads)) bytes per thread), more than the \
                 $per_thread bytes per thread of its shmem level; pass a larger shmem (see \
                 foreach_slice), or shmem = Val(:none) to evaluate each slice on one thread",
            ),
        )
    end
end
function shmem_level_error_values(name::String)
    m = match(r"_(\d+)_bytes_for_(\d+)_threads_at_(\d+)_", name)::RegexMatch
    return ntuple(n -> parse(Int, m.captures[n]::AbstractString), Val(3))
end

# Each thread steps through its indices in UInt32 by adding the thread count, which can be
# as large as the index count, so half of the UInt32 range is left for the step.
check_slice_count(n) =
    2 * n <= typemax(UInt32) ||
    throw(ArgumentError("GPU loops support at most 2^31 - 1 slice views"))

# Dynamic shared memory of a reduction kernel: the warp results of a block reduction,
# which are retired before the block allocates its flag in the same memory.
block_reduction_shmem(::Type{T}) where {T} =
    DynamicShmem(
        DataLayouts.aligned_buffer_bytes(max(MAX_WARPS_PER_BLOCK * sizeof(T), 1)),
        0,
    )

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
    # The launch needs results sized by the block count, so the kernel is compiled before it.
    kernel = CUDA.@cuda always_inline = true launch = false kernel_function(
        results, finished_blocks, arg, UInt32(1),
    )
    shmem = block_reduction_shmem(T)
    (; threads, blocks) = launch_configuration(kernel, length(arg); shmem)
    results = DataLayouts.scoped_array(scope, T, blocks; buffer = true)

    fill!(finished_blocks, UInt32(0))
    auto_launch!(
        kernel_function,
        (results, finished_blocks, arg, UInt32(blocks));
        threads_s = threads,
        blocks_s = blocks,
        shmem,
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
        max_results = scope == ThisBlock() ? MAX_WARPS_PER_BLOCK : Int(num_warps)
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
        ntuple(Base.Fix1(>>, DataLayouts.num_threads(scope)), Val(num_offsets))
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
