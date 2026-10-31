has_inferred_slice_size(op::O, arg) where {O} =
    Val{true} == Utilities.return_type(Tuple{typeof(arg)}) do arg
        slice = op(arg, Tuple(first(DataLayouts.each_slice_index(op, arg)))...)
        Val(DataLayouts.has_inferred_size(slice))
    end
has_inferred_slice_size(op::O, arg::Union{Tuple, NamedTuple}) where {O} =
    unrolled_all(Base.Fix1(has_inferred_slice_size, op), arg)

DataLayouts.needs_loop_setup(::ThisHost) = true

# A fused loop's kernel needs inferrable slices, and its arguments must fit in the parameter
# memory that CUDA.jl allows: 4 KiB, or 32 KiB for PTX 8.1+ on Volta and newer devices.
# The size is inferred, so the device is only queried for arguments larger than 4 KiB. The
# arguments of this and the other functions of a loop that are not inlined are a Vararg of
# static length (see DataLayouts._foreach_slice).
function DataLayouts.fits_slice_loop(
    ::ThisHost,
    op::O,
    f::F,
    mask::M,
    args::Vararg{Any, N},
) where {O, F, M, N}
    unrolled_all(Base.Fix1(has_inferred_slice_size, op), args) || return false
    T = Core.Compiler.return_type(kernel_arguments, Tuple{F, M, typeof(args)})
    isconcretetype(T) || return false
    # The margin covers the KernelState argument and alignment padding.
    return sizeof(T) + 64 <= 4096 || sizeof(T) + 64 <= max_kernel_parameter_bytes()
end

# The arguments of a fused loop's kernel (see the launch in DataLayouts._foreach_slice below),
# including the mask that its function captures.
kernel_arguments(f, mask, args) =
    CUDA.cudaconvert((f, mask, Grids.toggle_compact_args(args...)))

function max_kernel_parameter_bytes()
    compiler = isdefined(CUDA, :compiler_config) ? CUDA : CUDA.CUDACore # moved in CUDA 6
    ptx = compiler.compiler_config(CUDA.device()).params.ptx
    return CUDA.capability(CUDA.device()) >= v"7.0" && ptx >= v"8.1" ? 32764 : 4096
end

# Only run a fused foreach_slice kernel when every slice has an inferrable size.
# Also, avoid launching fused loops on MPI ranks that do not own any elements.
@inline DataLayouts._foreach_slice(
    scope::ThisHost{L, R},
    op::O,
    f::F,
    mask,
    enumerate,
    serial,
    args...,
) where {L, R, O, F} =
    if !unrolled_all(Base.Fix1(has_inferred_slice_size, op), args)
        DataLayouts.unfused_slice_loop(scope, op, f, args...; mask, enumerate)
    elseif !isempty(DataLayouts.each_slice_index(op, DataLayouts.first_indexable(args)))
        check_device_assumptions()
        # partition(::ThisKernel) descends directly to sub-blocks, so
        # slice_subscope never returns ThisBlock itself.
        kernel_subscope =
            serial isa Val{true} ? DataLayouts.ThisThread() :
            DataLayouts.slice_subscope(ThisKernel(), op, args...)
        subscope = slice_subscope_at_level(kernel_subscope, Val(L), op, f, args...)
        max_slices =
            length(DataLayouts.each_slice_index(op, DataLayouts.first_indexable(args)))
        check_index_count(max_slices)
        compact_args = Grids.toggle_compact_args(args...)
        launch_slice_loop(
            subscope,
            can_exceed_level(kernel_subscope, Val(L), op, f, args...),
            Val(R),
            op,
            f,
            mask,
            enumerate,
            compact_args,
            max_slices,
        )
    end

# The kernel of a slice loop whose slices are assigned to subscope, a closure whose type names
# the subscope and the body (see launch_slice_loop and SHMEM_LEVEL_CACHE).
slice_loop_kernel(subscope, op::O, f::F, mask, enumerate) where {O, F} =
    function (compact_kernel_args...)
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
        return nothing
    end

# Whether the kernel of a slice loop at a level of shared memory compiles, by the types of
# the kernel (which names its subscope; see slice_loop_kernel) and of its arguments (see
# launch_slice_loop, which launches a loop whose kernel does not at the next level): CUDA.jl
# does not cache a kernel that fails to compile, which would otherwise be compiled again at
# every launch of the loop.
const SHMEM_LEVEL_CACHE = Dict{Tuple{DataType, DataType}, Bool}()
const SHMEM_LEVEL_CACHE_LOCK = ReentrantLock()

# Launch a slice loop with every slice assigned to subscope. A kernel that does not compile
# because a buffer exceeds the level (see check_shmem_level) is replaced by the kernel at the
# next level (see next_shmem_level), up to one thread per slice, which allocates no shared
# memory: a level is the most shared memory that a slice is given, not the least, and the
# next level is reported (see warn_shmem_level_exceeded) when the kernel is first compiled.
# A loop whose level holds its buffers (see can_exceed_level) is launched without the next
# levels, whose launches would otherwise be compiled on the host with every loop's launch.
function launch_slice_loop(
    subscope::ThisSubBlock,
    ::Val{true},
    R::Val,
    op::O,
    f::F,
    mask,
    enumerate,
    compact_args,
    max_slices,
) where {O, F}
    kernel_function = slice_loop_kernel(subscope, op, f, mask, enumerate)
    key = (typeof(kernel_function), typeof(compact_args))
    compiles = @lock SHMEM_LEVEL_CACHE_LOCK get(SHMEM_LEVEL_CACHE, key, nothing)
    next = next_shmem_level(subscope)
    compiles === false && return launch_slice_loop(
        next, Val(true), R, op, f, mask, enumerate, compact_args, max_slices,
    )
    exceeded = launch_subblock_loop(kernel_function, R, compact_args, max_slices)
    isnothing(compiles) &&
        @lock SHMEM_LEVEL_CACHE_LOCK SHMEM_LEVEL_CACHE[key] = isnothing(exceeded)
    isnothing(exceeded) && return nothing
    warn_shmem_level_exceeded(f, subscope, next, exceeded)
    return launch_slice_loop(
        next, Val(true), R, op, f, mask, enumerate, compact_args, max_slices,
    )
end
function launch_slice_loop(
    subscope::ThisSubBlock,
    ::Val{false},
    R::Val,
    op::O,
    f::F,
    mask,
    enumerate,
    compact_args,
    max_slices,
) where {O, F}
    kernel_function = slice_loop_kernel(subscope, op, f, mask, enumerate)
    exceeded = launch_subblock_loop(kernel_function, R, compact_args, max_slices)
    isnothing(exceeded) || throw_shmem_level_exceeded(f, subscope, exceeded)
    return nothing
end

# One thread per slice; extra threads run empty loops, so the count of slices isn't a strict
# limit. The blocks of a masked loop are whole warps, which take slices in turn (see
# subscope_slice_indices). The kernel is compiled with the loop's register cap (see
# register_cap).
function launch_slice_loop(
    subscope::DataLayouts.ThisThread,
    _,
    ::Val{R},
    op::O,
    f::F,
    mask,
    enumerate,
    compact_args,
    max_slices,
) where {R, O, F}
    kernel_function = slice_loop_kernel(subscope, op, f, mask, enumerate)
    granularity = mask isa DataLayouts.NoMask ? nothing : THREADS_PER_WARP
    config = kernel -> launch_configuration(kernel, max_slices; strict = false, granularity)
    auto_launch!(
        kernel_function,
        compact_args;
        config,
        maxregs = register_cap(Val(R), subscope),
    )
    return nothing
end

# The slices of a masked loop with one thread per slice are dealt out to the warps of a
# kernel's blocks in turn, each warp taking consecutive slices, so that its active slices
# (which its launch does not count) run on every multiprocessor, instead of on those of its
# first blocks. The threads of a kernel whose blocks are not whole warps (which a slice loop
# never launches) take slices in the order of their ranks.
@inline function DataLayouts.subscope_slice_indices(
    ::DataLayouts.ThisThread,
    ::ThisKernel,
    mask::DataLayouts.IJHMask,
    op::O,
    args...,
) where {O}
    indices = DataLayouts.maskable_slice_indices(mask, op, args...)
    lanes = UInt32(THREADS_PER_WARP)
    thread = DataLayouts.thread_rank(ThisBlock()) - one(UInt32)
    block = DataLayouts.subscope_rank(ThisBlock(), ThisKernel()) - one(UInt32)
    num_blocks = DataLayouts.num_subscopes(ThisBlock(), ThisKernel())
    rank =
        iszero(DataLayouts.num_threads(ThisBlock()) % lanes) ?
        (thread ÷ lanes * num_blocks + block) * lanes + thread % lanes + one(UInt32) :
        DataLayouts.thread_rank(ThisKernel())
    n = DataLayouts.num_threads(ThisKernel())
    view_range = DataLayouts.strided_range(rank, n, length(indices))
    return @inbounds DataLayouts.subscope_index_view(ThisKernel(), indices, view_range)
end

# Launch a slice loop whose kernel assigns a slice to each sub-block, and return nothing, or
# the bytes of a buffer beyond the level (see auto_launch!) if the kernel was not launched. A
# block holds whole sub-blocks (the granularity; see the whole subscopes invariant in
# DataLayouts.subscope_indices): one wide sub-block (see max_subblock_launch_threads), or as
# many narrow ones as launch_configuration fits with their shared memory, the surplus
# sub-blocks of the last block running empty loops. The kernel is compiled with the loop's
# register cap (see register_cap).
function launch_subblock_loop(
    kernel_function::K,
    ::Val{R},
    compact_args,
    max_slices,
) where {K, R}
    subscope = kernel_function.subscope
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
    return auto_launch!(
        kernel_function,
        compact_args;
        config,
        shmem,
        maxregs = register_cap(Val(R), subscope),
    )
end

# Warn that a loop is launched at the next level of shared memory (see launch_slice_loop),
# once per kernel (at the launch whose compilation reports the bytes). A loop given no shared
# memory without its caller choosing so (its known buffers need none, or its slices are
# narrow sub-blocks whose buffers are not known; see slice_subscope_at_level) that allocates
# some is a debug message instead.
function warn_shmem_level_exceeded(f::F, ::ThisSubBlock{N, B}, next, bytes) where {F, N, B}
    needs = "foreach_slice body $(nameof(F)) needs $bytes bytes of shared memory for slices \
             of $N threads"
    at_next = "it is launched at the $(shmem_level_name(next)) level instead"
    if iszero(B)
        @debug "$needs, although it was given none; $at_next"
    else
        @warn "$needs, more than the $(region_bytes(N, B)) bytes of its shmem level \
               ($B bytes per thread); $at_next. Pass shmem = Val($(shmem_level_name(next))) \
               to choose the level."
    end
    return nothing
end
# A loop whose level was chosen to hold its known buffers (see can_exceed_level) that
# exceeds it has buffers that DataLayouts.required_shared_memory does not account for.
@noinline throw_shmem_level_exceeded(f::F, ::ThisSubBlock{N, B}, bytes) where {F, N, B} =
    error("foreach_slice body $(nameof(F)) needs $bytes bytes of shared memory for slices \
           of $N threads, more than the $(region_bytes(N, B)) bytes of the level chosen \
           for the buffers that DataLayouts.required_shared_memory reports")
shmem_level_name(::ThisSubBlock{N, B}) where {N, B} =
    B == shmem_level_bytes(Val(:medium), Val(N)) ? ":medium" :
    B == shmem_level_bytes(Val(:large), Val(N)) ? ":large" : ":max"
shmem_level_name(::DataLayouts.ThisThread) = ":none"

# The subscope of every slice of a loop at a level of shared memory (see ThisHost): one
# thread at :none, a sub-block whose threads have the level's bytes at a fixed level, and at
# :auto a sub-block at the first level that holds the buffers of the function's operators
# when they are known from the slices (see DataLayouts.required_shared_memory): no shared
# memory when the function allocates none, which leaves the memory of a multiprocessor to
# its L1 cache, then :medium (which never limits occupancy) or else exactly their bytes (the
# rest goes to more blocks and the L1 cache), and one thread per slice beyond :max. A loop
# whose buffers are not known is launched at :medium when its slices are wider than a warp,
# and without shared memory when they are not: sub-blocks of at most a warp cache the values
# they read without branching in registers (see Operators.caches_in_registers) and rarely
# allocate any buffer, while the reservation of the :medium level shrinks the L1 cache of
# the kernel (2-4% slower fused column bodies of 32 faces on an A100). A buffer that exceeds
# the level (a fixed level, or at :auto that of a loop whose buffers are not known, including
# one given none that allocates some after all) moves the loop to the next level (see
# launch_slice_loop and can_exceed_level).
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
    args::Vararg{Any, A},
) where {N, O, F, A}
    medium = ThisSubBlock{N, shmem_level_bytes(Val(:medium), Val(N))}()
    requirement = slices_required_shared_memory(medium, op, f, args...)
    isnothing(requirement) && return N <= THREADS_PER_WARP ? ThisSubBlock{N, 0}() : medium
    (bytes, uses) = requirement
    iszero(bytes) && !uses && return ThisSubBlock{N, 0}()
    bytes <= DataLayouts.total_buffer_limit(medium) && return medium
    max = ThisSubBlock{N, shmem_level_bytes(Val(:max), Val(N))}()
    bytes <= DataLayouts.total_buffer_limit(max) || return DataLayouts.ThisThread()
    return ThisSubBlock{N, cld(aligned_buffer_bytes(bytes), N)}()
end

# Whether a loop can exceed the level of shared memory of its slices (see
# slice_subscope_at_level), so that it is launched at the next levels when it does (see
# launch_slice_loop): every loop, except at the :auto level a loop whose buffers are known,
# since its level holds them.
can_exceed_level(_, _, op, f, args...) = Val(true)
function can_exceed_level(
    ::ThisSubBlock{N},
    ::Val{:auto},
    op::O,
    f::F,
    args::Vararg{Any, A},
) where {N, O, F, A}
    medium = ThisSubBlock{N, shmem_level_bytes(Val(:medium), Val(N))}()
    return Val(isnothing(slices_required_shared_memory(medium, op, f, args...)))
end

# The level of shared memory after that of a sub-block (see slice_subscope_at_level):
# :medium after none, :large after :medium, :max after :large, and one thread per slice
# after :max.
next_shmem_level(::ThisSubBlock{N, B}) where {N, B} =
    B < shmem_level_bytes(Val(:medium), Val(N)) ?
    ThisSubBlock{N, shmem_level_bytes(Val(:medium), Val(N))}() :
    B < shmem_level_bytes(Val(:large), Val(N)) ?
    ThisSubBlock{N, shmem_level_bytes(Val(:large), Val(N))}() :
    B < shmem_level_bytes(Val(:max), Val(N)) ?
    ThisSubBlock{N, shmem_level_bytes(Val(:max), Val(N))}() : DataLayouts.ThisThread()

# Bytes of shared memory that f needs on every slice and whether it allocates any (or nothing
# if they are unknown), found by inference alone (like has_inferred_slice_size), since they
# only depend on the types of the slices; a result that inference cannot fold is an error, not
# a fallback. The body f is an argument of the inferred function rather than captured by it,
# so that a body that captures values keeps the call foldable. The slices are reassigned after
# they are taken: slicing a host array rebuilds its layout with the host's scope (see
# DataLayouts.rebuild), which would undo the reassignment.
slices_required_shared_memory(subscope, op::O, f::F, args::Vararg{Any, N}) where {O, F, N} =
    val_value(
        Utilities.return_type(Tuple{F, typeof(args)}) do f, args
            index =
                first(DataLayouts.each_slice_index(op, DataLayouts.first_indexable(args)))
            slices = DataLayouts.slice_every_arg(op, index, args...)
            scoped_slices = unrolled_map(Base.Fix2(DataLayouts.reassign, subscope), slices)
            bytes = DataLayouts.required_shared_memory(f, scoped_slices...)
            Val(
                isnothing(bytes) ? nothing :
                (bytes, DataLayouts.uses_shared_memory(f, scoped_slices...)),
            )
        end,
    )
val_value(::Type{Val{x}}) where {x} = x

# Every region of a sub-block slice loop is given the bytes of its level (none at the level
# without shared memory; see slice_subscope_at_level).
subblock_shmem(::ThisSubBlock{N, B}) where {N, B} = DynamicShmem(N * B, N)

# Dynamic shared memory of a reduction kernel: the warp results of a block reduction,
# which are retired before the block allocates its flag in the same memory.
block_reduction_shmem(::Type{T}) where {T} =
    DynamicShmem(
        aligned_buffer_bytes(max(MAX_WARPS_PER_BLOCK * sizeof(T), 1)),
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
                final_val = parallel_reduce(ThisWarp(), op, local_val, num_active)
                if is_first_thread_in(ThisWarp())
                    @inbounds results[1] = final_val
                end
            end
        end
        return nothing
    end

    # The warp results of a block are stored with the type of the fold on each thread, which
    # a wider init widens (see reduce_points(::ThisCooperativeGroup)), so the type sizes the
    # launch's shared memory. The fold is inferred on a thread of the host, whose indices
    # the host cannot compute.
    T = DataLayouts.thread_fold_type(
        op,
        DataLayouts.reassign(arg, DataLayouts.ThisThread());
        kernel_kwargs...,
    )
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

# Combination of the first num_values threads' values, returned to every thread (or only to
# the first thread when warps are partially active; see shuffle_reduce). Multi-warp scopes
# reduce each warp, and then the warp results, which are read from shared memory into the
# threads of a subscope that has one thread per warp in the full scope.
@inline parallel_reduce(scope::ThisCooperativeGroup, op::O, value, num_values) where {O} =
    if scope != ThisBlock() && shuffles_in_warp(scope)
        shuffle_reduce(scope, op, value, num_values)
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
        shuffle_reduce(subscope, op, lane_value, num_warps)
    end

# The warp results that parallel_reduce and parallel_scan exchange in shared memory.
DataLayouts.combined_values_bytes(scope::ThisCooperativeGroup, ::Type{T}) where {T} =
    scope != ThisBlock() && shuffles_in_warp(scope) ? 0 :
    DataLayouts.scoped_buffer_bytes(
        scope,
        DataLayouts.static_num_threads(warps_to_threads(scope)) * sizeof(T),
    )

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
        prefix = shuffle_scan(subscope, op, lane_value, num_warps, false)

        partial_result_prefix_index = reverse ? num_warps - warp_index : warp_index - 1
        partial_result_prefix =
            shuffle_point_value(subscope, prefix, max(partial_result_prefix_index, 1))
        partial_result =
            partial_result_prefix_index <= 0 ? partial_warp_result :
            op(partial_result_prefix, partial_warp_result)
        final_result = shuffle_point_value(subscope, prefix, num_warps)
        (partial_result, final_result)
    end

# A slice loop launched at a level that its kernel exceeds recurses through the levels after
# it (see launch_slice_loop), each a sub-block of another type, which the inference recursion
# limit would widen to an abstract sub-block, dispatching the launches at run time.
Utilities.@drop_recursion_limits launch_slice_loop
