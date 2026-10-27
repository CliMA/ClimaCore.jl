import CUDA
import ClimaCore.Fields
import ClimaCore.DataLayouts
import ClimaCore.Utilities

function uncached_device_attributes()
    device = CUDA.device()
    get_attr(code) = Int(CUDA.attribute(device, code))
    sm_version = (
        get_attr(CUDA.DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
        get_attr(CUDA.DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
    )
    sm_version[1] > 12 && throw(ArgumentError("Missing device attributes for \
                                               Compute Capability $(sm_version[1])"))
    return (;
        threads_per_warp = get_attr(CUDA.DEVICE_ATTRIBUTE_WARP_SIZE),

        # kernel launch configuration limits
        max_block_dim_x = get_attr(CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X),
        max_block_dim_y = get_attr(CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y),
        max_block_dim_z = get_attr(CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z),
        max_grid_dim_x = get_attr(CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X),
        max_grid_dim_y = get_attr(CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y),
        max_grid_dim_z = get_attr(CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z),

        # occupancy limits exposed by the CUDA API
        sm_count = get_attr(CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        max_threads_per_block = get_attr(CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
        max_threads_per_sm = get_attr(CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR),
        max_regs_per_block = get_attr(CUDA.DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK),
        max_regs_per_sm = get_attr(CUDA.DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR),
        max_shmem_per_block = get_attr(CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK),
        max_shmem_per_sm =
        get_attr(CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR),
        reserved_shmem_per_block =
        get_attr(CUDA.DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK),
        # a kernel can use more than max_shmem_per_block after opting in (see
        # launch_configuration)
        max_shmem_per_block_optin =
        get_attr(CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN),

        # more hardware properties from version 12.9 of the CUDA Runtime Headers
        # https://gitlab.com/nvidia/headers/cuda-individual/cudart/-/blob/main/cuda_occupancy.h#L606-779
        regs_per_allocation = 256,
        shmem_per_allocation = sm_version[1] < 8 ? 256 : 128,
        max_regs_per_thread = sm_version[1] < 7 ? 255 : 256,
        schedulers_per_sm = sm_version == (6, 0) ? 2 : 4,
        max_blocks_per_sm =
        sm_version[1] < 5 || sm_version in ((7, 5), (8, 6), (8, 7)) ? 16 :
        sm_version in ((8, 9), (10, 1)) || sm_version[1] == 12 ? 24 : 32,

        # custom property to represent how the 2 schedulers/SM of CC 6.0 are
        # bumped to 4 schedulers/SM for compatibility with other CC 6.x versions
        # https://gitlab.com/nvidia/headers/cuda-individual/cudart/-/blob/main/cuda_occupancy.h#L1669-1692
        max_schedulers_per_sm = 4,
    )
end

const DEVICE_ATTRIBUTE_CACHE =
    Ref{Utilities.return_type(uncached_device_attributes, Tuple{})}()
const DEVICE_ATTRIBUTE_CACHE_INITIALIZED = Threads.Atomic{Bool}(false)

"""
    device_attributes()

Return the collection of hardware properties from the CUDA API and CUDA Runtime Headers,
obtained through calls to `CUDA.attribute`.

Attributes are obtained by querying the C driver, which adds measurable latency
if done more than once, so all attributes are cached after the first call. An
`Atomic` initialization flag is used to ensure that every thread either sees a
fully written cache or sets the cache itself. Any thread can set the cache, as
ClimaComms only targets one device from threads launched by the same process.
"""
function device_attributes()
    DEVICE_ATTRIBUTE_CACHE_INITIALIZED[] && return DEVICE_ATTRIBUTE_CACHE[]
    DEVICE_ATTRIBUTE_CACHE[] = uncached_device_attributes()
    DEVICE_ATTRIBUTE_CACHE_INITIALIZED[] = true
    return DEVICE_ATTRIBUTE_CACHE[]
end

# cudaOccMaxActiveBlocksPerMultiprocessor times the number of multiprocessors,
# and its limiting factor (num_blocks/num_threads/registers/shared_memory); the
# block-barrier limit is intentionally ignored. CUDA's occupancy calculator
# counts the named-barrier slots used by PTX `bar.sync N` (N > 0), `mbarrier`,
# and similar instructions. ClimaCore uses only `sync_threads()` (PTX
# `bar.sync 0`, the implicit CTA barrier, which does not consume a named-barrier
# slot), `sync_warp()` (a warp convergence instruction), and `threadfence()` (a
# memory fence, not a barrier). None of these affects the named-barrier count,
# so the limit is irrelevant here.
# https://gitlab.com/nvidia/headers/cuda-individual/cudart/-/blob/main/cuda_occupancy.h#L1282-1777
function max_active_blocks(threads_per_block, regs_per_thread, shmem_per_block)
    attrs = device_attributes()
    round_up(n, divisor) = cld(n, divisor) * divisor # round n up to next multiple of divisor

    (max_blocks_per_sm, limit) = (attrs.max_blocks_per_sm, :num_blocks)

    if !iszero(threads_per_block)
        threads_per_block > attrs.max_threads_per_block && return (0, :num_threads)
        warps_per_block = cld(threads_per_block, attrs.threads_per_warp)
        (max_warps_per_sm, warp_limit) =
            (fld(attrs.max_threads_per_sm, attrs.threads_per_warp), :num_threads)
        if !iszero(regs_per_thread)
            regs_per_thread > attrs.max_regs_per_thread && return (0, :registers)
            min_regs_per_warp = regs_per_thread * attrs.threads_per_warp
            allocated_warps_per_block =
                round_up(warps_per_block, attrs.max_schedulers_per_sm)
            allocated_regs_per_warp = round_up(min_regs_per_warp, attrs.regs_per_allocation)
            allocated_regs_per_block = allocated_warps_per_block * allocated_regs_per_warp
            allocated_regs_per_block > attrs.max_regs_per_block && return (0, :registers)
            max_regs_per_scheduler = fld(attrs.max_regs_per_sm, attrs.schedulers_per_sm)
            max_warps_per_scheduler = fld(max_regs_per_scheduler, allocated_regs_per_warp)
            max_warps_per_sm_by_regs = max_warps_per_scheduler * attrs.schedulers_per_sm
            if max_warps_per_sm_by_regs < max_warps_per_sm
                (max_warps_per_sm, warp_limit) = (max_warps_per_sm_by_regs, :registers)
            end
        end
        max_blocks_per_sm_by_warps = fld(max_warps_per_sm, warps_per_block)
        if max_blocks_per_sm_by_warps < max_blocks_per_sm
            (max_blocks_per_sm, limit) = (max_blocks_per_sm_by_warps, warp_limit)
        end
    end

    if !iszero(shmem_per_block)
        min_shmem_per_block = shmem_per_block + attrs.reserved_shmem_per_block
        max_shmem_per_block =
            attrs.max_shmem_per_block_optin + attrs.reserved_shmem_per_block
        allocated_shmem_per_block =
            round_up(min_shmem_per_block, attrs.shmem_per_allocation)
        allocated_shmem_per_block > max_shmem_per_block && return (0, :shared_memory)
        max_blocks_per_sm_by_shmem = fld(attrs.max_shmem_per_sm, allocated_shmem_per_block)
        if max_blocks_per_sm_by_shmem < max_blocks_per_sm
            (max_blocks_per_sm, limit) = (max_blocks_per_sm_by_shmem, :shared_memory)
        end
    end

    return (max_blocks_per_sm * attrs.sm_count, limit)
end

"""
    DynamicShmem(region_bytes, region_threads)

Dynamic shared memory of a kernel launch, as a function of its number of threads per block:
`region_bytes` for every `region_threads` threads, or for the whole block when
`region_threads` is 0 (see `scoped_static_array`). A `Function`, so that
`CUDA.launch_configuration` accepts it as its `shmem`.
"""
struct DynamicShmem <: Function
    region_bytes::Int
    region_threads::Int
end
(shmem::DynamicShmem)(threads_per_block) =
    iszero(shmem.region_threads) ? shmem.region_bytes :
    shmem.region_bytes * fld(threads_per_block, shmem.region_threads)
const NO_DYNAMIC_SHMEM = DynamicShmem(0, 0)

# cudaOccMaxPotentialOccupancyBlockSize times the maximum number of waves,
# optimized for single-stream execution of both small and large workloads
# https://gitlab.com/nvidia/headers/cuda-individual/cudart/-/blob/main/cuda_occupancy.h#L1865-1965
function uncached_launch_configuration(
    cu_func,
    strict,
    max_waves,
    default_granularity,
    dynamic_shmem,
    config_args...,
)
    attrs = device_attributes()
    cu_func_attrs = CUDA.attributes(cu_func)
    regs_per_thread = Int(cu_func_attrs[CUDA.FUNC_ATTRIBUTE_NUM_REGS])
    shmem_per_block = Int(cu_func_attrs[CUDA.FUNC_ATTRIBUTE_SHARED_SIZE_BYTES])
    default_max_threads_per_block = min(
        Int(cu_func_attrs[CUDA.FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK]),
        attrs.max_threads_per_block,
    )

    get_user_max_threads_per_block() = attrs.max_block_dim_x
    get_user_max_threads_per_block(user_max_threads) = user_max_threads
    get_user_max_threads_per_block(user_max_threads_per_block, _) =
        user_max_threads_per_block

    get_user_max_blocks(_) = attrs.max_grid_dim_x
    get_user_max_blocks(threads_per_block, user_max_threads) =
        (strict ? fld : cld)(user_max_threads, threads_per_block) # round down when strict
    get_user_max_blocks(_, _, user_max_blocks) = user_max_blocks

    user_max_threads_per_block = get_user_max_threads_per_block(config_args...)
    user_constraints_ignorable = user_max_threads_per_block >= default_max_threads_per_block
    max_threads_per_block = min(user_max_threads_per_block, default_max_threads_per_block)

    # Block sizes are searched in whole multiples of this unit, one warp unless
    # a caller needs a coarser one: a partially populated sub-block skips the
    # points of the threads it is missing (see the whole subscopes invariant in
    # DataLayouts.subscope_indices). The unit is clamped to the largest launchable
    # block, so the search always has a candidate.
    granularity = max(something(default_granularity, 0), attrs.threads_per_warp)
    unit = min(granularity, max_threads_per_block)

    # A warp-unit caller also gets the largest launchable block even when it is
    # not a whole number of warps (matching CUDA's own search); only a caller
    # that asked for a coarser unit needs the clamped candidate dropped, since
    # a non-whole block is not merely suboptimal for it but incorrect.
    block_sizes =
        isnothing(default_granularity) ?
        (
            min(units * unit, max_threads_per_block) for
            units in 1:cld(max_threads_per_block, unit)
        ) : (units * unit for units in 1:fld(max_threads_per_block, unit))
    # Iterating from small to large block sizes, prefer configurations with more
    # total threads; on ties, prefer larger blocks (fewer blocks means less
    # block-scheduling latency), but never accept a new configuration with fewer
    # blocks unless it still has at least one block per multiprocessor. This
    # spreads small workloads across as many multiprocessors as possible, while
    # also confining large workloads to as few blocks as possible.
    (best_threads_per_block, best_num_blocks, best_limit) = (0, 0, :user)
    for threads_per_block in block_sizes
        (max_blocks_per_wave, active_blocks_limit) = max_active_blocks(
            threads_per_block,
            regs_per_thread,
            shmem_per_block + dynamic_shmem(threads_per_block),
        )
        user_max_blocks = get_user_max_blocks(threads_per_block, config_args...)
        user_constraints_ignorable &= user_max_blocks >= max_blocks_per_wave * max_waves
        (num_blocks, limit) =
            user_max_blocks <= max_blocks_per_wave * max_waves ? (user_max_blocks, :user) :
            (max_blocks_per_wave * max_waves, active_blocks_limit)
        if threads_per_block * num_blocks >= best_threads_per_block * best_num_blocks &&
           (num_blocks >= attrs.sm_count || num_blocks >= best_num_blocks)
            (best_threads_per_block, best_num_blocks, best_limit) =
                (threads_per_block, num_blocks, limit)
        end
    end

    # Every candidate was rejected (see max_active_blocks), or the user asked for no
    # threads; the shared memory of a launch is the usual cause, since a kernel's static
    # resources are checked against the device's largest block when it is compiled, and its
    # dynamic shared memory counts against the same limit per block.
    iszero(best_num_blocks) && throw(
        ArgumentError(
            "No block of a whole number of $unit threads can be launched: a block of $unit \
             threads needs $(dynamic_shmem(unit)) bytes of dynamic shared memory and \
             $shmem_per_block bytes of static shared memory, more than the \
             $(attrs.max_shmem_per_block_optin) bytes of shared memory available per block \
             (the kernel can be launched with at most $default_max_threads_per_block \
             threads per block)",
        ),
    )

    # Compare searches unaffected by config_args against CUDA's implementation,
    # which only ever considers whole warps.
    if DataLayouts.VALIDATE_LAUNCH_CONFIGURATIONS[] &&
       user_constraints_ignorable &&
       isnothing(default_granularity)
        min_blocks_per_wave = cld(best_num_blocks, max_waves)
        cuda_config = (; blocks = min_blocks_per_wave, threads = best_threads_per_block)
        # CUDA's shmem is the dynamic shared memory alone (it adds the static memory itself).
        @assert CUDA.launch_configuration(cu_func; shmem = dynamic_shmem) == cuda_config
    end

    # A block above the function's limit on dynamic shared memory (48 KB less its static
    # shared memory, until it is raised) needs the limit raised first: once, since the
    # configuration is cached, and never lowered, since the launches of the function's other
    # configurations may need the earlier limit.
    dynamic_bytes = dynamic_shmem(best_threads_per_block)
    attribute = CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    cu_func_attrs[attribute] < dynamic_bytes && (cu_func_attrs[attribute] = dynamic_bytes)

    return (;
        threads = best_threads_per_block,
        blocks = best_num_blocks,
        limit = best_limit,
    )
end

# Keyed by function handles, so that lookups allocate nothing (the type of the handle is
# CUDA.CUfunction before CUDA.jl v6 and CUDA.CUDACore.CUfunction after).
const CUfunction = fieldtype(CUDA.CuFunction, :handle)
# The module that compiles kernels (CUDA before CUDA.jl v6 and CUDACore after), through
# which GPUCompiler is reached.
const CUDA_COMPILER = parentmodule(CUDA.cufunction)

# The maxregs keyword of a kernel compilation (the register cap of a slice loop; see
# register_cap), or none when the count is left to ptxas.
maxregs_kwarg(::Nothing) = (;)
maxregs_kwarg(maxregs::Int) = (; maxregs)

# A buffer that exceeds the shared memory level of its slice loop fails to compile (see
# check_shmem_level); the marker is the name of the undefined function that the check calls,
# read from the invalid IR error's metadata, and the largest one gives the bytes the loop
# needs (see launch_slice_loop, which launches the loop at the level that holds them), or 0
# when the error has no marker. The metadata is untyped, so the markers are read in a loop
# that narrows it, rather than by a closure that would be dispatched on it.
function shmem_level_error_bytes(err::CUDA_COMPILER.GPUCompiler.InvalidIRError)
    bytes = 0
    for (_, _, meta) in err.errors
        meta isa String || continue
        startswith(meta, SHMEM_LEVEL_ERROR) || continue
        digits = chop(
            meta;
            head = ncodeunits(SHMEM_LEVEL_ERROR) + 1,
            tail = ncodeunits("_bytes"),
        )
        bytes = max(bytes, something(tryparse(Int, digits), 0))
    end
    return bytes
end
const LAUNCH_CONFIGURATION_CACHE = Dict{
    Tuple{CUfunction, Bool, Int, Int, DynamicShmem, Int, Int},
    @NamedTuple{threads::Int, blocks::Int, limit::Symbol},
}()
const LAUNCH_CONFIGURATION_CACHE_LOCK = ReentrantLock()

"""
    launch_configuration(kernel; [max_waves], [granularity], [shmem])
    launch_configuration(kernel, max_threads; strict = true, [max_waves], [granularity], [shmem])
    launch_configuration(kernel, max_threads_per_block, max_blocks; [max_waves], [granularity], [shmem])

Analogue of `CUDA.launch_configuration` optimized for single-stream execution,
which maximizes occupancy across the entire device instead of just one
multiprocessor, and which spreads small workloads across as many multiprocessors
as possible. While `CUDA.launch_configuration` and its underlying C function
`cudaOccMaxPotentialOccupancyBlockSize` only accept a single optional argument,
`max_threads_per_block`, this function has three different variants:

  - When no positional arguments are specified, only the hardware constraints on
    occupancy are considered (like CUDA's implementation without any arguments).
  - When `max_threads` is specified, the total number of threads is also limited.
    If `strict` is `true`, this acts as a hard upper bound on the thread count.
    Otherwise, an extra block of threads may be scheduled, allowing loops that
    can handle surplus threads to assign every loop iteration to a unique thread.
  - When `max_threads_per_block` and `max_blocks` are both specified, the block
    and grid dimensions of the launch configuration are individually limited
    (similar to CUDA's implementation, but with the addition of `max_blocks`).

In each case, `max_waves` can be specified to control how many waves of blocks
are scheduled. Kernels with uneven load distributions (e.g., due to conditional
branches or nonuniform memory accesses) may require multiple waves to prevent
some multiprocessors from idling while others are still active. However, block
scheduling also adds measurable latency, so the number of waves should not be
increased unless load redistribution is necessary. By default, `max_waves` is
set to 1, but the `CLIMA_CUDA_MAX_WAVES` environment variable overrides this.

A `granularity` may also be specified, in which case only block sizes that are
whole multiples of it are considered. This defaults to the warp size, which is
the smallest unit a block can usefully be made of; a loop that hands each slice
to a sub-block needs the coarser unit of one sub-block (see
the whole subscopes invariant in `DataLayouts.subscope_indices`).

A kernel that uses dynamic shared memory is given it as a [`DynamicShmem`](@ref) `shmem`,
which every block size considered has to accommodate along with the kernel's static
shared memory.

Like `CUDA.launch_configuration`, this returns a `NamedTuple` containing the
optimal number of threads per block and the optimal number of blocks, and it
also returns a symbol identifying the dominant factor that limits occupancy:

  - `:user` when limited by the user-specified constraints (e.g., `max_threads`)
  - `:num_blocks` when limited by the number of blocks per multiprocessor
  - `:num_threads` when limited by the number of threads per multiprocessor
  - `:registers` when limited by the register pressure of each thread
  - `:shared_memory` when limited by the shared memory requirements of each block

Computing a launch configuration requires querying the C driver for the kernel's
register pressure and shared memory requirements, which adds measurable latency
if done more than once per kernel, so the result is cached in a `Dict`. A
lock is used to prevent multiple threads from updating the cache simultaneously,
since a `Dict` should never be read as it is being rehashed.
"""
function launch_configuration(
    kernel::CUDA.HostKernel,
    config_args...;
    strict = true,
    max_waves = MAX_WAVES[],
    granularity = nothing,
    shmem::DynamicShmem = NO_DYNAMIC_SHMEM,
)
    cu_func = kernel.fun
    cache_key = (
        cu_func.handle,
        strict,
        max_waves,
        something(granularity, -1),
        shmem,
        get(config_args, 1, -1),
        get(config_args, 2, -1),
    )
    return @lock LAUNCH_CONFIGURATION_CACHE_LOCK get!(
        LAUNCH_CONFIGURATION_CACHE,
        cache_key,
    ) do
        uncached_launch_configuration(
            cu_func, strict, max_waves, granularity, shmem, config_args...,
        )
    end
end

const reported_stats = Dict()
# Cache of kernel names, keyed by the same pair that identifies a kernel
# specialization: the kernel function's type and the tuple type of its
# arguments. Concretely typed so that the lookup in `auto_launch!` is
# inference-friendly (see the comment there).
const kernel_names = Dict{Tuple{Type, Type}, Union{String, Nothing}}()

# A compile-time constant, not a `Ref`, so that inference folds away the
# development-only statistics block in `auto_launch!` (which is otherwise
# type-unstable and fails the fused-broadcast `@test_opt`). Set
# `CLIMA_COLLECT_KERNEL_STATS` before the extension precompiles to enable it,
# or redefine this method (as perf/stress_test_compiler.jl does).
collect_kernel_stats() = COLLECT_KERNEL_STATS
# Every kernel launched while RECORD_KERNELS[] is true, for tests of kernel resources.
const RECORD_KERNELS = Ref(false)
const RECORDED_KERNELS = CUDA.HostKernel[]

function _memory_bytes(memory, key::Symbol)
    if hasproperty(memory, key)
        return Int(getproperty(memory, key))
    elseif memory isa NamedTuple && haskey(memory, key)
        return Int(memory[key])
    else
        return 0
    end
end

# Robustly parse boolean-like environment variables
function _getenv_bool(var::AbstractString; default::Bool = false)
    raw = get(ENV, var, nothing)
    raw === nothing && return default
    s = lowercase(strip(String(raw)))
    if s in ("1", "true", "t", "yes", "y", "on")
        return true
    elseif s in ("0", "false", "f", "no", "n", "off")
        return false
    else
        # fall back to parse as integer (non-zero -> true)
        try
            return parse(Int, s) != 0
        catch
            @warn "Unrecognized boolean env var value; using default" var = var val = raw default =
                default
            return default
        end
    end
end

# Parse integer-valued environment variables. Warns and falls back to the
# default on unparsable input, and errors on a value outside `min:max` so a
# misconfigured cap fails loudly at load instead of being silently clipped or
# failing at the first kernel launch. Only called from __init__: kernel-launch
# paths must read the Refs below, not ENV.
function _getenv_int(
    var::AbstractString,
    default::Int;
    min::Int = 1,
    max::Int = typemax(Int),
)
    raw = get(ENV, var, nothing)
    raw === nothing && return default
    value = tryparse(Int, strip(String(raw)))
    if isnothing(value)
        @warn "Unrecognized integer env var value; using default" var = var val =
            raw default = default
        return default
    end
    value < min && error("$var must be ≥ $min, got $value")
    value > max && error("$var must be ≤ $max, got $value")
    return value
end

# Refs to hold settings read from environment variables in __init__: kernel
# naming from stack traces, per-launch kernel statistics, and the wave count
# used by the launch-configuration search.
const NAME_KERNELS_FROM_STACK_TRACE = Ref{Bool}(false)
const COLLECT_KERNEL_STATS =
    _getenv_bool("CLIMA_COLLECT_KERNEL_STATS"; default = false)
const MAX_WAVES = Ref{Int}(1)
const DSS_MAX_THREADS = Ref{Int}(256)

# Always reload when module is imported so precompilation doesn't make it "stick"
function __init__()
    NAME_KERNELS_FROM_STACK_TRACE[] = _getenv_bool(
        "CLIMA_NAME_CUDA_KERNELS_FROM_STACK_TRACE"; default = false,
    )
    MAX_WAVES[] = _getenv_int("CLIMA_CUDA_MAX_WAVES", 1)
    # 1024 is CUDA's threads-per-block limit; this cap feeds block sizes
    # directly, so a larger value would fail at the first kernel launch.
    DSS_MAX_THREADS[] = _getenv_int("CLIMA_DSS_MAX_THREADS", 256; max = 1024)
end

name_kernels_from_stack_trace() = NAME_KERNELS_FROM_STACK_TRACE[]

# Modules to ignore when constructing kernel names from stack traces
const IGNORE_MODULES = (
    :Base,
    :Core,
    :GPUCompiler,
    :CUDA,
    :NVTX,
    :ClimaCoreCUDAExt,
)

# Functions within ClimaCore to ignore when determining relevant stack frames
const CLIMACORE_IGNORE_FUNCS = (
    :materialize,
    :materialize!,
    :copyto!,
    :foreach,
    :unrolled_foreach,
    Symbol("macro expansion"),
)
# Helper function to check if a stack frame is relevant
@inline function is_relevant_frame(frame::Base.StackTraces.StackFrame)
    frame_method = frame.linfo isa Core.CodeInstance ? frame.linfo.def : frame.linfo
    frame_method isa Core.MethodInstance || return false
    mod = frame_method.def.module::Module
    mod === DataLayouts && return false # loop machinery below user-facing calls
    mod_name = fullname(mod)[1]
    mod_name == :ClimaCore && frame.func::Symbol ∈ CLIMACORE_IGNORE_FUNCS && return false
    return mod_name ∉ IGNORE_MODULES
end

# Extract file path from a MethodInstance as a string
@inline function fpath_from_method_instance(mi::Core.MethodInstance)
    return string(mi.def.file::Symbol)::String
end

# Arguments are converted once per launch, before its kernel is compiled: CUDA.jl v6
# converts them in a KernelCall, which also tracks the memory that they use, and CUDA.jl v5
# kernel calls skip their own conversion with convert = Val(false) (isbits closures need no
# conversion, so they skip Adapt).
#
# A compile error from a buffer that exceeds the shared memory level of its slice loop (see
# check_shmem_level) is returned here as the bytes that the loop needs (see
# shmem_level_error_bytes). The try block wraps the compilation, not the launch: a value
# that is live in a try block is boxed on the heap, which would copy the launch's arguments.
@static if isdefined(CUDA, :KernelCall)
    converted_kernel_call(f!, args) = CUDA.KernelCall(f!, args...)
    function compile_kernel(call, name; maxregs = nothing)
        try
            return compile_kernel_call(call, name, maxregs)
        catch err
            err isa CUDA_COMPILER.GPUCompiler.InvalidIRError || rethrow()
            bytes = shmem_level_error_bytes(err)
            iszero(bytes) && rethrow()
            return bytes
        end
    end
    compile_kernel_call(call, ::Nothing, maxregs) =
        CUDA.kernel_compile(call; always_inline = true, maxregs_kwarg(maxregs)...)
    compile_kernel_call(call, name, maxregs) =
        CUDA.kernel_compile(call; name, always_inline = true, maxregs_kwarg(maxregs)...)
    launch_kernel(kernel, call, threads, blocks, shmem) =
        CUDA.kernel_launch(kernel, call; threads, blocks, shmem)
else
    converted_kernel_call(f!::F!, args) where {F!} = (;
        f! = isbitstype(F!) ? f! : CUDA.cudaconvert(f!),
        args = map(CUDA.cudaconvert, args),
        source = (f!, args),
    )
    compile_kernel(call, name; maxregs = nothing) =
        compile_function(call.f!, argument_types(call), name, maxregs)
    function compile_function(f!::F!, tt, name, maxregs) where {F!}
        try
            return compile_function_call(f!, tt, name, maxregs)
        catch err
            err isa CUDA_COMPILER.GPUCompiler.InvalidIRError || rethrow()
            bytes = shmem_level_error_bytes(err)
            iszero(bytes) && rethrow()
            return bytes
        end
    end
    compile_function_call(f!, tt, ::Nothing, maxregs) =
        CUDA.cufunction(f!, tt; always_inline = true, maxregs_kwarg(maxregs)...)
    compile_function_call(f!, tt, name, maxregs) =
        CUDA.cufunction(f!, tt; name, always_inline = true, maxregs_kwarg(maxregs)...)
    argument_types(call) = Tuple{map(Core.Typeof, call.args)...}
    launch_kernel(kernel, call, threads, blocks, shmem) = GC.@preserve call kernel(
        call.args...;
        threads,
        blocks,
        shmem,
        convert = Val(false),
    )
end

"""
    auto_launch!(f!, args; [nitems], [config], [threads_s], [blocks_s], [shmem])

Launch a cuda kernel, using `launch_configuration` to determine the number of
threads and blocks.

The threads and blocks are computed from the compiled kernel by `config` if it is
given, or else so that each of `nitems` items gets its own thread; otherwise, the
suggested threads and blocks (`threads_s`, `blocks_s`) are used. The kernel is launched
with the [`DynamicShmem`](@ref) `shmem` evaluated at its number of threads per block (none
by default), which a `config` must also pass to `launch_configuration`, since the launch
configuration raises the limit on dynamic shared memory of a kernel that needs more than
the default (a launch given `threads_s` must stay within it). Every kernel that allocates
dynamic shared memory (see `scoped_static_array`) must be given enough of it.

Return `nothing` once the kernel is launched. A kernel that does not compile because a
buffer exceeds the shared memory level of its slice loop (see `check_shmem_level`) is not
launched, and the bytes it needs are returned instead (see `launch_slice_loop`).
"""
function auto_launch!(
    f!::F!,
    args;
    nitems = nothing,
    config = nothing,
    threads_s = nothing,
    blocks_s = nothing,
    shmem::DynamicShmem = NO_DYNAMIC_SHMEM,
    maxregs = nothing,
) where {F!}
    # If desired, compute a kernel name from the stack trace and store in
    # a global Dict, which serves as an in memory cache
    kernel_name = nothing
    if name_kernels_from_stack_trace()
        # Key on the kernel function's type and the tuple type of its
        # arguments, the pair that identifies a kernel specialization. Both are
        # statically known here, so building the key introduces no runtime
        # dispatch. That matters because `name_kernels_from_stack_trace()`
        # reads a `Ref` and so cannot be const-folded: inference sees this
        # branch on every launch path, and a dispatch here would show up in
        # every `@test_opt` over a kernel launch.
        #
        # Unlike the method instance it replaces, this key does not change
        # when Revise redefines the kernel function, so a revised kernel keeps
        # the name cached before the edit, including a possibly stale file and
        # line, until `empty!(kernel_names)` or a restart. Only the name is
        # stale: CUDA's compilation cache tracks the world age and recompiles
        # the revised kernel either way.
        key = (F!, typeof(args))
        kernel_name_exists = key in keys(kernel_names)
        if !kernel_name_exists
            # Construct the kernel name, ignoring modules we don't care about
            stack = stacktrace()
            first_relevant_index = findfirst(is_relevant_frame, stack)
            if !isnothing(first_relevant_index)
                # Don't include file if this is inside an NVTX annotation
                frame = stack[first_relevant_index]::Base.StackTraces.StackFrame
                func_name = string(frame.func)
                if contains(func_name, "#")
                    func_name = split(func_name, "#")[1]
                end
                frame_method =
                    frame.linfo isa Core.CodeInstance ? frame.linfo.def : frame.linfo
                fp_split =
                    splitpath(fpath_from_method_instance(frame_method::Core.MethodInstance))
                if "NVTX" in fp_split
                    fp_string = "_NVTX"
                    line_string = ""
                else
                    # Trim base directory off of file path to shorten
                    package_index = findfirst(fp_split) do part
                        startswith(part, "Clima")
                    end
                    if isnothing(package_index)
                        package_index = findfirst(p -> p == ".julia", fp_split)
                    end
                    if isnothing(package_index)
                        package_index = findfirst(p -> p == "src", fp_split)
                    end
                    if isnothing(package_index)
                        package_index = 1
                    end
                    fp_string =
                        "_FILE_" *
                        string(joinpath(fp_split[package_index:end]...))
                    line_string = "_L" * string(frame.line)
                end
                name_str = string(func_name) * fp_string * line_string
                kernel_name = replace(name_str, r"[^A-Za-z0-9]" => "_")
            end
            @debug "Using kernel name: $kernel_name"
            kernel_names[key] = kernel_name
        end
        kernel_name = kernel_names[key]
    end

    # Alternative to CUDA.@cuda that passes the kernel to launch_configuration.
    call = converted_kernel_call(f!, args)
    kernel = compile_kernel(call, kernel_name; maxregs)
    kernel isa CUDA.HostKernel || return kernel # the bytes of a buffer beyond the level
    if !isnothing(config)
        (; threads, blocks) = config(kernel)
    elseif isnothing(nitems)
        (threads, blocks) = (threads_s, blocks_s)
    else
        threads = launch_configuration(kernel, nitems; strict = false, shmem).threads
        blocks = cld(nitems, threads)
    end
    launch_kernel(kernel, call, threads, blocks, shmem(prod(threads)))
    RECORD_KERNELS[] && push!(RECORDED_KERNELS, kernel)

    if collect_kernel_stats() # only for development use
        key = (F!, typeof(args), CUDA.registers(kernel))
        # CUDA.registers(kernel) > 50 || return nothing # for debugging
        # occursin("single_field_solve_kernel", string(nameof(F!))) || return nothing
        if !haskey(reported_stats, key)
#! format: off
            s = ""
            s *= "Launching kernel $f! with following config:\n"
            nitems_str = isnothing(nitems) ? "unknown" : string(nitems)
            s *= "     nitems:         $(nitems_str)\n"
            isnothing(threads_s) || (s *= "     threads_s:      $(threads_s)\n")
            isnothing(blocks_s) || (s *= "     blocks_s:       $(blocks_s)\n")
            isnothing(threads) || (s *= "     threads:        $(threads)\n")
            isnothing(blocks) || (s *= "     blocks:         $(blocks)\n")
            (isnothing(threads_s) || isnothing(threads)) || (s *= "     Δthreads:       $(threads - prod(threads_s))\n")
            (isnothing(blocks_s) || isnothing(blocks)) || (s *= "     Δblocks:        $(blocks - prod(blocks_s))\n")
            s *= "     maxthreads:     $(CUDA.maxthreads(kernel))\n"
            s *= "     registers:      $(CUDA.registers(kernel))\n"
            isnothing(threads_s) || ( s *= "     threads_s_frac: $(prod(threads_s)/CUDA.maxthreads(kernel))\n")
            memory = CUDA.memory(kernel)
            local_bytes = _memory_bytes(memory, :local)
            shared_bytes = _memory_bytes(memory, :shared)
            const_bytes = _memory_bytes(memory, :constant)
            s *= "     memory:         $(memory)\n"
            profile_line =
                "CUDA_PROFILE: kernel=$(something(kernel_name, nameof(F!))) " *
                "registers=$(CUDA.registers(kernel)) " *
                "local=$(local_bytes) shared=$(shared_bytes) constant=$(const_bytes) " *
                "maxthreads=$(CUDA.maxthreads(kernel))"
            s *= "     $(profile_line)\n"
            @info s
            println(profile_line)
#! format: on
            reported_stats[key] = true
            # error("Oops") # for debugging
            # Main.Infiltrator.@exfiltrate # for debugging/performance optimization
        end
    end
    return nothing
end

"""
    thread_index()

Return the linear index of the current thread across all blocks:

```julia
(CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
```
"""
@inline thread_index() =
    (CUDA.blockIdx().x - UInt32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
