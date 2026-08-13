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

Collection of hardware properties from the CUDA API and CUDA Runtime Headers,
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
        max_shmem_per_block = attrs.max_shmem_per_block + attrs.reserved_shmem_per_block
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

# cudaOccMaxPotentialOccupancyBlockSize times the maximum number of waves,
# optimized for single-stream execution of both small and large workloads
# https://gitlab.com/nvidia/headers/cuda-individual/cudart/-/blob/main/cuda_occupancy.h#L1865-1965
function uncached_launch_configuration(cu_func, strict, default_max_waves, config_args...)
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

    # Launch a single wave of blocks unless a caller asks for more. This
    # replaces a heuristic that launched fld(regs_per_thread, 48) + 1 waves, on
    # the assumption that higher register pressure leads to uneven load
    # distributions. An A100 sweep over max_waves in (1, 2, 4) found one wave to
    # be at least as fast throughout and clearly fastest for reductions, where
    # the equal-shape lazy reduction went from 145 to 129 us, so that assumption
    # was costing more in block-scheduling than it recovered.
    max_waves = something(default_max_waves, 1)

    # Iterating from small to large block sizes, prefer configurations with more
    # total threads; on ties, prefer larger blocks (fewer blocks means less
    # block-scheduling latency), but never accept a new configuration with fewer
    # blocks unless it still has at least one block per multiprocessor. This
    # spreads small workloads across as many multiprocessors as possible, while
    # also confining large workloads to as few blocks as possible.
    (best_threads_per_block, best_num_blocks, best_limit) = (0, 0, :user)
    for warps_per_block in 1:cld(max_threads_per_block, attrs.threads_per_warp)
        threads_per_block =
            min(warps_per_block * attrs.threads_per_warp, max_threads_per_block)
        (max_blocks_per_wave, active_blocks_limit) =
            max_active_blocks(threads_per_block, regs_per_thread, shmem_per_block)
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

    # Compare searches unaffected by config_args against CUDA's implementation.
    if DataLayouts.VALIDATE_LAUNCH_CONFIGURATIONS[] && user_constraints_ignorable
        min_blocks_per_wave = cld(best_num_blocks, max_waves)
        cuda_config = (; blocks = min_blocks_per_wave, threads = best_threads_per_block)
        @assert CUDA.launch_configuration(cu_func) == cuda_config
    end

    return (;
        threads = best_threads_per_block,
        blocks = best_num_blocks,
        limit = best_limit,
    )
end

const LAUNCH_CONFIGURATION_CACHE =
    IdDict{Any, @NamedTuple{threads::Int, blocks::Int, limit::Symbol}}()
const LAUNCH_CONFIGURATION_CACHE_LOCK = ReentrantLock()

"""
    launch_configuration(f, args; max_waves = nothing)
    launch_configuration(f, args, max_threads; strict = true, max_waves = nothing)
    launch_configuration(f, args, max_threads_per_block, max_blocks; max_waves = nothing)

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
increased unless load redistribution is necessary. Setting `max_waves` to
`nothing` gives it a dynamic value based on the kernel's register pressure.

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
if done more than once per kernel, so the result is cached in an `IdDict`. A
lock is used to prevent multiple threads from updating the cache simultaneously,
since an `IdDict` should never be read as it is being rehashed.
"""
function launch_configuration(
    f::F,
    args,
    config_args...;
    strict = true,
    max_waves = nothing,
) where {F}
    cu_func = (CUDA.@cuda always_inline = true launch = false f(args...)).fun
    cache_key = (cu_func, strict, max_waves, config_args...)
    lock(LAUNCH_CONFIGURATION_CACHE_LOCK)
    try
        return get!(LAUNCH_CONFIGURATION_CACHE, cache_key) do
            uncached_launch_configuration(cache_key...)
        end
    finally
        unlock(LAUNCH_CONFIGURATION_CACHE_LOCK)
    end
end

threads_via_occupancy(f::F, args) where {F} = launch_configuration(f, args).threads
function config_via_occupancy(f::F, nitems, args) where {F}
    threads_per_block = launch_configuration(f, args, nitems; strict = false).threads
    return (; threads = threads_per_block, blocks = cld(nitems, threads_per_block))
end

const reported_stats = Dict()
const kernel_names = IdDict()

collect_kernel_stats() = false

# --- Register pressure diagnostics -------------------------------------------
#
# A kernel that reaches the architectural register cap has zero headroom: every
# additional live value must spill to local (off-chip) memory. That cliff is
# invisible today -- the model runs, just slowly -- and it is reached by the
# COMBINATION of what several packages contribute to one fused broadcast, so no
# single package sees it coming. Making it observable is cheap, because
# `launch_configuration` already queries register pressure per kernel.
#
# Controlled by `CLIMACORE_REGISTER_PRESSURE`:
#   off   (default) no cost beyond an env lookup
#   warn            log once per kernel
#   error           raise, so CI cannot ignore it
#
# `CLIMACORE_REGISTER_PRESSURE_LIMIT` sets the register threshold (default 255,
# the sm_20+ architectural maximum). Lower it to catch kernels approaching the
# cap rather than sitting on it.
#
# `CLIMACORE_REGISTER_PRESSURE_IGNORE` is a comma-separated list of substrings;
# a kernel whose name matches any of them is skipped. Without an escape hatch a
# hard error gets switched off wholesale the first time it fires on a kernel
# somebody already knows about.

const REGISTER_PRESSURE_REPORTED = Set{Any}()
const REGISTER_PRESSURE_LOCK = ReentrantLock()

# The architectural cap on registers per thread for sm_20 and later. Above this
# ptxas has no choice but to spill.
const MAX_REGISTERS_PER_THREAD = 255

function register_pressure_action()
    raw = lowercase(strip(get(ENV, "CLIMACORE_REGISTER_PRESSURE", "off")))
    (isempty(raw) || raw in ("off", "false", "0", "no")) && return :off
    raw in ("warn", "warning", "true", "1", "yes") && return :warn
    raw in ("error", "strict") && return :error
    @warn "Unrecognized CLIMACORE_REGISTER_PRESSURE=$(raw); treating as off" maxlog = 1
    return :off
end

function register_pressure_limit()
    raw = get(ENV, "CLIMACORE_REGISTER_PRESSURE_LIMIT", nothing)
    raw === nothing && return MAX_REGISTERS_PER_THREAD
    parsed = tryparse(Int, strip(raw))
    if parsed === nothing || parsed <= 0
        @warn "Invalid CLIMACORE_REGISTER_PRESSURE_LIMIT=$(raw); using $(MAX_REGISTERS_PER_THREAD)" maxlog =
            1
        return MAX_REGISTERS_PER_THREAD
    end
    return parsed
end

function register_pressure_ignored(name)
    raw = get(ENV, "CLIMACORE_REGISTER_PRESSURE_IGNORE", "")
    isempty(strip(raw)) && return false
    s = string(name)
    return any(p -> !isempty(p) && occursin(p, s), strip.(split(raw, ",")))
end

"""
    measure_spill(f!, args)

Exact per-thread spill (stores, loads) in bytes for this kernel, or `nothing`.

`CUDA.memory(k).local` cannot answer this: it is the stack frame INCLUDING spill
slots, so a kernel with a large returned value looks identical to one that
spills. Only `ptxas -v` separates them. That costs a recompile, so it runs only
for kernels that already tripped the register threshold -- a handful per run --
and the result is cached with the warning.

`dump_module = true` is required: without it only the entry function is emitted,
ptxas cannot resolve the called device functions, and the register/spill numbers
that come back are wrong rather than absent.
"""
function measure_spill(f!::F!, args) where {F!}
    ptxas = try
        only(CUDA.ptxas().exec)
    catch
        return nothing
    end
    (ptxas === nothing || !isfile(ptxas)) && return nothing
    try
        gargs = map(CUDA.cudaconvert, args)
        tt = Tuple{map(Core.Typeof, gargs)...}
        io = IOBuffer()
        CUDA.code_ptx(io, f!, tt; kernel = true, always_inline = true, dump_module = true)
        ptx_file = tempname() * ".ptx"
        write(ptx_file, String(take!(io)))
        cap = CUDA.capability(CUDA.device())
        arch = "sm_$(cap.major)$(cap.minor)"
        out, err = IOBuffer(), IOBuffer()
        cmd = `$ptxas -arch=$arch -v -o $(tempname()).cubin $ptx_file`
        proc = run(pipeline(ignorestatus(cmd); stdout = out, stderr = err))
        success(proc) || return nothing
        log = String(take!(err)) * String(take!(out))
        grab(re) = (m = match(re, log); m === nothing ? 0 : parse(Int, m[1]))
        return (
            stores = grab(r"(\d+) bytes spill stores"),
            loads = grab(r"(\d+) bytes spill loads"),
        )
    catch err
        @debug "spill measurement failed" err
        return nothing
    end
end

"""
    check_register_pressure(kernel, kernel_name, f!, args)

Report kernels at or above the register limit, once each.

Reports registers and local memory. NOTE that local memory is the stack frame
INCLUDING any spill slots, not spill alone -- CUDA.jl exposes no way to separate
them, and a kernel can carry a large stack frame while spilling nothing (the
returned value of a microphysics tendency function does exactly that). So this
flags "no headroom left", which is exact, rather than "is spilling", which is
not measurable here. Confirm actual spilling with `ptxas -v` or Nsight Compute.
"""
function check_register_pressure(kernel, kernel_name, f!::F!, args) where {F!}
    action = register_pressure_action()
    action === :off && return nothing
    name = something(kernel_name, nameof(F!))
    register_pressure_ignored(name) && return nothing

    registers = CUDA.registers(kernel)
    registers >= register_pressure_limit() || return nothing

    # Deduplicate warnings only. An error must always raise: otherwise a kernel
    # that was warned about earlier in the process could never fail the build,
    # which silently defeats CLIMACORE_REGISTER_PRESSURE=error.
    if action === :warn
        key = (objectid(f!), name, registers)
        lock(REGISTER_PRESSURE_LOCK)
        try
            key in REGISTER_PRESSURE_REPORTED && return nothing
            push!(REGISTER_PRESSURE_REPORTED, key)
        finally
            unlock(REGISTER_PRESSURE_LOCK)
        end
    end

    local_bytes = _memory_bytes(CUDA.memory(kernel), :local)
    limit = register_pressure_limit()
    spill = measure_spill(f!, args)
    spill_str = if spill === nothing
        "spill: UNMEASURED (ptxas unavailable); $(local_bytes) bytes local memory, " *
        "which is stack frame INCLUDING spill slots and so cannot prove spilling"
    elseif spill.stores > 0
        "SPILLING $(spill.stores) bytes stored / $(spill.loads) loaded per thread"
    else
        "not spilling (0 bytes), but with no headroom left"
    end
    msg = join(
        [
            "Kernel `$(name)`: $(registers) registers per thread " *
            "(limit $(limit), architectural max $(MAX_REGISTERS_PER_THREAD)) -- " *
            "$(spill_str).",
            "There is no headroom: any further register pressure, from this package " *
            "or any other package contributing to this broadcast, must spill to " *
            "off-chip memory.",
            "Set CLIMACORE_REGISTER_PRESSURE=off to disable, or add a substring of " *
            "the kernel name to CLIMACORE_REGISTER_PRESSURE_IGNORE.",
        ],
        "\n",
    )
    action === :error ? error(msg) : @warn msg
    return nothing
end


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

# Create a ref to hold the setting determining whether to name kernels from
# stack trace
const NAME_KERNELS_FROM_STACK_TRACE = Ref{Bool}(false)

# Always reload when module is imported so precompilation doesn't make it "stick"
function __init__()
    NAME_KERNELS_FROM_STACK_TRACE[] = _getenv_bool(
        "CLIMA_NAME_CUDA_KERNELS_FROM_STACK_TRACE"; default = false,
    )
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

# Functions withing ClimaCore to ignore when determining relevant stack frames
const CLIMACORE_IGNORE_FUNCS =
    (:materialize, :materialize!, :foreach, :unrolled_foreach, Symbol("macro expansion"))
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

"""
    auto_launch!(f!::F!, args,
        ::Union{
            Int,
            NTuple{N, <:Int},
            AbstractArray,
            DataLayout,
            Field,
        };
        auto = false,
        threads_s,
        blocks_s,
        always_inline = true,
        shmem = 0,
    )

Launch a cuda kernel, using `CUDA.launch_configuration` (if `auto=true`)
to determine the number of threads/blocks.

Suggested threads and blocks (`threads_s`, `blocks_s`) can be given
to benchmark compare against auto-determined threads/blocks (if `auto=false`).
"""
function auto_launch!(
    f!::F!,
    args,
    nitems::Union{Integer, Nothing} = nothing;
    auto = false,
    threads_s = nothing,
    blocks_s = nothing,
    always_inline = true,
    caller = :unknown,
    shmem = 0,
) where {F!}
    # If desired, compute a kernel name from the stack trace and store in
    # a global Dict, which serves as an in memory cache
    kernel_name = nothing
    if name_kernels_from_stack_trace()
        # Create a key from the method instance and types of the args
        key = objectid(CUDA.methodinstance(typeof(f!), typeof(args)))
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

    if auto
        @assert !isnothing(nitems)
        if nitems ≥ 0
            # Note: `name = nothing` here will revert to default behavior
            kernel = CUDA.@cuda name = kernel_name always_inline = true launch =
                false f!(args...)
            config = launch_configuration(f!, args)
            threads = min(nitems, config.threads)
            blocks = cld(nitems, threads)
            kernel(args...; threads, blocks) # This knows to use always_inline from above.
            check_register_pressure(kernel, kernel_name, f!, args)
        end
    else
        kernel =
            CUDA.@cuda name = kernel_name always_inline = always_inline threads =
                threads_s blocks = blocks_s shmem = shmem f!(args...)
        check_register_pressure(kernel, kernel_name, f!, args)
    end

    if collect_kernel_stats() # only for development use
        key = (F!, typeof(args), CUDA.registers(kernel))
        # CUDA.registers(kernel) > 50 || return nothing # for debugging
        # occursin("single_field_solve_kernel", string(nameof(F!))) || return nothing
        if !haskey(reported_stats, key)
            kernel = CUDA.@cuda always_inline = true launch = false f!(args...)
            config = launch_configuration(f!, args)
            threads = isnothing(nitems) ? nothing : min(nitems, config.threads)
            blocks = isnothing(nitems) ? nothing : cld(nitems, threads)
            # For now, let's just collect info, later we can benchmark
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
        # end
    end
    return nothing
end

"""
    thread_index()

Return the threadindex:
```
(CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
```
"""
@inline thread_index() =
    (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x

"""
    kernel_indexes(tidx, n)
Return a tuple of indexes from the kernel,
where `tidx` is the cuda thread index and
`n` is a tuple of max lengths along each
dimension of the accessed data.
"""
Base.@propagate_inbounds kernel_indexes(tidx, n::Tuple) =
    CartesianIndices(map(x -> Base.OneTo(x), n))[tidx]

"""
    valid_range(tidx, n::Int)

Returns a `Bool` indicating if the thread index
(`tidx`) is in the valid range, based on `n`, a
tuple of max lengths along each dimension of the

accessed data.
```julia
function kernel!(data, n)
    @inbounds begin
        tidx = thread_index()
        if valid_range(tidx, n)
            I = kernel_indexes(tidx, n)
            do_work!(data[I])
        end
    end
end
```
"""
@inline valid_range(tidx, n) = 1 ≤ tidx ≤ n
