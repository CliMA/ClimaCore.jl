import CUDA
import ClimaCore.Fields
import ClimaCore.DataLayouts
import ClimaCore.DataLayouts: empty_kernel_stats

const reported_stats = Dict()
const kernel_names = IdDict()
# Call via ClimaCore.DataLayouts.empty_kernel_stats()
empty_kernel_stats(::ClimaComms.CUDADevice) = empty!(reported_stats)
collect_kernel_stats() = false

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

# Helper function to check if a stack frame is relevant
@inline function is_relevant_frame(frame::Base.StackTraces.StackFrame)
    linfo = frame.linfo
    linfo isa Core.MethodInstance || return false
    mod = linfo.def.module::Module
    mod_name = fullname(mod)[1]
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
            AbstractData,
            Field,
        };
        auto = false,
        threads_s,
        blocks_s,
        always_inline = true
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
                fp_split =
                    splitpath(fpath_from_method_instance(frame.linfo::Core.MethodInstance))
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
            config = CUDA.launch_configuration(kernel.fun)
            threads = min(nitems, config.threads)
            blocks = cld(nitems, threads)
            kernel(args...; threads, blocks) # This knows to use always_inline from above.
        end
    else
        kernel =
            CUDA.@cuda name = kernel_name always_inline = always_inline threads =
                threads_s blocks = blocks_s f!(args...)
    end

    if collect_kernel_stats() # only for development use
        key = (F!, typeof(args), CUDA.registers(kernel))
        # CUDA.registers(kernel) > 50 || return nothing # for debugging
        # occursin("single_field_solve_kernel", string(nameof(F!))) || return nothing
        if !haskey(reported_stats, key)
            @assert !isnothing(nitems)
            kernel = CUDA.@cuda always_inline = true launch = false f!(args...)
            config = CUDA.launch_configuration(kernel.fun)
            threads = min(nitems, config.threads)
            blocks = cld(nitems, threads)
            # For now, let's just collect info, later we can benchmark
#! format: off
            s = ""
            s *= "Launching kernel $f! with following config:\n"
            s *= "     nitems:         $(nitems)\n"
            isnothing(threads_s) || (s *= "     threads_s:      $(threads_s)\n")
            isnothing(blocks_s) || (s *= "     blocks_s:       $(blocks_s)\n")
            s *= "     threads:        $(threads)\n"
            s *= "     blocks:         $(blocks)\n"
            isnothing(threads_s) || (s *= "     Δthreads:       $(threads - prod(threads_s))\n")
            isnothing(blocks_s) || (s *= "     Δblocks:        $(blocks - prod(blocks_s))\n")
            s *= "     maxthreads:     $(CUDA.maxthreads(kernel))\n"
            s *= "     registers:      $(CUDA.registers(kernel))\n"
            isnothing(threads_s) || ( s *= "     threads_s_frac: $(prod(threads_s)/CUDA.maxthreads(kernel))\n")
            s *= "     memory:         $(CUDA.memory(kernel))\n"
            @info s
#! format: on
            reported_stats[key] = true
            # error("Oops") # for debugging
            # Main.Infiltrator.@exfiltrate # for debugging/performance optimization
        end
        # end
    end
    return nothing
end

function threads_via_occupancy(f!::F!, args) where {F!}
    kernel = CUDA.@cuda always_inline = true launch = false f!(args...)
    config = CUDA.launch_configuration(kernel.fun)
    return config.threads
end

"""
    config_via_occupancy(f!::F!, nitems, args) where {F!}

Returns a named tuple of `(:threads, :blocks)` that contains an approximate
optimal launch configuration for the kernel `f!` with arguments `args`, given
`nitems` total items to process.

If the number of items is greater than the minimal number of threads required for the config
suggested by `CUDA.launch_configuration` to be valid, that config is returned. Otherwise,
the threads are spread out across more SMs to improve occupancy.
"""
function config_via_occupancy(f!::F!, nitems, args) where {F!}
    kernel = CUDA.@cuda always_inline = true launch = false f!(args...)
    config = CUDA.launch_configuration(kernel.fun)
    SM_count = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    max_block_size = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
    if cld(nitems, config.threads) < config.blocks
        # gpu will not saturate, so spread out threads across more SMs
        even_distribution_threads = cld(nitems, SM_count)
        # Ensure we don't exceed max block size (usually limited by register pressure)
        # If so, attempt to halve the number of threads
        even_distribution_threads =
            even_distribution_threads > max_block_size ? div(even_distribution_threads, 2) :
            even_distribution_threads
        # it should be safe to assume even_distribution_threads < config.threads here
        threads = min(even_distribution_threads, config.threads)
        blocks = cld(nitems, threads)
    else
        threads = min(nitems, config.threads)
        blocks = cld(nitems, threads)
    end
    return (; threads, blocks)
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
