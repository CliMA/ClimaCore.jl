import CUDA
import ClimaCore.Fields
import ClimaCore.DataLayouts
import ClimaCore.DataLayouts: empty_kernel_stats
import ClimaCore.DebugOnly: name_kernels_from_stack_trace
import CUDA.GPUCompiler: methodinstance

const reported_stats = Dict()
const kernel_names = IdDict()
# Call via ClimaCore.DataLayouts.empty_kernel_stats()
empty_kernel_stats(::ClimaComms.CUDADevice) = empty!(reported_stats)
collect_kernel_stats() = false

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
        key = objectid(methodinstance(typeof(f!), typeof(args)))
        kernel_name_exists = key in keys(kernel_names)
        if !kernel_name_exists
            # Construct the kernel name, ignoring modules we don't care about
            ignore_modules = [
                :Base,
                :Core,
                :GPUCompiler,
                :CUDA,
                :NVTX,
                :ClimaCoreCUDAExt,
                :ClimaCore,
            ]
            stack = stacktrace()
            first_relevant_index = findfirst(stack) do frame
                frame.linfo isa Core.MethodInstance && (
                    fullname(frame.linfo.def.module)[1] ∉ ignore_modules
                )
            end
            if !isnothing(first_relevant_index)
                # Don't include file if this is inside an NVTX annotation
                frame = stack[first_relevant_index]
                func_name = string(frame.func)
                if contains(func_name, "#")
                    func_name = split(func_name, "#")[1]
                end
                file_path = frame.linfo.def.file
                fp_split = split(string(file_path), "/")
                if "NVTX" in fp_split
                    fp_string = "_NVTX"
                    line_string = ""
                else
                    # Trim base directory off of file path
                    package_index = findfirst(fp_split) do part
                        startswith(part, "Clima")
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
