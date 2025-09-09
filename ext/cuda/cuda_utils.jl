import CUDA
import ClimaCore.Fields
import ClimaCore.DataLayouts
import ClimaCore.DataLayouts: empty_kernel_stats
import ClimaCore.DebugOnly: profile_rename_kernel_names

const reported_stats = Dict()
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
    if auto
        @assert !isnothing(nitems)
        if nitems ≥ 0
            if profile_rename_kernel_names()
                stacktraceStr = String(StackTrace())
                @info stacktraceStr
                kernel = CUDA.@cuda always_inline = true launch = false name = stacktraceStr f!(args...)
            else
                kernel = CUDA.@cuda always_inline = true launch = false f!(args...)
            end
            config = CUDA.launch_configuration(kernel.fun)
            threads = min(nitems, config.threads)
            blocks = cld(nitems, threads)
            kernel(args...; threads, blocks) # This knows to use always_inline from above.
        end
    else
        if profile_rename_kernel_names()
            stack_strs = [string(i.func) for i in stacktrace()]
            # do some filtering on the names to remove unhelpful FirstOrderOneSided
            unhelpful_names = ["auto_launch", "macro expansion"]
            for name in unhelpful_names
                deleteat!(stack_strs, findall(item -> occursin(name, item), stack_strs))
            end
            kernel_name = join(stack_strs, "_")
            kernel = CUDA.@cuda always_inline = always_inline threads = threads_s blocks = blocks_s name = kernel_name f!(args...)
        else
            kernel = CUDA.@cuda always_inline = always_inline threads = threads_s blocks = blocks_s f!(args...)
        end
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
