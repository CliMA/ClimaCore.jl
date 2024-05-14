import CUDA
import ClimaCore.Fields
import ClimaCore.DataLayouts
import ClimaCore.DataLayouts: empty_kernel_stats

get_n_items(field::Fields.Field) =
    prod(size(parent(Fields.field_values(field))))
get_n_items(data::DataLayouts.AbstractData) = prod(size(parent(data)))
get_n_items(arr::AbstractArray) = prod(size(parent(arr)))
get_n_items(tup::Tuple) = prod(tup)

const reported_stats = Dict()
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
        threads_s,
        blocks_s,
        always_inline = true
    )

Launch a cuda kernel, using `CUDA.launch_configuration`
to determine the number of threads/blocks.

Suggested threads and blocks (`threads_s`, `blocks_s`) can be given
to benchmark compare against auto-determined threads/blocks.
"""
function auto_launch!(
    f!::F!,
    args,
    data;
    threads_s,
    blocks_s,
    always_inline = true,
) where {F!}
    nitems = get_n_items(data)
    # For now, we'll simply use the
    # suggested threads and blocks:
    kernel =
        CUDA.@cuda always_inline = always_inline threads = threads_s blocks =
            blocks_s f!(args...)

    if collect_kernel_stats() # only for development use
        key = (F!, typeof(args))
        if !haskey(reported_stats, key)
            kernel = CUDA.@cuda always_inline = true launch = false f!(args...)
            config = CUDA.launch_configuration(kernel.fun)
            threads = min(nitems, config.threads)
            blocks = cld(nitems, threads)
            s = ""
            s *= "Launching kernel $f! with following config:\n"
            s *= "     nitems:         $(nitems)\n"
            s *= "     threads_s:      $(threads_s)\n"
            s *= "     blocks_s:       $(blocks_s)\n"
            s *= "     threads:        $(threads)\n"
            s *= "     blocks:         $(blocks)\n"
            s *= "     Δthreads:       $(threads - prod(threads_s))\n"
            s *= "     Δblocks:        $(blocks - prod(blocks_s))\n"
            s *= "     maxthreads:     $(CUDA.maxthreads(kernel))\n"
            s *= "     registers:      $(CUDA.registers(kernel))\n"
            s *= "     threads_s_frac: $(prod(threads_s)/CUDA.maxthreads(kernel))\n"
            s *= "     memory:         $(CUDA.memory(kernel))\n"
            @info s
            reported_stats[key] = true
        end
        # For now, let's just collect info, later we can benchmark
        # kernel(args...; threads, blocks) # This knows to use always_inline from above.
    end
end

"""
    kernel_indexes(n)
Return a tuple of indexes from the kernel,
where `n` is a tuple of max lengths along each
dimension of the accessed data.
"""
function kernel_indexes(n::Tuple)
    tidx = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    inds = if 1 ≤ tidx ≤ prod(n)
        CartesianIndices(map(x -> Base.OneTo(x), n))[tidx].I
    else
        ntuple(x -> -1, length(n))
    end
    return inds
end

"""
    valid_range(inds, n)
Returns a `Bool` indicating if the thread index
is in the valid range, based on `inds` (the result
of `kernel_indexes`) and `n`, a tuple of max lengths
along each dimension of the accessed data.
```julia
function kernel!(data, n)
    inds = kernel_indexes(n)
    if valid_range(inds, n)
        do_work!(data[inds...])
    end
end
```
"""
valid_range(inds::NTuple, n::Tuple) = all(i -> 1 ≤ inds[i] ≤ n[i], 1:length(n))
function valid_range(n::Tuple)
    inds = kernel_indexes(n)
    return all(i -> 1 ≤ inds[i] ≤ n[i], 1:length(n))
end
