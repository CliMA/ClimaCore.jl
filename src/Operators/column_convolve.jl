import UnrolledUtilities: unrolled_sum

"""
Represents a 1D stencil to be applied by a convolution operator.

At the moment the functionality is doubled wrt  to BandMatrixRow, but the latter
cannot be used due to the module order (Operators are defined before Matrix Fields)

"""
struct ConvolutionKernel{ld, N, T}
    entries::NTuple{N, T}
    ConvolutionKernel{ld, N, T}(entries::NTuple{N, Any}) where {ld, N, T} =
        new{ld, N, T}(entries)
end

ConvolutionKernel{ld}(entries::Vararg{Any, N}) where {ld, N} =
    ConvolutionKernel{ld, N}(entries...)

ConvolutionKernel{ld, N}(entries::Vararg{Any, N}) where {ld, N} =
    ConvolutionKernel{ld, N, promote_type(map(typeof, entries)...)}(entries)

function (row::ConvolutionKernel{ld, bw, T})(args::Vararg{Any, N}) where {ld, bw, T, N}
    N == bw || error(
        "BandMatrixRow with bandwidth $bw expected $bw arguments, but got $N",
    )

    result = unrolled_sum(Iterators.zip(row.entries, args)) do (entry, arg)
        entry * arg
    end
    return result
end

# Some indexing wrapper to allow extrapolation
# Will dispatch on the BC (and operator?) in the future
Base.@propagate_inbounds getidx_extrapolated(x, idx, value) =
    idx < 1 ? value : idx > length(x) ? value : x[idx]

"""
    apply_kernel!(out, kernel::ConvolutionKernel, in)

Applies a 1D convolution kernel on a column view

`in` and `out` are DataLayouts.VF representing single columns.

  - We may need a `getindex` wrapper that will handle BCs

"""
function apply_kernel!(out, kernel::ConvolutionKernel{ld, bw}, in) where {ld, bw}

    # Should we have `eachindex` here. Should DataLayouts implement it? will they after 
    # refactor?
    @inbounds begin
        for i in 1:length(in)
            args = unrolled_map(1:bw) do j
                getidx_extrapolated(in, i + ld + j - 1, zero(eltype(in)))
            end
            out[i] = kernel(args...)
        end
    end
end
