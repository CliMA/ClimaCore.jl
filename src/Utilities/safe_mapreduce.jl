struct NoInit end

# Block size below which the pairwise reduction switches to a sequential loop,
# matching Base.pairwise_blocksize(f, op) from Base's pairwise mapreduce.
const PAIRWISE_BLOCKSIZE = 1024

# Number of accumulators of each sequential loop, which CPUs can update in
# parallel to hide the latency of op (and vectorize when they read interleaved
# floats).
const NUM_ACCUMULATORS = 8

# Whether LLVM vectorizes @simd loops over the maxima and minima of floats, as
# version 18 (Julia 1.12) and later do.
const VECTORIZES_SIMD_MAX_AND_MIN = Base.libllvm_version >= v"18"

"""
    safe_mapreduce(f, op, itr, positions; [init], is_contiguous_itr)

Apply `f` to every element of the indexable collection `itr` and reduce the results
with `op`, like `Base.mapreduce(f, op, itr; init)`, with the guarantee that the
call can be compiled in GPU kernels. Unlike `Base.mapreduce`, this never reaches
the empty collection error path, which builds a string that cannot be compiled for
a GPU; if `init` is not provided, there must be elements to reduce. Only the elements of
`itr` at the linear indices in the unit range `positions` are reduced.

The reduction is only sequential for iterator lengths below 1024; for longer
iterators, the reduction is pairwise, so its roundoff error grows logarithmically
rather than linearly with length. Sequential reductions interleave several
accumulators, which CPUs can update in parallel, in an order that only depends on the
number of elements; each component of a reduction over `AutoBroadcaster`s then matches
the reduction of that component alone. Sums and products of values that are not
contiguous (`is_contiguous_itr = Val(false)`), and products of matrices, give each
accumulator a contiguous chunk of values in order, so that values that cancel locally
(e.g., in conservation checks) are added first, as in a sequential fold. Sums and
products of contiguous floats use `@simd` loops instead, as do maxima and minima of
floats over a unit range of indices when LLVM vectorizes them (version 18 or later).
When `init` is provided, it is combined with the result.
"""
Base.@propagate_inbounds function safe_mapreduce(
    f::F,
    op::O,
    itr,
    positions;
    init = NoInit(),
    is_contiguous_itr,
) where {F, O}
    first_index, last_index = first(positions), last(positions)
    if init isa NoInit
        @assert first_index <= last_index # positions must be nonempty if init is missing
        return mapreduce_pairwise(f, op, itr, first_index, last_index, is_contiguous_itr)
    end
    first_index > last_index && return init
    return op(
        init,
        mapreduce_pairwise(f, op, itr, first_index, last_index, is_contiguous_itr),
    )
end

# Recursively split a non-empty collection in half until each block is small
# enough to reduce with a single sequential loop.
@noconsts function mapreduce_pairwise(
    f::F,
    op::O,
    itr,
    first_index,
    last_index,
    is_contiguous_itr,
) where {F, O}
    if last_index - first_index >= PAIRWISE_BLOCKSIZE
        middle_index = (first_index + last_index) ÷ 2
        return op(
            mapreduce_pairwise(f, op, itr, first_index, middle_index, is_contiguous_itr),
            mapreduce_pairwise(f, op, itr, middle_index + 1, last_index, is_contiguous_itr),
        )
    end
    @inline value(index) = f(@inbounds itr[index])
    is_sum_or_product =
        op isa Union{typeof(+), typeof(Base.add_sum), typeof(*), typeof(Base.mul_prod)}
    T = Core.Compiler.return_type(f, Tuple{eltype(itr)})
    use_simd =
        T <: Base.IEEEFloat && (
            is_sum_or_product && is_contiguous_itr isa Val{true} ||
            VECTORIZES_SIMD_MAX_AND_MIN &&
            op isa Union{typeof(max), typeof(min)} &&
            itr isa AbstractUnitRange
        )
    # Products of matrices do not commute, so their accumulators fold values in order.
    is_matrix_product =
        op isa Union{typeof(*), typeof(Base.mul_prod)} && T <: AbstractMatrix
    use_contiguous_accumulators =
        is_sum_or_product && (is_contiguous_itr isa Val{false} || is_matrix_product)
    if use_simd
        result = value(first_index)
        @simd for index in (first_index + 1):last_index
            result = op(result, value(index))
        end
        return result
    end
    # Accumulator n folds the values at first_index + (n - 1) * spacing + step * i for
    # 0 <= i < chunk_length, and the remaining values are folded into their result.
    chunk_length = (last_index - first_index + 1) ÷ NUM_ACCUMULATORS
    (spacing, step) =
        use_contiguous_accumulators ? (chunk_length, 1) : (1, NUM_ACCUMULATORS)
    @inline @noconsts values(index) =
        ntuple(
            (@inline @noconsts n -> value(index + (n - 1) * spacing)),
            Val(NUM_ACCUMULATORS),
        )
    if chunk_length > 0
        # The first row initializes the accumulators.
        local results
        for i in 0:(chunk_length - 1)
            row = values(first_index + i * step)
            results = iszero(i) ? row : unrolled_map(op, results, row)
        end
        result = unrolled_reduce(op, results)
        index = first_index + NUM_ACCUMULATORS * chunk_length
    else
        result = value(first_index)
        index = first_index + 1
    end
    while index <= last_index
        result = op(result, value(index))
        index += 1
    end
    return result
end
