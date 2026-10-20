struct NoInit end

# Block size below which the pairwise reduction switches to a sequential loop,
# matching Base.pairwise_blocksize(f, op) from Base's pairwise mapreduce.
const PAIRWISE_BLOCKSIZE = 1024

# Number of accumulators of each sequential loop, which CPUs can update in parallel to hide
# the latency of op (and vectorize when they read interleaved floats).
const NUM_ACCUMULATORS = 8

"""
    safe_mapreduce(f, op, itr, [positions]; [init], [is_contiguous_itr])

Apply `f` to every element of the indexable collection `itr` and reduce the results
with `op`, like `Base.mapreduce(f, op, itr; init)`, with the guarantee that the
call can be compiled in GPU kernels. Unlike `Base.mapreduce`, this never reaches
the empty collection error path, which builds a string that cannot be compiled for
a GPU; if `init` is not provided, there must be elements to reduce. When the unit range
`positions` is given, only the elements of `itr` at these linear indices are reduced.

The reduction is only sequential for iterator lengths below 1024; for longer
iterators, the reduction is pairwise, so its roundoff error grows logarithmically
rather than linearly with length. Sequential reductions interleave several
accumulators, which CPUs can update in parallel. Unlike the reassociation of a `@simd`
loop, the order in which `op` is applied only depends on the number of elements, so each
component of a reduction over `AutoBroadcaster`s matches the reduction of that
component alone. Sums and products of floats over contiguous values
(`is_contiguous_itr = Val(true)`), which `f` reads without divisions, use `@simd` loops
instead, which vectorize them more efficiently. Other sums and products interleave
accumulators that each fold a contiguous chunk of values in order, so values that cancel
locally (e.g., in conservation checks) are added before others, as in a sequential fold.
When `init` is provided, it is combined with the result.
"""
Base.@propagate_inbounds function safe_mapreduce(
    f::F,
    op::O,
    itr,
    positions = firstindex(itr):lastindex(itr);
    init = NoInit(),
    is_contiguous_itr = Val(false),
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
# enough to reduce with a single sequential loop. This is not
# @propagate_inbounds, which implies @inline, so that the loop and any f inlined
# into it are optimized once here, rather than again in every inlined caller. Its result
# type only depends on argument types, so it is not constant-propagated: constant bounds
# (e.g., of a column) would infer it again, with every inlined read of a value once per
# constant index. LLVM vectorizes @simd sums and products of contiguous floats, where
# they are faster and compile faster. Over other values, reads that need divisions stop
# vectorization, and accumulators that each fold a contiguous chunk of values are up to
# twice as fast as @simd loops, while keeping sums that cancel locally about as accurate as
# sequential folds. Other reductions (e.g., max and min, whose results do not depend on
# order, and column sums of containers) are fastest with accumulators that read interleaved
# values, except products of matrices, which do not commute.
Base.@constprop :none function mapreduce_pairwise(
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
    use_simd = is_sum_or_product && T <: Base.IEEEFloat && is_contiguous_itr isa Val{true}
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
    # 0 <= i < chunk_length, and the remaining values are folded into their result. As
    # above, values and its closure are not constant-propagated, which would infer them
    # again (with every inlined read) for each accumulator and for constant spacings.
    chunk_length = (last_index - first_index + 1) ÷ NUM_ACCUMULATORS
    (spacing, step) =
        use_contiguous_accumulators ? (chunk_length, 1) : (1, NUM_ACCUMULATORS)
    @inline_types_only values(index) = ntuple(
        @inline_types_only(n -> value(index + (n - 1) * spacing)),
        Val(NUM_ACCUMULATORS),
    )
    if chunk_length > 0
        # The rows of values are read at one site, so that every read is compiled once; the
        # first row initializes the accumulators (LLVM peels that iteration).
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
