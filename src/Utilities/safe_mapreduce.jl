struct NoInit end

# Block size below which the pairwise reduction switches to a sequential loop,
# matching Base.pairwise_blocksize(f, op) from Base's pairwise mapreduce.
const PAIRWISE_BLOCKSIZE = 1024

"""
    safe_mapreduce(f, op, itr; [init])

Analogue of `Base.mapreduce(f, op, itr; [init])` for an indexable collection
`itr`, with the additional guarantee that it can be compiled in GPU kernels.
Unlike `Base.mapreduce`, this never reaches the empty collection error path,
which builds a string that cannot be compiled for a GPU; if `init` is not
provided, this assumes that `itr` is nonempty.

When `init` is available, the reduction is a sequential left fold seeded by
`init`. Otherwise, the reduction is only sequential for iterator lengths below
1024; for longer iterators, the reduction is pairwise, so its roundoff error
grows logarithmically rather than linearly with respect to length. Sequential
reductions use `@simd` loops, meaning that associativity is not guaranteed.
"""
Base.@propagate_inbounds function safe_mapreduce(
    f::F,
    op::O,
    itr;
    init = NoInit(),
) where {F, O}
    first_index, last_index = firstindex(itr), lastindex(itr)
    if init isa NoInit
        @assert first_index <= last_index # itr must be nonempty if init is missing
        return mapreduce_pairwise(f, op, itr, first_index, last_index)
    end
    value = init
    @simd for index in first_index:last_index
        value = op(value, f(@inbounds itr[index]))
    end
    return value
end

# Recursively split a non-empty collection in half until each block is small
# enough to reduce with a single vectorized sequential loop.
Base.@propagate_inbounds function mapreduce_pairwise(
    f::F,
    op::O,
    itr,
    first_index,
    last_index,
) where {F, O}
    if last_index - first_index >= PAIRWISE_BLOCKSIZE
        middle_index = (first_index + last_index) ÷ 2
        return op(
            mapreduce_pairwise(f, op, itr, first_index, middle_index),
            mapreduce_pairwise(f, op, itr, middle_index + 1, last_index),
        )
    end
    value = f(@inbounds itr[first_index])
    @simd for index in (first_index + 1):last_index
        value = op(value, f(@inbounds itr[index]))
    end
    return value
end
