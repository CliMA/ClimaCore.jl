# Wrappers for sizes and indices that are passed to kernels as type parameters.
@inline unval(x) = x
@inline unval(::Val{x}) where {x} = x

"""
    linear_thread_idx()

Return the linear index of the current thread across all blocks, computed from
CUDA's `threadIdx`, `blockIdx` and `blockDim`.
"""
@inline linear_thread_idx() =
    (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x

#####
##### Custom partitions
#####

##### linear partition
@inline function linear_partition(nitems::Integer, n_max_threads::Integer)
    @assert nitems > 0
    threads = min(nitems, n_max_threads)
    blocks = cld(nitems, threads)
    return (; threads, blocks)
end
@inline linear_is_valid_index(i::Integer, data) = 1 ≤ i ≤ length(data)

##### Column-wise
@inline function cartesian_indices_columnwise(data)
    (_, Ni, Nj, Nh) = size(data)
    return CartesianIndices(map(Base.OneTo, (Ni, Nj, Nh)))
end

##### Element-wise (e.g., limiters)
# TODO

##### Multiple-field solve partition
@inline function cartesian_indices_multiple_field_solve(data; Nnames)
    (_, Ni, Nj, Nh) = size(data)
    return CartesianIndices(map(Base.OneTo, (Ni, Nj, Nh, Nnames)))
end
