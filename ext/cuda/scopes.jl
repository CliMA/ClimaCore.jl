const THREADS_PER_WARP = 32
const MAX_WARPS_PER_BLOCK = 32

# Only check the first launch: device attributes are fixed for a given device,
# and querying them on every launch would add measurable latency.
const device_assumptions_checked = Ref(false)
function check_device_assumptions()
    device_assumptions_checked[] && return nothing
    device = CUDA.device()
    if (
        THREADS_PER_WARP != CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_WARP_SIZE) ||
        MAX_WARPS_PER_BLOCK * THREADS_PER_WARP !=
        CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    )
        major = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        minor = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        throw(ArgumentError("Compute Capability $major.$minor is not supported"))
    end
    device_assumptions_checked[] = true
    return nothing
end

DataLayouts.DataScope(::Type{<:CUDA.CuArray}) = ThisHost()
DataLayouts.DataScope(::Type{<:CUDA.CuDeviceArray{<:Any, <:Any, A}}) where {A} =
    A == CUDA.AS.Local ? DataLayouts.ThisThread() :
    A == CUDA.AS.Shared ? ThisBlock() : ThisKernel()

"""
    ThisHost()

[`DataScope`](@ref) that represents the host device for a GPU. This scope is
assigned to any [`DataLayout`](@ref) backed by a `CuArray`, and it is replaced
with its device-side analogue [`ThisKernel`](@ref) through `Adapt.jl`. Aside
from array allocations, other standard `DataScope` operations are not supported.
"""
struct ThisHost <: DataLayouts.DataScope end

DataLayouts.num_threads(::ThisHost) = throw(ArgumentError("Cannot get num_threads on host"))
DataLayouts.thread_rank(::ThisHost) = throw(ArgumentError("Cannot get thread_rank on host"))
DataLayouts.scoped_array(::ThisHost, ::Type{T}, dims) where {T} =
    CUDA.CuArray{T}(undef, dims)

"""
    ThisKernel()

[`DataScope`](@ref) that represents all available threads on a GPU. Operations
that require synchronizations or array allocations are not supported.

NOTE: This assumes that kernels are always launched with one-dimensional grids.
Support for multidimensional grids may be added in a future release.
"""
struct ThisKernel <: DataLayouts.DataScope end

DataLayouts.partition(::ThisKernel) = ThisBlock()
DataLayouts.num_partitions(::ThisKernel) = CUDA.gridDim().x
DataLayouts.partition_rank(::ThisKernel) = CUDA.blockIdx().x

"""
    ThisCooperativeGroup

Abstract type that represents a "cooperative group" from the
[`CG`](https://cuda.juliagpu.org/stable/development/kernel/#Cooperative-groups)
module in `CUDA.jl`, which is built on top of the `cooperative_groups`
[extension](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cooperative-groups)
that comes prepackaged with CUDA.
"""
abstract type ThisCooperativeGroup <: DataLayouts.DataScope end

"""
    ThisBlock()

[`DataScope`](@ref) that represents one thread block of [`ThisKernel`](@ref).
Operations that require dynamically-sized array allocations are not supported.

NOTE: This assumes that kernels are always launched with one-dimensional blocks.
Support for multidimensional blocks may be added in a future release.
"""
struct ThisBlock <: ThisCooperativeGroup end

DataLayouts.partition(::ThisBlock) = ThisWarp()
DataLayouts.num_threads(::ThisBlock) = CUDA.blockDim().x
DataLayouts.thread_rank(::ThisBlock) = CUDA.threadIdx().x
DataLayouts.synchronize(::ThisBlock) = CUDA.sync_threads()
DataLayouts.scoped_static_array(::ThisBlock, ::Type{T}, dims) where {T} =
    CUDA.CuStaticSharedArray(T, dims)

"""
    ThisSubBlock{N}()

[`DataScope`](@ref) that represents `N` threads in [`ThisBlock`](@ref), where
`N` is typically a power of two.
"""
struct ThisSubBlock{N} <: ThisCooperativeGroup end

"""
    ThisWarp()

Special case of [`ThisSubBlock`](@ref) that represents an entire warp.
Operations that require dynamically-sized array allocations are not supported.
"""
const ThisWarp = ThisSubBlock{THREADS_PER_WARP}

DataLayouts.partition(::ThisSubBlock{N}) where {N} =
    N < 4 ? DataLayouts.ThisThread() : ThisSubBlock{N ÷ 2}()
DataLayouts.num_threads(::ThisSubBlock{N}) where {N} = N
DataLayouts.thread_rank(::ThisSubBlock{N}) where {N} =
    N > THREADS_PER_WARP ? (DataLayouts.thread_rank(ThisBlock()) - 1) % N + 1 :
    N < THREADS_PER_WARP ? (CUDA.laneid() - 1) % N + 1 : CUDA.laneid()
DataLayouts.synchronize(::ThisSubBlock{N}) where {N} =
    N > THREADS_PER_WARP ? DataLayouts.synchronize(ThisBlock()) : CUDA.sync_warp()

# Assign threads in a sub-block one slice of an array shared across their block.
function DataLayouts.scoped_static_array(scope::ThisSubBlock, ::Type{T}, dims) where {T}
    max_subblocks = MAX_WARPS_PER_BLOCK * DataLayouts.num_subscopes(scope, ThisWarp())
    array = DataLayouts.scoped_static_array(ThisBlock(), T, (dims..., max_subblocks))
    subblock_index = DataLayouts.subscope_rank(scope, ThisBlock())
    return @inbounds view(array, ntuple(Returns(:), Val(length(dims)))..., subblock_index)
end

# The last sub-block in a block may be only partially filled, so its active
# thread count is computed from the block's total. This is not the same as using
# CUDA.active_mask, which may be inconsistent across the lanes of a warp (see
# https://stackoverflow.com/questions/54055195).
num_active_threads(scope) = DataLayouts.num_threads(scope)
function num_active_threads(scope::ThisSubBlock)
    max_active_threads = DataLayouts.num_threads(scope)
    block_offset = (DataLayouts.subscope_rank(scope, ThisBlock()) - 1) * max_active_threads
    return clamp(DataLayouts.num_threads(ThisBlock()) - block_offset, 0, max_active_threads)
end

# A strided view of CartesianIndices is a ReshapedArray whose bounds checking
# cannot be compiled in a GPU kernel, so use a reshape-free wrapper instead. The
# wrapper must be indexable rather than a lazy generator because safe_mapreduce
# folds over the positions of each thread's subset of indices.
struct StridedCartesianIndices{N, I <: CartesianIndices{N}, V} <:
       AbstractVector{CartesianIndex{N}}
    indices::I
    view_range::V
end

Base.size(strided::StridedCartesianIndices) = (length(strided.view_range),)
Base.@propagate_inbounds Base.getindex(strided::StridedCartesianIndices, n::Integer) =
    strided.indices[strided.view_range[n]]

# Iterate by advancing the range and doing a single Cartesian lookup per point,
# like a generator over the range would. The AbstractArray method of
# Base.iterate instead calls getindex with a position, and that extra layer of
# indexing stays live when bounds checks are forced.
Base.@propagate_inbounds function Base.iterate(strided::StridedCartesianIndices, state...)
    next = iterate(strided.view_range, state...)
    isnothing(next) && return nothing
    return (@inbounds strided.indices[next[1]]), next[2]
end

# Point loops must not run under @simd, whose loop restructuring makes kernels
# measurably slower: with bounds checks forced, it inflates the launch latency
# by ~20% (the per-point index conversion is repeated multiple times per loop).
DataLayouts.simd_over_indices(::StridedCartesianIndices) = false

@inline function DataLayouts.subscope_index_view(
    ::Union{ThisKernel, ThisCooperativeGroup},
    indices::CartesianIndices,
    view_range,
)
    @boundscheck checkbounds(indices, view_range)
    return StridedCartesianIndices(indices, view_range)
end
