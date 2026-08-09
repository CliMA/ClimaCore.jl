import UnrolledUtilities: unrolled_all, unrolled_allequal, unrolled_flatmap

const THREADS_PER_WARP = 32
const MAX_WARPS_PER_BLOCK = 32

# To reduce latency, only check device attributes before the first launch.
const DEVICE_ASSUMPTIONS_CHECKED = Ref(false)
function check_device_assumptions()
    DEVICE_ASSUMPTIONS_CHECKED[] && return true
    (; threads_per_warp, max_threads_per_block) = device_attributes()
    if THREADS_PER_WARP != threads_per_warp ||
       MAX_WARPS_PER_BLOCK * THREADS_PER_WARP != max_threads_per_block
        capability = CUDA.capability(CUDA.device())
        throw(ArgumentError("Compute Capability $(capability.major).\
                             $(capability.minor) is not supported"))
    end
    DEVICE_ASSUMPTIONS_CHECKED[] = true
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
DataLayouts.scoped_array(::ThisHost, ::Type{T}, dims; buffer = false) where {T} =
    buffer ? DataLayouts.task_reduction_buffer(CUDA.CuArray{T, 1}, dims) :
    CUDA.CuArray{T}(undef, dims)

"""
    ThisKernel()

[`DataScope`](@ref) that represents all available threads on a GPU. Operations
that require synchronizations or array allocations are not supported.

NOTE: This assumes that kernels are always launched with one-dimensional grids.
Support for multidimensional grids may be added in a future release.
"""
struct ThisKernel <: DataLayouts.DataScope end

@inline DataLayouts.partition(::ThisKernel) = ThisBlock()
@inline DataLayouts.num_partitions(::ThisKernel) = CUDA.gridDim().x
@inline DataLayouts.partition_rank(::ThisKernel) = CUDA.blockIdx().x
@inline DataLayouts.num_threads(::ThisKernel) = CUDA.gridDim().x * CUDA.blockDim().x
@inline DataLayouts.thread_rank(::ThisKernel) =
    (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
@inline DataLayouts.strided_access(::ThisKernel) = true

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

@inline DataLayouts.partition(::ThisBlock) = DataLayouts.ThisThread()
@inline DataLayouts.num_partitions(::ThisBlock) = CUDA.blockDim().x
@inline DataLayouts.partition_rank(::ThisBlock) = CUDA.threadIdx().x
@inline DataLayouts.num_threads(::ThisBlock) = CUDA.blockDim().x
@inline DataLayouts.thread_rank(::ThisBlock) = CUDA.threadIdx().x
@inline DataLayouts.synchronize(::ThisBlock) = CUDA.sync_threads()
@inline DataLayouts.scoped_static_array(::ThisBlock, ::Type{T}, dims) where {T} =
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

@inline DataLayouts.partition(::ThisSubBlock{N}) where {N} =
    N < 4 ? DataLayouts.ThisThread() : ThisSubBlock{N ÷ 2}()
@inline DataLayouts.num_threads(::ThisSubBlock{N}) where {N} = N
@inline DataLayouts.thread_rank(::ThisSubBlock{N}) where {N} =
    N > THREADS_PER_WARP ? (DataLayouts.thread_rank(ThisBlock()) - 1) % N + 1 :
    N < THREADS_PER_WARP ? (CUDA.laneid() - 1) % N + 1 : CUDA.laneid()
@inline DataLayouts.synchronize(::ThisSubBlock{N}) where {N} =
    N > THREADS_PER_WARP ? DataLayouts.synchronize(ThisBlock()) : CUDA.sync_warp()

@inline DataLayouts.is_subscope(::ThisSubBlock, ::ThisBlock) = true
@inline DataLayouts.num_subscopes(::ThisSubBlock{N}, ::ThisBlock) where {N} =
    cld(CUDA.blockDim().x, N)
@inline DataLayouts.subscope_rank(::ThisSubBlock{N}, ::ThisBlock) where {N} =
    cld(CUDA.threadIdx().x, N)

# Assign threads in a sub-block one slice of an array shared across their block.
@inline function DataLayouts.scoped_static_array(
    scope::ThisSubBlock,
    ::Type{T},
    dims,
) where {T}
    max_subblocks = MAX_WARPS_PER_BLOCK * DataLayouts.num_subscopes(scope, ThisWarp())
    array = DataLayouts.scoped_static_array(ThisBlock(), T, (dims..., max_subblocks))
    subblock_index = DataLayouts.subscope_rank(scope, ThisBlock())
    return @inbounds view(array, ntuple(Returns(:), Val(length(dims)))..., subblock_index)
end

# The last sub-block in a block may be only partially filled, so its active
# thread count is computed from the block's total. This is not the same as using
# CUDA.active_mask, which may be inconsistent across the lanes of a warp (see
# https://stackoverflow.com/questions/54055195).
@inline num_active_threads(scope) = DataLayouts.num_threads(scope)
@inline function num_active_threads(scope::ThisSubBlock)
    max_active_threads = DataLayouts.num_threads(scope)
    block_offset = (DataLayouts.subscope_rank(scope, ThisBlock()) - 1) * max_active_threads
    return clamp(DataLayouts.num_threads(ThisBlock()) - block_offset, 0, max_active_threads)
end

Adapt.@adapt_structure DataLayouts.StridedCartesianIndices

# Point loops must not run under @simd, whose loop restructuring makes kernels
# measurably slower: with bounds checks forced, it inflates the launch latency
# by ~20% (the per-point index conversion is repeated multiple times per loop).
DataLayouts.simd_over_indices(::DataLayouts.StridedCartesianIndices) = false
DataLayouts.simd_over_indices(
    ::SubArray{
        <:Any,
        1,
        <:Union{DataLayouts.ActiveColumnIndices, DataLayouts.ActivePointIndices},
    },
) = false
DataLayouts.simd_over_indices(
    ::Union{DataLayouts.ActiveColumnIndices, DataLayouts.ActivePointIndices},
) = false

Base.@propagate_inbounds @inline function DataLayouts.subscope_indices(
    ::ThisBlock,
    ::ThisKernel,
    indices,
)
    rank = CUDA.blockIdx().x
    n = CUDA.gridDim().x
    view_range = rank:n:length(indices)
    return DataLayouts.subscope_index_view(ThisKernel(), indices, view_range)
end

@inline function DataLayouts.subscope_index_view(
    ::Union{ThisKernel, ThisCooperativeGroup},
    indices::CartesianIndices{N},
    view_range,
) where {N}
    return DataLayouts.StridedCartesianIndices(indices, view_range)
end

@inline function DataLayouts.subscope_index_view(
    ::Union{ThisKernel, ThisCooperativeGroup},
    indices::DataLayouts.StridedCartesianIndices{I, V},
    view_range,
) where {I, V}
    new_range = indices.view_range[view_range]
    return DataLayouts.StridedCartesianIndices(indices.indices, new_range)
end

@inline function DataLayouts.subscope_index_view(
    ::Union{ThisKernel, ThisCooperativeGroup},
    indices::DataLayouts.ActiveColumnIndices,
    view_range,
)
    return DataLayouts.ActiveColumnIndices(indices.mask, indices.indices[view_range])
end

@inline function DataLayouts.subscope_index_view(
    ::Union{ThisKernel, ThisCooperativeGroup},
    indices::DataLayouts.ActivePointIndices{Nv},
    view_range,
) where {Nv}
    return DataLayouts.ActivePointIndices{Nv}(indices.mask, indices.indices[view_range])
end

# A unit range indexed at the positions in view_range is the same range of
# positions shifted by its offset, which is zero for the Base.OneTo ranges
# returned by eachindex.
@inline function DataLayouts.subscope_index_view(
    ::Union{ThisKernel, ThisCooperativeGroup},
    indices::AbstractUnitRange,
    view_range,
)
    @boundscheck checkbounds(indices, view_range)
    return @inbounds indices[view_range]
end

# Unmasked point loops on device scopes iterate eachindex instead of the
# Cartesian each_slice_index: when every argument supports linear indexing,
# this avoids the integer divisions that decompose a thread's linear index into
# a CartesianIndex at every point. Host scopes keep Cartesian indices; see
# each_maskable_slice_index in DataLayouts for why.
# The signature has to stay restricted to device scopes. The return type below
# is a union of a linear range and a CartesianIndices that only resolves where
# the extents are statically known, so covering host scopes as well would make
# merely loading this extension turn every CPU point loop over a layout with a
# dynamic extent into a dynamically dispatched one.
# Layouts are collected recursively, so that a singleton layout nested inside a
# broadcast expression is seen by the size check below; a nested broadcast's own
# combined size would hide it, and a linear index does not project onto
# singleton dimensions.
# The layouts are collected into a tuple with unrolled_flatmap, which keeps
# every intermediate type concrete. Folding the checks into a reduction over a
# reference layout would instead accumulate a union over the distinct layout
# types in the broadcast expression, which exceeds inference's union-splitting
# limits for heterogeneous expressions and makes the size and linearity checks
# dynamic during GPU compilation.
@inline get_all_non_point_layouts(arg::DataLayouts.DataLayout) =
    DataLayouts.is_non_point_arg(arg) ? (arg,) : ()
@inline get_all_non_point_layouts(
    bc::DataLayouts.MaybeFusedDataLayoutBroadcast,
) = unrolled_flatmap(get_all_non_point_layouts, DataLayouts.layout_args(bc))
@inline get_all_non_point_layouts(other) = ()

@inline is_device_linear(arg) = Base.IndexStyle(arg) == IndexLinear()

@inline function DataLayouts.each_maskable_slice_index(
    ::Union{ThisKernel, ThisCooperativeGroup},
    ::NoMask,
    ::typeof(view),
    args...,
)
    # 0-dimensional layouts (e.g. DataF args of fused broadcasts) broadcast to
    # every point, so they do not participate in the size or linearity checks.
    layouts = unrolled_flatmap(get_all_non_point_layouts, args)
    isempty(layouts) && return Base.OneTo(1)
    if unrolled_allequal(size, layouts) &&
       unrolled_all(is_device_linear, layouts)
        return Base.OneTo(length(first(layouts)))
    else
        # Singleton and mismatched extents require Cartesian indices, which
        # Broadcast.newindex projects onto each argument's dimensions; genuine
        # mismatches were already rejected on the host by check_slice_extents.
        return DataLayouts.each_slice_index(view, args...)
    end
end
