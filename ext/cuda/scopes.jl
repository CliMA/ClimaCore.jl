import UnrolledUtilities: unrolled_map, unrolled_all, unrolled_allequal, unrolled_flatmap
import CUDA: LLVM # Used by shmem_pointer to emit a shared-memory global.

const THREADS_PER_WARP = 32
const MAX_WARPS_PER_BLOCK = 32

# Smallest sub-block that a slice loop can assign a slice to, and therefore the
# last stop before ThisThread in the chain of scopes that slice_subscope
# descends through; see partition(::ThisSubBlock) below.
const MIN_THREADS_PER_SUBBLOCK = 2

# Cap on the number of threads in a block that assigns one slice to each of its
# sub-blocks. A sub-block's shared memory is allocated for the largest number of
# sub-blocks a block can hold (see scoped_static_array below), and that
# allocation is compiled into the kernel before its launch configuration is
# known, so the two have to agree on a bound. Without a cap the bound is a full
# 1024-thread block, which makes every slab loop reserve four times the shared
# memory it can use -- and shared memory is what limits the occupancy of spectral
# element kernels. The kernels that slice loops replaced used 256; halving the
# cap to 128 halves the shared memory reserved per block, which raised occupancy
# and measured faster on an A100 baroclinic wave (h_elem = 30, z_elem = 63).
# Slabs with more points than the cap fall back to multiple points per thread
# (see DataLayouts.slice_subscope).
const MAX_SUBBLOCK_LAUNCH_THREADS = 128

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

# A kernel is partitioned into sub-blocks of MAX_SUBBLOCK_LAUNCH_THREADS threads
# rather than into blocks, so that every scope in the chain that slice_subscope
# descends through has a compile-time thread count (see static_num_threads
# below): DataLayouts.register_similar needs that count to be constant to size a
# thread's register storage, and a dynamically-sized ThisBlock at the top of the
# chain would make it a run-time value for every slice with more than
# THREADS_PER_WARP points. ThisBlock stays a subscope of ThisKernel even though
# it left the partition chain, since it is the scope of every shared-memory
# allocation (see DataScope(::Type{<:CUDA.CuDeviceArray}) above).
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

# A block is partitioned into warps rather than into individual threads, so that
# slice_subscope can descend past ThisBlock and give a slice with no more points
# than a warp has lanes a group of exactly that many threads. Partitioning
# straight into threads makes num_threads(partition(ThisBlock())) equal to 1,
# which stops the descent at ThisBlock for every slice with more than one point:
# a 4x4 GLL slab then occupies a whole 16-thread block, wasting half of the
# block's only warp and capping occupancy at the hardware limit of 32 blocks per
# multiprocessor (512 of 2048 threads), instead of packing 16 slabs into one
# 256-thread block. Beyond a warp there is nothing to gain.
#
# The chain is only this long because every num_slice_points method is a function
# of an argument's type. When a slice's point count was a run-time value, every
# scope the descent could stop at ended up in the Union that slice_subscope
# returned, and inference widened a Union of more than three scopes to Any, which
# made every slice loop dispatch dynamically and allocate inside GPU kernels.
@inline DataLayouts.partition(::ThisBlock) = ThisWarp()
@inline DataLayouts.num_threads(::ThisBlock) = CUDA.blockDim().x
@inline DataLayouts.thread_rank(::ThisBlock) = CUDA.threadIdx().x
@inline DataLayouts.synchronize(::ThisBlock) = CUDA.sync_threads()

# Equal sizes invariant: two static shared-memory allocations are given the
# same memory only when they ask for the same number of bytes. Separately
# compiled allocations share whenever their globals have the same name, at the
# size of whichever module merged first; CUDA.CuStaticSharedArray names every
# global "shmem", so unequal allocations can share and write out of bounds.
# Putting the byte size in the name (this is otherwise CuStaticSharedArray's
# llvmcall) lets only EQUAL allocations share, so the buffer reuse invariant in
# Operators/spectralelement.jl has to hold for every equally sized pair.
@generated function shmem_pointer(::Type{T}, ::Val{bytes}) where {T, bytes}
    LLVM.@dispose ctx = LLVM.Context() begin
        pointer_type = convert(LLVM.LLVMType, Core.LLVMPtr{T, CUDA.AS.Shared})
        llvm_f, _ = LLVM.Interop.create_function(pointer_type)
        array_type = LLVM.ArrayType(LLVM.Int8Type(), bytes)
        global_var = LLVM.GlobalVariable(
            LLVM.parent(llvm_f), array_type, "shmem_$(bytes)B", CUDA.AS.Shared)
        LLVM.linkage!(global_var, LLVM.API.LLVMInternalLinkage)
        LLVM.initializer!(global_var, LLVM.null(array_type))
        LLVM.alignment!(global_var, max(32, Base.datatype_alignment(T)))
        LLVM.@dispose builder = LLVM.IRBuilder() begin
            LLVM.position!(builder, LLVM.BasicBlock(llvm_f, "entry"))
            zeros = [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]
            first_byte = LLVM.gep!(builder, array_type, global_var, zeros)
            LLVM.ret!(builder, LLVM.bitcast!(builder, first_byte, pointer_type))
        end
        LLVM.Interop.call_function(llvm_f, Core.LLVMPtr{T, CUDA.AS.Shared})
    end
end
@inline DataLayouts.scoped_static_array(::ThisBlock, ::Type{T}, dims) where {T} =
    CUDA.CuDeviceArray{T, length(dims), CUDA.AS.Shared}(
        shmem_pointer(T, Val(prod(dims) * sizeof(T))), (dims...,))

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

# Sub-blocks are halved until they hold MIN_THREADS_PER_SUBBLOCK threads, which
# are partitioned into single threads. Halving keeps every sub-block aligned
# with the lanes of a warp, which thread_rank and synchronize below rely on.
@inline DataLayouts.partition(::ThisSubBlock{N}) where {N} =
    N <= MIN_THREADS_PER_SUBBLOCK ? DataLayouts.ThisThread() : ThisSubBlock{N ÷ 2}()
@inline DataLayouts.num_threads(::ThisSubBlock{N}) where {N} = N
@inline DataLayouts.static_num_threads(::ThisSubBlock{N}) where {N} = N
@inline DataLayouts.thread_rank(::ThisSubBlock{N}) where {N} =
    N > THREADS_PER_WARP ? (DataLayouts.thread_rank(ThisBlock()) - 1) % N + 1 :
    N < THREADS_PER_WARP ? (CUDA.laneid() - 1) % N + 1 : CUDA.laneid()
@inline DataLayouts.synchronize(::ThisSubBlock{N}) where {N} =
    N > THREADS_PER_WARP ? DataLayouts.synchronize(ThisBlock()) : CUDA.sync_warp()

@inline DataLayouts.partition(::ThisKernel) =
    ThisSubBlock{MAX_SUBBLOCK_LAUNCH_THREADS}()
@inline DataLayouts.is_subscope(::ThisBlock, ::ThisKernel) = true
@inline DataLayouts.num_subscopes(::ThisBlock, ::ThisKernel) = CUDA.gridDim().x
@inline DataLayouts.subscope_rank(::ThisBlock, ::ThisKernel) = CUDA.blockIdx().x

@inline DataLayouts.is_subscope(::ThisSubBlock, ::ThisBlock) = true
@inline DataLayouts.num_subscopes(::ThisSubBlock{N}, ::ThisBlock) where {N} =
    cld(CUDA.blockDim().x, N)
@inline DataLayouts.subscope_rank(::ThisSubBlock{N}, ::ThisBlock) where {N} =
    cld(CUDA.threadIdx().x, N)

# Largest block that a slice loop assigning every slice to this subscope may be
# launched with, before DataLayouts.subscope_launch_threads rounds it down to a
# whole number of sub-blocks.
#
# ONE WIDE SUB-BLOCK PER BLOCK. A sub-block of at most a warp has a barrier of
# its own (sync_warp), independent of every other warp in the block, so a block
# may hold as many such sub-blocks as MAX_SUBBLOCK_LAUNCH_THREADS allows. A wider
# sub-block has no barrier of its own and is synchronized with the block-wide
# sync_threads instead (see synchronize(::ThisSubBlock) above), so a block must
# hold exactly one of them: two sub-blocks in one block are given consecutive
# ranks by subscope_indices, which hands rank r the strided subset r:n:num_slices
# of the loop's slices, and those subsets differ in length by one whenever
# num_slices is not a multiple of n. The sub-block with the extra slice then
# reaches that round's barriers after its neighbours have already returned from
# the kernel, and sync_threads reached by only part of a block is undefined
# behavior. Giving each wide sub-block a block of its own makes the block-wide
# barrier exactly the sub-block's barrier, and makes subscope_rank(scope,
# ThisBlock()) always one, so the shared memory below holds a single sub-block's
# worth of values instead of MAX_SUBBLOCK_LAUNCH_THREADS ÷ N of them.
@inline max_subblock_launch_threads(::ThisSubBlock{N}) where {N} =
    N > THREADS_PER_WARP ? N : MAX_SUBBLOCK_LAUNCH_THREADS

# Assign threads in a sub-block one slice of an array shared across their block.
# The sub-block count comes from max_subblock_launch_threads, the cap on a
# sub-block slice loop's block size, rather than from num_subscopes(scope,
# ThisWarp()): a warp is a subscope of a ThisSubBlock{N} with N > THREADS_PER_WARP
# and not the other way around, so that call throws an InvalidSubscopeError for
# wide sub-blocks.
@inline function DataLayouts.scoped_static_array(
    scope::ThisSubBlock{N}, ::Type{T}, dims,
) where {N, T}
    max_subblocks = cld(max_subblock_launch_threads(scope), N)
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
    view_range = DataLayouts.strided_range(rank, n, length(indices))
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
    indices::DataLayouts.StridedCartesianIndices,
    view_range,
)
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

# Unmasked point loops on device scopes iterate over eachindex instead of the
# Cartesian each_slice_index. When every argument supports linear indexing, this
# avoids the integer divisions required for linear-to-Cartesian conversion.
@inline DataLayouts.each_maskable_slice_index(
    ::Union{ThisKernel, ThisCooperativeGroup},
    ::NoMask,
    ::typeof(view),
    args...,
) = eachindex(args...)
