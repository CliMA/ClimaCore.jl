const THREADS_PER_WARP = 32
const MAX_WARPS_PER_BLOCK = 32
const MAX_SHMEM_PER_BLOCK = 48 * 1024

# Smallest sub-block that a slice loop can assign a slice to, and therefore the
# last stop before ThisThread in the chain of scopes that slice_subscope
# descends through; see partition(::ThisSubBlock) below.
const MIN_THREADS_PER_SUBBLOCK = 2

# Number of threads in the largest sub-block that a slice loop can assign a slice to, at
# the top of the chain of scopes that slice_subscope descends through (see
# partition(::ThisKernel) below). Slices with more points than this are evaluated with
# multiple points per thread (see DataLayouts.slice_subscope).
const MAX_SUBBLOCK_LAUNCH_THREADS = 128

# To reduce latency, only check device attributes before the first launch.
const DEVICE_ASSUMPTIONS_CHECKED = Ref(false)
function check_device_assumptions()
    DEVICE_ASSUMPTIONS_CHECKED[] && return true
    (; threads_per_warp, max_threads_per_block, max_shmem_per_block) =
        device_attributes()
    if THREADS_PER_WARP != threads_per_warp ||
       MAX_WARPS_PER_BLOCK * THREADS_PER_WARP != max_threads_per_block ||
       MAX_SHMEM_PER_BLOCK != max_shmem_per_block
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

The type parameter is the level of shared memory that slice loops over the scope give
each slice (see `shmem` in [`DataLayouts.foreach_slice`](@ref)), `:auto` unless another
level is chosen (see [`DataLayouts.with_shared_memory`](@ref)).
"""
struct ThisHost{L} <: DataLayouts.DataScope end
ThisHost() = ThisHost{:auto}()

# Bytes of dynamic shared memory per thread at a level of DataLayouts.foreach_slice (:auto is
# chosen per loop; see slice_subscope_at_level), for slices of N threads. They are a type
# parameter of ThisSubBlock, so they are read from the device once per process by a generated
# function; no slice loop runs while the extension precompiles.
@generated shmem_level_bytes(::Val{L}, ::Val{N}) where {L, N} =
    L isa Integer ? Int(L) : L == :none ? 0 : device_shmem_level_bytes(L, N)
DataLayouts.with_shared_memory(::ThisHost, ::Val{L}) where {L} = ThisHost{L}()

DataLayouts.num_threads(::ThisHost) = throw(ArgumentError("Cannot get num_threads on host"))
DataLayouts.thread_rank(::ThisHost) = throw(ArgumentError("Cannot get thread_rank on host"))
DataLayouts.scoped_array(::ThisHost, ::Type{T}, dims; buffer = false) where {T} =
    buffer ? DataLayouts.task_reduction_buffer(CUDA.CuArray{T, 1}, dims) :
    CUDA.CuArray{T}(undef, dims)

"""
    ThisKernel()

[`DataScope`](@ref) that represents all available threads on a GPU. Operations
that require synchronizations or array allocations are not supported.

`num_threads` and `thread_rank` assume that kernels are launched with one-dimensional
grids.
"""
struct ThisKernel <: DataLayouts.DataScope end

# A kernel is partitioned into sub-blocks of MAX_SUBBLOCK_LAUNCH_THREADS threads
# rather than into blocks, so that every scope in the chain that slice_subscope
# descends through has a compile-time thread count (see static_num_threads
# below): DataLayouts.register_similar needs that count to be constant to size a
# thread's register storage, and a dynamically-sized ThisBlock at the top of the
# chain would make it a run-time value for every slice with more than
# THREADS_PER_WARP points. ThisBlock stays a subscope of ThisKernel even though
# it is not part of the static partitioning chain; it is the scope of every
# shared-memory allocation (see DataScope(::Type{<:CUDA.CuDeviceArray}) above).
@inline DataLayouts.num_threads(::ThisKernel) =
    DataLayouts.num_subscopes(ThisBlock(), ThisKernel()) *
    DataLayouts.num_threads(ThisBlock())
@inline function DataLayouts.thread_rank(::ThisKernel)
    block_rank = DataLayouts.subscope_rank(ThisBlock(), ThisKernel())
    return (block_rank - one(block_rank)) * DataLayouts.num_threads(ThisBlock()) +
           DataLayouts.thread_rank(ThisBlock())
end
@inline DataLayouts.strided_access(::ThisKernel) = true

"""
    ThisCooperativeGroup

Supertype for scopes that represent a "cooperative group" from the
[`CG`](https://cuda.juliagpu.org/stable/development/kernel/#Cooperative-groups)
module in `CUDA.jl`, which is built on the `cooperative_groups`
[extension](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cooperative-groups)
of CUDA.

Subtypes:

  - [`ThisBlock`](@ref): one thread block.
  - [`ThisSubBlock`](@ref): `N` threads of a block; [`ThisWarp`](@ref) is the `N = 32` case.
"""
abstract type ThisCooperativeGroup <: DataLayouts.DataScope end

"""
    ThisBlock()

[`DataScope`](@ref) that represents one thread block of [`ThisKernel`](@ref).
Operations that require dynamically-sized array allocations are not supported.

`num_threads` and `thread_rank` assume that kernels are launched with one-dimensional
blocks.
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
@inline DataLayouts.partition(::ThisBlock) = ThisWarp()
# As UInt32s, since a signed division by a power of two has to allow for negative values.
@inline DataLayouts.num_threads(::ThisBlock) = CUDA.blockDim().x % UInt32
@inline DataLayouts.thread_rank(::ThisBlock) = CUDA.threadIdx().x % UInt32
@inline DataLayouts.synchronize(::ThisBlock) = CUDA.sync_threads()

"""
    ThisSubBlock{N, B}()

[`DataScope`](@ref) that represents `N` threads in [`ThisBlock`](@ref), where
`N` is typically a power of two. Each thread can use `B` bytes of the block's dynamic
shared memory for the buffers of its sub-block (see `scoped_static_array`), which is set
by the level of the slice loop that assigns a slice to the sub-block (see [`ThisHost`](@ref))
and is the `:medium` level by default. Sub-blocks of the same threads are the same subscope,
whatever their shared memory (e.g., for `is_subscope`, which finds a slice loop's sub-block
among the partitions of a kernel).
"""
struct ThisSubBlock{N, B} <: ThisCooperativeGroup end
ThisSubBlock{N}() where {N} = ThisSubBlock{N, shmem_level_bytes(Val(:medium), Val(N))}()
Base.:(==)(::ThisSubBlock{N}, ::ThisSubBlock{N}) where {N} = true

"""
    ThisWarp()

Special case of [`ThisSubBlock`](@ref) that represents an entire warp.
Operations that require dynamically-sized array allocations are not supported.
"""
const ThisWarp = ThisSubBlock{THREADS_PER_WARP}

# Sub-blocks are halved until they hold MIN_THREADS_PER_SUBBLOCK threads, which
# are partitioned into single threads. Halving keeps every sub-block aligned
# with the lanes of a warp, which thread_rank and synchronize below rely on.
@inline DataLayouts.partition(::ThisSubBlock{N, B}) where {N, B} =
    N <= MIN_THREADS_PER_SUBBLOCK ? DataLayouts.ThisThread() : ThisSubBlock{N ÷ 2, B}()
@inline DataLayouts.num_threads(::ThisSubBlock{N}) where {N} = N % UInt32
@inline DataLayouts.static_num_threads(::ThisSubBlock{N}) where {N} = N
# A sub-block holds consecutive ranks of its block (or warp), so its ranks are remainders.
@inline function DataLayouts.thread_rank(::ThisSubBlock{N}) where {N}
    N == THREADS_PER_WARP && return CUDA.laneid() % UInt32
    rank = DataLayouts.thread_rank(N > THREADS_PER_WARP ? ThisBlock() : ThisWarp())
    return rem(rank - one(rank), N % UInt32) + one(rank)
end
@inline DataLayouts.synchronize(::ThisSubBlock{N}) where {N} =
    N > THREADS_PER_WARP ? DataLayouts.synchronize(ThisBlock()) : CUDA.sync_warp()

@inline DataLayouts.partition(::ThisKernel) =
    ThisSubBlock{MAX_SUBBLOCK_LAUNCH_THREADS}()
@inline DataLayouts.is_subscope(::ThisBlock, ::ThisKernel) = true
@inline DataLayouts.num_subscopes(::ThisBlock, ::ThisKernel) = CUDA.gridDim().x % UInt32
@inline DataLayouts.subscope_rank(::ThisBlock, ::ThisKernel) = CUDA.blockIdx().x % UInt32

@inline DataLayouts.is_subscope(::ThisSubBlock, ::ThisBlock) = true

# Largest block that a slice loop assigning every slice to this subscope may be launched
# with; the launch keeps every block to whole sub-blocks (see the granularity of
# launch_configuration in _foreach_slice).
#
# ONE WIDE SUB-BLOCK PER BLOCK. A sub-block of at most a warp has a barrier of
# its own (sync_warp), independent of every other warp in the block, so a
# block may hold as many such sub-blocks as its threads allow (the shared memory of
# every sub-block is sized by the number of sub-blocks a block is launched with; see
# scoped_static_array). A wider sub-block has no barrier of its own and is synchronized with
# the block-wide sync_threads instead (see synchronize(::ThisSubBlock) above),
# so a block must hold exactly one of them: two sub-blocks in one block are
# given consecutive ranks by subscope_indices, which hands rank r the strided
# subset r:n:num_slices of the loop's slices, and those subsets differ in
# length by one whenever num_slices is not a multiple of n. The sub-block with
# the extra slice then
# reaches that round's barriers after its neighbours have already returned from
# the kernel, and sync_threads reached by only part of a block is undefined
# behavior. Giving each wide sub-block a block of its own makes the block-wide
# barrier exactly the sub-block's barrier, and makes subscope_rank(scope,
# ThisBlock()) always one.
@inline max_subblock_launch_threads(::ThisSubBlock{N}) where {N} =
    N > THREADS_PER_WARP ? N : MAX_WARPS_PER_BLOCK * THREADS_PER_WARP

# The levels of shared memory per thread for sub-blocks of N threads on the device (see
# shmem_level_bytes): :medium keeps shared memory from limiting occupancy (see
# full_occupancy_shmem_per_thread); :max is the most that a kernel can be launched with per
# block after opting in (see auto_launch!), divided among the threads of a block,
# which holds at least a warp of narrow sub-blocks that share its memory.
device_shmem_level_bytes(level, N) =
    level == :medium ? full_occupancy_shmem_per_thread(N) :
    level == :max ?
    fld(device_attributes().max_shmem_per_block_optin, max(N, THREADS_PER_WARP)) :
    throw(ArgumentError(DataLayouts.invalid_shmem_level_string(Val(level))))

# Bytes of shared memory per thread that keep shared memory from limiting the occupancy of a
# kernel whose slices are sub-blocks of N threads, unless registers already do: the memory
# of a multiprocessor, divided among the blocks it runs at full occupancy (each holding one
# wide sub-block, or 1024 threads of narrow ones; see max_subblock_launch_threads), less what
# the driver reserves for each block. The :medium level; 66 bytes for 64 threads on an A100,
# whose multiprocessors run 2048 threads in up to 32 blocks with 164 KB of shared memory.
function full_occupancy_shmem_per_thread(N)
    (; max_shmem_per_sm, max_threads_per_sm, max_blocks_per_sm, reserved_shmem_per_block) =
        device_attributes()
    block_threads = N > THREADS_PER_WARP ? N : MAX_WARPS_PER_BLOCK * THREADS_PER_WARP
    blocks = min(max_blocks_per_sm, max_threads_per_sm ÷ block_threads)
    return fld(fld(max_shmem_per_sm, blocks) - reserved_shmem_per_block, block_threads)
end

# One region of dynamic shared memory per sub-block (or one for a block), with the allocations
# at an offset stored for all regions at once. Co-live allocations have different offsets (see
# the buffer reuse invariant in Operators.apply_operator), and every allocation synchronizes
# its scope, which retires the allocations it overlaps. The bounds check against the launched
# bytes holds when the level check passes, so it is skipped.
@inline function DataLayouts.scoped_static_array(
    scope::Union{ThisBlock, ThisSubBlock},
    ::Type{T},
    dims,
    ::Val{offset} = Val(0),
) where {T, offset}
    bytes = DataLayouts.aligned_buffer_bytes(prod(dims) * sizeof(T))
    check_shmem_level(scope, Val(DataLayouts.aligned_buffer_bytes(offset) + bytes))
    region = region_rank(scope)
    first_byte =
        UInt32(DataLayouts.aligned_buffer_bytes(offset)) * num_regions(scope) +
        (region - one(region)) * UInt32(bytes)
    DataLayouts.synchronize(scope)
    return @inbounds CUDA.CuDynamicSharedArray(T, (dims...,), first_byte)
end

# An allocation that does not fit in the region of its sub-block fails to compile, with an
# error that names the bytes it needs (see launch_at_level, which reports it). Values that
# can be recomputed are never allocated beyond the level (see Operators.caches_values), so
# this is reached by buffers that cannot be, like those of spectral element operators and
# of register temporaries read at neighboring points. Block allocations are sized by their
# launches instead (see DynamicShmem).
@inline check_shmem_level(::ThisBlock, _) = nothing
@generated function check_shmem_level(
    ::ThisSubBlock{N, B},
    ::Val{bytes},
) where {N, B, bytes}
    bytes <= N * B && return :(nothing)
    name = Symbol("$(SHMEM_LEVEL_ERROR)_$(bytes)_bytes_for_$(N)_threads_at_$(B)_per_thread")
    return :(ccall($(QuoteNode(name)), Cvoid, ()))
end
const SHMEM_LEVEL_ERROR = "climacore_shared_memory_exceeds_level"
@inline num_regions(::ThisBlock) = UInt32(1)
@inline num_regions(scope::ThisSubBlock) =
    DataLayouts.num_threads(ThisBlock()) ÷ DataLayouts.num_threads(scope)
@inline region_rank(::ThisBlock) = UInt32(1)
@inline region_rank(scope::ThisSubBlock) = DataLayouts.subscope_rank(scope, ThisBlock())

DataLayouts.total_buffer_limit(::Union{ThisHost, ThisKernel}) = typemax(Int)
DataLayouts.total_buffer_limit(::ThisBlock) = MAX_SHMEM_PER_BLOCK
DataLayouts.total_buffer_limit(::ThisSubBlock{N, B}) where {N, B} = N * B

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
# @simd would count positions in the strided subsets of device threads, dividing out their
# length (see DataLayouts.index_positions); host StridedRange{Int}s keep it to vectorize.
DataLayouts.simd_over_indices(::DataLayouts.StridedRange{UInt32}) = false
DataLayouts.simd_over_indices(::SubArray{<:Any, 1, <:DataLayouts.ActiveIndices}) = false
DataLayouts.simd_over_indices(::DataLayouts.ActiveIndices) = false

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

# The aligned bytes of a layout in an array from scoped_static_array, computed from its type
# alone (the parent array's type, its base type, and the inferred size), so that the Val folds.
@inline function DataLayouts.shared_buffer_bytes(data::DataLayouts.DataLayout)
    A = DataLayouts.parent_type(data)
    is_shared = A <: CUDA.CuDeviceArray{<:Any, <:Any, CUDA.AS.Shared}
    (is_shared && DataLayouts.has_inferred_size(data)) || return Val(0)
    Nf = DataLayouts.num_basetypes(eltype(A), eltype(data))
    bytes = sizeof(eltype(A)) * Nf * unrolled_prod(DataLayouts.inferred_size(data))
    return Val(DataLayouts.aligned_buffer_bytes(bytes))
end
