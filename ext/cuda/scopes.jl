const THREADS_PER_WARP = 32
const MAX_WARPS_PER_BLOCK = 32
const MAX_SHMEM_PER_BLOCK = 48 * 1024

# Smallest sub-block that a slice loop can assign a slice to, and therefore the
# last stop before ThisThread in the chain of scopes that slice_subscope
# descends through; see partition(::ThisSubBlock) below.
const MIN_THREADS_PER_SUBBLOCK = 2

# Cap on the number of threads in a block that assigns one slice to each of
# its sub-blocks. A sub-block's shared memory is allocated for the largest
# number of sub-blocks a block can hold (see scoped_static_array below), and
# that allocation is compiled into the kernel before its launch configuration
# is known, so the two have to agree on a bound. Without a cap the bound is a
# full 1024-thread block, which makes every slab loop reserve eight times the
# shared memory it can use -- and shared memory is what limits the occupancy
# of spectral element kernels. A cap of 128 reserves half the shared memory
# per block of a 256-thread bound, which raised occupancy and measured faster
# on an A100 baroclinic wave (h_elem = 30, z_elem = 63). Slabs with more
# points than the cap fall back to multiple points per thread (see
# DataLayouts.slice_subscope).
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
each slice (see [`DataLayouts.with_shared_memory`](@ref)), `:medium` by default.
"""
struct ThisHost{L} <: DataLayouts.DataScope end
ThisHost() = ThisHost{:medium}()

# Bytes of dynamic shared memory per thread at each level of DataLayouts.foreach_slice.
# At :none, every slice is evaluated by one thread, with private buffers.
shmem_level_bytes(::Val{:none}) = 0
shmem_level_bytes(::Val{:low}) = shmem_per_thread() ÷ 4
shmem_level_bytes(::Val{:medium}) = shmem_per_thread() ÷ 2
shmem_level_bytes(::Val{:max}) = shmem_per_thread()
shmem_level_bytes(::Val{bytes}) where {bytes} =
    bytes isa Integer && bytes >= 0 ? Int(bytes) :
    throw(
        ArgumentError(
            "Shared memory level must be Val(:none), Val(:low), Val(:medium), Val(:max), \
             or Val(n) for n bytes per thread, got Val($(repr(bytes)))",
        ),
    )
function DataLayouts.with_shared_memory(::ThisHost, level::Val{L}) where {L}
    shmem_level_bytes(level) # check the level
    return ThisHost{L}()
end

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

# Equal sizes invariant: two static shared-memory allocations are given the
# same memory only when they ask for the same number of bytes. Separately
# compiled allocations share whenever their globals have the same name, at the
# size of whichever module merged first; CUDA.CuStaticSharedArray names every
# global "shmem", so unequal allocations can share and write out of bounds.
# Putting the byte size in the name (this is otherwise CuStaticSharedArray's
# llvmcall) lets only EQUAL allocations share, so the buffer reuse invariant of
# Operators.apply_operator has to hold for every equally sized pair.
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

# Base pointer of a kernel's dynamic shared memory.
@generated function dynamic_shmem_base(::Type{T}) where {T}
    LLVM.@dispose ctx = LLVM.Context() begin
        pointer_type = convert(LLVM.LLVMType, Core.LLVMPtr{T, CUDA.AS.Shared})
        llvm_f, _ = LLVM.Interop.create_function(pointer_type)
        array_type = LLVM.ArrayType(LLVM.Int8Type(), 0)
        global_var = LLVM.GlobalVariable(
            LLVM.parent(llvm_f), array_type, "climacore_dynamic_shmem", CUDA.AS.Shared)
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

@inline static_shared_array(::ThisBlock, ::Type{T}, dims) where {T} =
    CUDA.CuDeviceArray{T, length(dims), CUDA.AS.Shared}(
        shmem_pointer(T, Val(prod(dims) * sizeof(T))), (dims...,))

"""
    ThisSubBlock{N}()

[`DataScope`](@ref) that represents `N` threads in [`ThisBlock`](@ref), where
`N` is typically a power of two.
"""
struct ThisSubBlock{N, B} <: ThisCooperativeGroup end
# B is the number of bytes of dynamic shared memory that each thread of the sub-block can
# use for its buffers (see DataLayouts.total_buffer_limit), set by the level of the slice
# loop that the sub-block evaluates (see ThisHost), and :medium otherwise.
ThisSubBlock{N}() where {N} = ThisSubBlock{N, shmem_level_bytes(Val(:medium))}()
# Sub-blocks of the same threads are the same subscope, whatever their shared memory (e.g.,
# for is_subscope, which finds a slice loop's sub-block among the partitions of a kernel).
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

# Largest block that a slice loop assigning every slice to this subscope may be
# launched with, before DataLayouts.subscope_launch_threads rounds it down to a
# whole number of sub-blocks.
#
# ONE WIDE SUB-BLOCK PER BLOCK. A sub-block of at most a warp has a barrier of
# its own (sync_warp), independent of every other warp in the block, so a
# block may hold as many such sub-blocks as MAX_SUBBLOCK_LAUNCH_THREADS
# allows. A wider sub-block has no barrier of its own and is synchronized with
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
# ThisBlock()) always one, so the shared memory below holds a single sub-block's
# worth of values instead of MAX_SUBBLOCK_LAUNCH_THREADS ÷ N of them.
#
# With dynamic shared memory, each sub-block's buffers are sized by the number of
# sub-blocks that a block is launched with (see dynamic_shared_array), so narrow sub-blocks
# need no cap other than the largest block.
@inline max_subblock_launch_threads(::ThisSubBlock{N}) where {N} =
    N > THREADS_PER_WARP ? N :
    use_dynamic_shmem() ? MAX_WARPS_PER_BLOCK * THREADS_PER_WARP :
    MAX_SUBBLOCK_LAUNCH_THREADS

# Assign threads in a sub-block one slice of an array shared across their
# block. The sub-block count comes from max_subblock_launch_threads, the cap
# on a sub-block slice loop's block size, rather than from
# num_subscopes(scope, ThisWarp()): a warp is a subscope of a ThisSubBlock{N}
# with N > THREADS_PER_WARP and not the other way around, so that call throws
# an InvalidSubscopeError for wide sub-blocks.
@inline function static_shared_array(scope::ThisSubBlock{N}, ::Type{T}, dims) where {N, T}
    max_subblocks = cld(max_subblock_launch_threads(scope), N)
    array = static_shared_array(ThisBlock(), T, (dims..., max_subblocks))
    subblock_index = DataLayouts.subscope_rank(scope, ThisBlock())
    return @inbounds view(array, ntuple(Returns(:), Val(length(dims)))..., subblock_index)
end

# Whether the shared arrays of blocks and sub-blocks are allocated in dynamic shared
# memory (see dynamic_shared_array) instead of static globals (see static_shared_array).
# A compile-time constant, set by CLIMA_CUDA_DYNAMIC_SHMEM before the extension
# precompiles, or by redefining this method.
use_dynamic_shmem() = USE_DYNAMIC_SHMEM
const USE_DYNAMIC_SHMEM = _getenv_bool("CLIMA_CUDA_DYNAMIC_SHMEM"; default = false)

# Bytes of dynamic shared memory that each thread of a sub-block can use for its buffers
# (see DataLayouts.total_buffer_limit), at most the shared memory of a multiprocessor
# divided among the most threads it can run, so that shared memory does not limit the
# occupancy of a kernel unless registers already do. The default is the most that an A100
# allows with blocks of at least 128 threads (167936 bytes per multiprocessor, 1024 of them
# reserved for each of 16 blocks, among 2048 threads); a compile-time constant, set by
# CLIMA_CUDA_SHMEM_PER_THREAD before the extension precompiles. An expression whose buffers
# do not fit fails to compile (see check_shmem_level); the :max level of foreach_slice.
const SHMEM_PER_THREAD = _getenv_int("CLIMA_CUDA_SHMEM_PER_THREAD", 74)
shmem_per_thread() = SHMEM_PER_THREAD # can be redefined, like use_dynamic_shmem

# Allocations are aligned to 32 bytes, like the globals of shmem_pointer.
aligned_shmem_bytes(bytes) = cld(bytes, 32) * 32

# Dynamic shared memory is divided into one region per sub-block (or a single region for
# a block), and an allocation of `bytes` bytes per region at byte `offset` of its region
# is stored for all regions at once, at byte `offset` times the number of regions. Device
# code then only needs the number of regions in its block, while a launch gives every
# region the bytes of its sub-block's level (see ThisHost and _foreach_slice). Allocations
# that are live at the same time are given different offsets (see the buffer reuse
# invariant in Operators.apply_operator), and every allocation synchronizes its scope,
# so that no thread is still reading an earlier allocation that it overlaps.
@inline function dynamic_shared_array(
    scope::Union{ThisBlock, ThisSubBlock},
    ::Type{T},
    dims,
    ::Val{offset},
) where {T, offset}
    bytes = aligned_shmem_bytes(prod(dims) * sizeof(T))
    check_shmem_level(scope, Val(aligned_shmem_bytes(offset) + bytes))
    base = dynamic_shmem_base(UInt8)
    region = region_rank(scope)
    first_byte =
        UInt32(aligned_shmem_bytes(offset)) * num_regions(scope) +
        (region - one(region)) * UInt32(bytes)
    pointer = reinterpret(Core.LLVMPtr{T, CUDA.AS.Shared}, base + first_byte)
    DataLayouts.synchronize(scope)
    return CUDA.CuDeviceArray{T, length(dims), CUDA.AS.Shared}(pointer, (dims...,))
end

# An allocation that does not fit in the region of its sub-block fails to compile, with an
# error that names the bytes it needs (see _foreach_slice, which reports it). Arguments that
# would only be cached are recomputed instead (see Operators.caches_values), so this is
# reached by buffers that cannot be recomputed, like those of spectral element operators
# and of register temporaries read at neighboring points.
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

@inline DataLayouts.scoped_static_array(
    scope::Union{ThisBlock, ThisSubBlock},
    ::Type{T},
    dims,
    offset::Val = Val(0),
) where {T} =
    use_dynamic_shmem() ? dynamic_shared_array(scope, T, dims, offset) :
    static_shared_array(scope, T, dims)

DataLayouts.total_buffer_limit(::Union{ThisHost, ThisKernel}) = typemax(Int)
DataLayouts.total_buffer_limit(::ThisBlock) = MAX_SHMEM_PER_BLOCK
DataLayouts.total_buffer_limit(scope::ThisSubBlock{N, B}) where {N, B} =
    use_dynamic_shmem() ? N * B :
    fld(MAX_SHMEM_PER_BLOCK, cld(max_subblock_launch_threads(scope), N))
DataLayouts.aliases_buffers(::Union{ThisBlock, ThisSubBlock}) = use_dynamic_shmem()

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

# The bytes of a layout in an array from dynamic_shared_array or static_shared_array.
@generated function DataLayouts.shared_buffer_bytes(data::DataLayouts.DataLayout)
    A = DataLayouts.parent_type(data)
    shared_array_type = Union{
        CUDA.CuDeviceArray{<:Any, <:Any, CUDA.AS.Shared},
        SubArray{<:Any, <:Any, <:CUDA.CuDeviceArray{<:Any, <:Any, CUDA.AS.Shared}},
    }
    (A <: shared_array_type && DataLayouts.has_inferred_size(data)) || return :(Val(0))
    B = eltype(A)
    Nf = DataLayouts.num_basetypes(B, eltype(data))
    return :(Val($(sizeof(B) * Nf * prod(DataLayouts.inferred_size(data)))))
end
