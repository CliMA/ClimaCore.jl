const THREADS_PER_WARP = 32
const MAX_WARPS_PER_BLOCK = 32

function check_device_assumptions()
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
end

@inline x_component((; x, y, z)) =
    isone(y) && isone(z) ? x :
    throw(ArgumentError("y and z dimensions in launch configuration are not supported"))

DataLayouts.DataScope(::Type{<:CUDA.CuArray}) = ThisHost()
DataLayouts.DataScope(::Type{<:CUDA.CuDeviceArray{<:Any, <:Any, A}}) where {A} =
    A == CUDA.AS.Local ? ThisThread() : A == CUDA.AS.Shared ? ThisBlock() : ThisKernel()

"""
    ThisHost()

[`DataScope`](@ref) that represents the host device for a GPU. This scope can be
assigned to any [`DataLayout`](@ref) backed by a `CuArray`, and it is replaced
with its device-side analogue [`ThisKernel`](@ref) by `Adapt.jl`. Aside from
array allocations, other standard `DataScope` operations are not supported.
"""
struct ThisHost <: DataLayouts.DataScope end

Adapt.adapt_structure(::CUDA.KernelAdaptor, ::ThisHost) = ThisKernel()
DataLayouts.num_threads(::ThisHost) = throw(ArgumentError("Cannot get num_threads on host"))
DataLayouts.thread_rank(::ThisHost) = throw(ArgumentError("Cannot get thread_rank on host"))
DataLayouts.scoped_array(::ThisHost, ::Type{T}, dims) where {T} =
    CUDA.CuArray{T}(undef, dims)

"""
    ThisKernel()

[`DataScope`](@ref) that represents all available threads on a GPU. This scope
can only be assigned to a [`DataLayout`](@ref) through `Adapt.jl`. Operations
that require synchronizations or array allocations are not supported.
"""
struct ThisKernel <: DataLayouts.DataScope end

DataLayouts.partition(::ThisKernel) = ThisBlock()
DataLayouts.num_partitions(::ThisKernel) = x_component(CUDA.gridDim())
DataLayouts.partition_rank(::ThisKernel) = x_component(CUDA.blockIdx())

"""
    ThisBlock()

[`DataScope`](@ref) that represents one thread block of [`ThisKernel`](@ref).
"""
struct ThisBlock <: DataLayouts.DataScope end

DataLayouts.partition(::ThisBlock) = ThisWarp()
DataLayouts.num_threads(::ThisBlock) = x_component(CUDA.blockDim())
DataLayouts.thread_rank(::ThisBlock) = x_component(CUDA.threadIdx())
DataLayouts.synchronize(::ThisBlock) = CUDA.sync_threads()
DataLayouts.scoped_array(::ThisBlock, ::Type{T}, dims) where {T} =
    CUDA.CuDynamicSharedArray(T, dims)
DataLayouts.scoped_static_array(::ThisBlock, ::Type{T}, dims) where {T} =
    CUDA.CuStaticSharedArray(T, dims)

"""
    ThisSubBlock{N}()

[`DataScope`](@ref) that represents `N` threads in [`ThisBlock`](@ref), where
`N` is typically a power of two.
"""
struct ThisSubBlock{N} <: DataLayouts.DataScope end

"""
    ThisWarp()

Special case of [`ThisSubBlock`](@ref) that represents an entire warp.
"""
const ThisWarp = ThisSubBlock{THREADS_PER_WARP}

DataLayouts.partition(::ThisSubBlock{N}) where {N} =
    N < 4 ? DataLayouts.ThisThread() : ThisSubBlock{N ÷ 2}()
DataLayouts.num_threads(::ThisSubBlock{N}) where {N} = N
DataLayouts.thread_rank(::ThisSubBlock{N}) where {N} =
    N > THREADS_PER_WARP ? (x_component(CUDA.threadIdx()) - 1) % N + 1 :
    N < THREADS_PER_WARP ? (CUDA.laneid() - 1) % N + 1 : CUDA.laneid()
DataLayouts.synchronize(::ThisSubBlock) =
    N > THREADS_PER_WARP ? CUDA.sync_threads() : CUDA.sync_warp()

# Assign threads in a sub-block one slice of an array shared across their block.
function DataLayouts.scoped_array(scope::ThisSubBlock, ::Type{T}, dims) where {T}
    num_subblocks = DataLayouts.num_subscopes(scope, ThisBlock())
    array = DataLayouts.scoped_array(ThisBlock(), T, (dims..., num_subblocks))
    subblock_id = DataLayouts.subscope_rank(scope, ThisBlock())
    return @inbounds view(array, ntuple(Returns(:), Val(length(dims)))..., subblock_id)
end
function DataLayouts.scoped_static_array(scope::ThisSubBlock, ::Type{T}, dims) where {T}
    max_subblocks = MAX_WARPS_PER_BLOCK * DataLayouts.num_subscopes(scope, ThisWarp())
    array = DataLayouts.scoped_static_array(ThisBlock(), T, (dims..., max_subblocks))
    subblock_id = DataLayouts.subscope_rank(scope, ThisBlock())
    return @inbounds view(array, ntuple(Returns(:), Val(length(dims)))..., subblock_id)
end
