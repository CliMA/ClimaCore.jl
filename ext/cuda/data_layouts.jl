import Adapt
import CUDA
import LLVM
import ClimaComms
import ClimaCore: DataLayouts
using UnrolledUtilities

# Kernels use UInt32 indices, but leave half the available index range unused.
# Threads of a loop step through their indices by adding the thread count, which
# can be as large as the index count, so each must fit in half the UInt32 range.
# Also, 32-bit array offsets can be sign-extended (see LLVM.jl's pointerref), so
# they must fit in half the UInt32 range to remain unchanged.
check_index_count(n) =
    2 * n <= typemax(UInt32) ||
    throw(ArgumentError("GPU kernels support at most 2^31 - 1 array elements or slices"))

include("scopes.jl")
include("loops.jl")

# Kernel parameters are limited to 4 KiB of memory before compute capability 7.0, and a
# 5-D CuDeviceArray uses 64 of those bytes (a pointer, 5 extents, and 2 lengths), a SubArray
# of one 128-160, so broadcasts over a few dozen fields cannot be launched as kernels. Since
# every extent of the array in a DataLayout is either fixed by the layout's type or
# identical to the corresponding extent of the parent array, an array is passed as its
# pointer with an extent for every dimension whose extent is not fixed by the type (a
# CompactDeviceArray), and a view as such an array with a UInt32 offset for every
# restricted dimension, plus a UInt32 extent for every restricted dimension whose extent is
# not fixed (a CompactDeviceView). They are read as CuDeviceArrays whose fixed extents are
# constants (see parent), so kernels load no other extents. The type parameters are the
# extent of each dimension `E` (with 0 for extents that are only available at runtime), the
# restricted dimensions `R` of a view, and the tuple lengths `K = length(R)` and `D`, the
# count of the stored extents (`count(iszero, E)`, or `count(d -> iszero(E[d]), R)` for a
# view).
struct CompactDeviceArray{T, N, E, D} <: DenseArray{T, N}
    ptr::LLVM.Interop.LLVMPtr{T, CUDA.AS.Global}
    dynamic_extents::NTuple{D, Int}
end
struct CompactDeviceView{T, N, E, R, K, D, A <: AbstractArray{T, N}} <:
       AbstractArray{T, N}
    parent::A
    offsets::NTuple{K, UInt32}
    dynamic_extents::NTuple{D, UInt32}
end

@inline device_array(array) = array
@inline device_array(array::CompactDeviceArray{T, N}) where {T, N} =
    CUDA.CuDeviceArray{T, N, CUDA.AS.Global}(array.ptr, size(array))

@inline Base.parent(
    data::DataLayouts.DataLayout{<:Any, <:Any, <:Any, <:Any, <:CompactDeviceArray},
) = device_array(getfield(data, :array))
Base.parent(array::CompactDeviceView) = device_array(array.parent)

DataLayouts.DataScope(::Type{<:CompactDeviceArray}) = ThisKernel()
DataLayouts.DataScope(
    ::Type{<:CompactDeviceView{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, A}},
) where {A} = DataLayouts.DataScope(A)

Base.IndexStyle(::Type{<:CompactDeviceArray}) = IndexLinear()

@inline Base.size(array::CompactDeviceArray{<:Any, N, E}) where {N, E} =
    ntuple(Val(N)) do d
        iszero(E[d]) || return Int(E[d])
        return array.dynamic_extents[count(
            e -> e <= d && iszero(E[e]),
            ntuple(identity, Val(N)),
        )]
    end
@inline Base.size(array::CompactDeviceView{<:Any, N, E, R}) where {N, E, R} =
    ntuple(Val(N)) do d
        iszero(E[d]) || return Int(E[d])
        d in R || return size(parent(array), d)
        return Int(array.dynamic_extents[count(r -> r <= d && iszero(E[r]), R)])
    end

# Index into the parent array that corresponds to an index into the view, shifted by the
# stored offset along each restricted dimension (generated, so that the position of each
# dimension in R is found when the method is generated, rather than by constant folding).
@generated function parent_index(
    array::CompactDeviceView{<:Any, N, <:Any, R},
    index::Tuple,
) where {N, R}
    shifted(d) =
        d in R ? :(index[$d] % UInt32 + array.offsets[$(findfirst(==(d), R))]) :
        :(index[$d] % UInt32)
    return :(Base.@_propagate_inbounds_meta; ($(map(shifted, 1:N)...),))
end

# Linear parent index of a Cartesian index into the view, in 32 bits (see DeviceArray).
@inline parent_linear_index(array::CompactDeviceView, index::Tuple) =
    first(last(Utilities.linear_index_and_stride(UInt32, array, index, Val(nothing))))

Base.@propagate_inbounds function Base.getindex(
    array::CompactDeviceView{<:Any, N},
    index::Vararg{Integer, N},
) where {N}
    @boundscheck checkbounds(array, index...)
    return @inbounds parent(array)[parent_linear_index(array, index)]
end

@inline linear_parent_index(
    array::CompactDeviceView{<:Any, <:Any, <:Any, R, K},
    index::Integer,
) where {R, K} =
    if iszero(K)
        index
    elseif IndexStyle(array) == IndexLinear()
        # In 32 bits: a 64-bit division by the stride takes many emulated GPU instructions.
        stride = prod(ntuple(Base.Fix1(size, array), Val(first(R) - 1))) % UInt32
        Nf = size(parent(array), first(R)) % UInt32
        f0 = first(array.offsets) % UInt32
        offset = (index - 1) % UInt32
        offset + (f0 + (offset ÷ stride) * (Nf - UInt32(1))) * stride + UInt32(1)
    else
        cart_idx =
            DataLayouts.single_axis_cartesian_index(CartesianIndices(size(array)), index)
        parent_linear_index(array, Tuple(cart_idx))
    end

Base.@propagate_inbounds function Base.getindex(
    array::CompactDeviceView,
    index::Integer,
)
    return @inbounds parent(array)[linear_parent_index(array, index)]
end

Base.@propagate_inbounds function Base.setindex!(
    array::CompactDeviceView{<:Any, N},
    value,
    index::Vararg{Integer, N},
) where {N}
    @boundscheck checkbounds(array, index...)
    @inbounds parent(array)[parent_linear_index(array, index)] = value
    return array
end

Base.@propagate_inbounds function Base.setindex!(
    array::CompactDeviceView,
    value,
    index::Integer,
)
    @inbounds parent(array)[linear_parent_index(array, index)] = value
    return array
end

Base.IndexStyle(::Type{<:CompactDeviceView{<:Any, <:Any, <:Any, <:Any, 0}}) = IndexLinear()
# A view restricted along a single dimension with a static extent of 1 (a
# single-field view along the F axis of its layout) is accessed with the
# constant-stride formula in linear_parent_index.
Base.IndexStyle(::Type{<:CompactDeviceView{<:Any, <:Any, E, R, 1}}) where {E, R} =
    isone(E[first(R)]) ? IndexLinear() : IndexCartesian()
Base.IndexStyle(::Type{<:CompactDeviceView}) = IndexCartesian()

# Index types generated by stable_view for unrestricted and restricted
# dimensions of the parent array in a DataLayout
const ViewDimIndex = Union{
    Base.Slice{<:Base.OneTo{<:Integer}},
    Base.OneTo{<:Integer},
    UnitRange{<:Integer},
}

extent_mismatch() = DimensionMismatch("DataLayout extents do not match its array")

# The extents of a layout's parent array that are fixed by the layout's type (see
# DataLayouts.inferred_parent_extents), as a type parameter: unlike a value, inference
# does not have to propagate it as a constant to find the type of the compact array.
@inline fixed_extents(array, data) =
    Val(DataLayouts.inferred_parent_extents(typeof(data), typeof(array)))

# A global array as a kernel parameter (see CompactDeviceArray), with the extents E fixed.
@inline function compact_device_array(
    array::CUDA.CuDeviceArray{T, N, CUDA.AS.Global},
    ::Val{E},
) where {T, N, E}
    length(E) == N || throw(extent_mismatch())
    dynamic = filter(d -> iszero(E[d]), ntuple(identity, Val(N)))
    compact = CompactDeviceArray{T, N, E, length(dynamic)}(
        pointer(array),
        map(d -> size(array, d), dynamic),
    )
    # Validates every extent assumption at launch time
    size(compact) == size(array) || throw(extent_mismatch())
    return compact
end

compact_device_view(array, data) = array

@inline compact_device_view(array::CUDA.CuDeviceArray{<:Any, <:Any, CUDA.AS.Global}, data) =
    compact_device_array(array, fixed_extents(array, data))

@inline compact_device_view(
    array::SubArray{
        <:Any,
        N,
        <:CUDA.CuDeviceArray{<:Any, <:Any, CUDA.AS.Global},
        <:NTuple{N, ViewDimIndex},
    },
    data::DataLayouts.DataLayout,
) where {N} = compact_device_view(array, fixed_extents(array, data))
@inline function compact_device_view(
    array::SubArray{<:Any, N, <:Any, I},
    ::Val{E},
) where {N, I, E}
    indices = parentindices(array)
    restricted = filter(d -> fieldtype(I, d) <: UnitRange, ntuple(identity, Val(N)))
    dynamic = filter(d -> iszero(E[d]), restricted)
    # A view's restricted dimensions say nothing about the extents of its parent.
    parent_extents = Val(ntuple(d -> fieldtype(I, d) <: UnitRange ? 0 : E[d], Val(N)))
    compact_parent = compact_device_array(parent(array), parent_extents)
    compact = CompactDeviceView{
        eltype(array),
        N,
        E,
        restricted,
        length(restricted),
        length(dynamic),
        typeof(compact_parent),
    }(
        compact_parent,
        map(d -> UInt32(first(indices[d]) - 1), restricted),
        map(d -> UInt32(length(indices[d])), dynamic),
    )
    # Validates every extent assumption at launch time, including that any
    # Base.OneTo indices span their full parent dimensions
    size(compact) == size(array) || throw(extent_mismatch())
    return compact
end

function Adapt.adapt_structure(to::CUDA.KernelAdaptor, data::DataLayouts.DataLayout)
    array = Adapt.adapt(to, parent(data))
    check_index_count(length(array isa SubArray ? parent(array) : array))
    return DataLayouts.rebuild(data, compact_device_view(array, data))
end

# GPU point index arithmetic is in 32 bits (see adapt_structure above): 64-bit integer ops
# take several instructions at every point. Point views have Int offsets, so that the
# offsets of their components fold into the immediate offsets of memory accesses (see
# Utilities.linear_index_and_stride).
const DeviceArray = Union{CUDA.CuDeviceArray, CompactDeviceView}
const DeviceSubArray = Utilities.UnitRangeSubArray{<:DeviceArray}

@inline Utilities.dense_array_and_index(array::CompactDeviceView, index) =
    Utilities.dense_array_and_index(parent(array), parent_index(array, index))

@inline device_offset_and_stride(array, index::CartesianIndex, val_F::Val) =
    Utilities.linear_index_and_stride(UInt32, array, Tuple(index), val_F)
@inline function device_offset_and_stride(
    array::CompactDeviceView,
    index::Integer,
    stride::Integer,
)
    # Parent indices are affine in every view coordinate, so the parent stride between the
    # components of a point is the distance between the parent indices of two of them.
    parent_index = linear_parent_index(array, index) % UInt32
    parent_stride = linear_parent_index(array, index + stride) % UInt32 - parent_index
    return (parent(array), (parent_index, parent_stride))
end
@inline device_offset_and_stride(
    array::CUDA.CuDeviceArray,
    index::Integer,
    stride::Integer,
) = (array, (index, stride % UInt32))

# One method per type: JET would union-split a Union method on hosts and report its calls.
for A in (:(CUDA.CuDeviceArray), :CompactDeviceView, :DeviceSubArray)
    @eval Base.@propagate_inbounds DataLayouts.parent_array_and_index_args(
        array::$A,
        (index, val_F)::Tuple{CartesianIndex, Val},
    ) = device_offset_and_stride(array, index, val_F)
end
# Points of CuArrays outside of kernels stay Cartesian, so that their views are SubArrays
# at Cartesian indices (which GPUArrays can broadcast over and copy), rather than SubArrays
# of ReshapedArrays at StridedRanges. Cartesian components also keep linear indices of
# layouts with several components from being converted to strided parent indices.
DataLayouts.has_cartesian_components(
    ::Union{CUDA.CuArray, Utilities.UnitRangeSubArray{<:CUDA.CuArray}},
) = true
DataLayouts.parent_array_and_index_args(
    array::Union{CUDA.CuArray, Utilities.UnitRangeSubArray{<:CUDA.CuArray}},
    index_args::Tuple{CartesianIndex, Val},
) = (array, index_args)

@inline function DataLayouts.view_struct(
    array::DeviceArray,
    ::Type{T},
    index::Integer,
    stride::Integer,
) where {T}
    Nf = DataLayouts.num_basetypes(eltype(array), T)
    (device_array, (offset, stride)) = device_offset_and_stride(array, index, stride)
    return DataLayouts.PointView{eltype(array), Nf, typeof(device_array)}(
        device_array, offset, stride,
    )
end

# CUDA 6 splits CUDA.jl into subpackages: GPUArrays is reached through
# CUDACore, and the RNG whose scalar `rand(rng, T::Type)` needs disambiguating
# below moves to cuRAND, while `CUDA.RNG` becomes GPUArrays' RNG. CUDA 5 keeps
# GPUArrays and that RNG (as `CUDA.RNG`) in CUDA itself.
const AbstractGPUArrayStyle =
    (isdefined(CUDA, :GPUArrays) ? CUDA.GPUArrays : CUDA.CUDACore.GPUArrays).AbstractGPUArrayStyle
const CUDA_RNGS =
    isdefined(CUDA, :cuRAND) ? (CUDA.RNG, CUDA.cuRAND.NativeRNG) : (CUDA.RNG,)

# Disambiguate the scalar-broadcast `copyto!` methods in ClimaCore against
# `copyto!(::AbstractArray, ::Broadcasted{<:AbstractGPUArrayStyle})` in
# GPUArrays. These styles are zero-dimensional, so ClimaCore's implementations
# (which fill the destination pointwise) are the correct ones to use. The
# corresponding StaticArrays and BlockArrays disambiguators live in
# `src/DataLayouts/loops.jl` and `src/Fields/fieldvector.jl`; these two cannot,
# because GPUArrays is not a dependency of ClimaCore.
@inline Base.copyto!(
    dest::DataLayouts.DataLayout,
    bc::Base.Broadcast.Broadcasted{<:AbstractGPUArrayStyle{0}},
    mask::DataLayouts.DataMask = DataLayouts.NoMask(),
) = DataLayouts._copyto_scalar_broadcast!(dest, bc, mask)

@inline Base.copyto!(
    dest::Fields.FieldVector,
    bc::Base.Broadcast.Broadcasted{<:AbstractGPUArrayStyle{0}},
) = Fields._copyto_scalar_broadcast!(dest, bc)

# Disambiguate `rand(::AbstractRNG, ::Type{<:Tensor})` in ClimaCore against the
# scalar `rand(rng, T::Type)` of CUDA's native RNG, which would allocate a
# `CuArray{<:Tensor}` and read back its only element. The components are drawn
# as a flat `CuArray{T}` and copied to the host: `rand(rng, C)` for a static
# array type `C` would hit the same scalar method and then an ambiguity between
# CUDA's and StaticArrays' `rand(rng, T, dims)`. A method on the `Union` of the
# RNG types would not be more specific than either scalar method, so each type
# gets its own. On CUDA 6, `CUDA.RNG` has no scalar `rand` to disambiguate
# against, but covering it makes `rand(CUDA.default_rng(), T)` work. These RNGs
# are host-side objects, so the host copy is always possible; kernels draw from
# the device RNG `Philox2x32`, which takes the generic method in Geometry.
for RNG in CUDA_RNGS
    @eval Base.rand(
        rng::$RNG,
        ::Type{Geometry.Tensor{N, T, B, C}},
    ) where {N, T, B, C} =
        Geometry.Tensor(C(Array(rand(rng, T, length(C)))), B.instance)
end
