import Adapt
import CUDA
import ClimaComms
import ClimaCore: DataLayouts
import UnrolledUtilities: unrolled_findfirst, unrolled_reduce, unrolled_map, unrolled_insert

include("scopes.jl")
include("loops.jl")
include("data_layouts_threadblock.jl")

# Kernel parameters are limited to 4 KiB of memory before compute capability
# 7.0, and a SubArray of a 5-D CuDeviceArray uses 128-160 of those bytes per
# broadcast argument (64 for the parent array, 48-80 for the index ranges, and
# 16 for precomputed linear-indexing fields), so broadcasts over a few dozen
# field views cannot be launched as kernels. Since every extent of the array in
# a DataLayout is either available from the layout's type or identical to the
# corresponding extent of the parent array, the index ranges can be replaced
# with an Int32 offset for every restricted dimension, plus an Int32 extent for
# every restricted dimension whose extent is not available from the type. The
# type parameters are the extent of each dimension `E` (with 0 for extents that
# are only available at runtime), the restricted dimensions `R`, and the tuple
# lengths `K = length(R)` and `D = count(d -> iszero(E[d]), R)`.
struct CompactDeviceView{T, N, E, R, K, D, A <: AbstractArray{T, N}} <:
       AbstractArray{T, N}
    parent::A
    offsets::NTuple{K, Int32}
    dynamic_extents::NTuple{D, Int32}
end

Base.parent(array::CompactDeviceView) = array.parent

DataLayouts.DataScope(
    ::Type{<:CompactDeviceView{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, A}},
) where {A} = DataLayouts.DataScope(A)

@inline Base.size(array::CompactDeviceView{<:Any, N, E, R}) where {N, E, R} =
    ntuple(Val(N)) do d
        iszero(E[d]) || return Int(E[d])
        d in R || return size(parent(array), d)
        return Int(array.dynamic_extents[count(r -> r <= d && iszero(E[r]), R)])
    end

# The search for d runs over type-parameter constants, so it constant-folds to
# either 0 or a single tuple field access. It is unrolled, since the recursion
# in Base.findfirst does not always constant-fold within deeply nested kernels.
@inline function parent_offset(
    array::CompactDeviceView{<:Any, <:Any, <:Any, R},
    ::Val{d},
) where {R, d}
    position = unrolled_findfirst(==(d), R)
    return isnothing(position) ? 0 : Int(array.offsets[position])
end

# Integer indices (the indices of points) are shifted in 32 bits; see DeviceArray.
@inline parent_index_dim(array::CompactDeviceView, idx::Integer, ::Val{d}) where {d} =
    idx % UInt32 + parent_offset(array, Val(d)) % UInt32

@inline parent_index_dim(
    array::CompactDeviceView,
    r::AbstractUnitRange,
    ::Val{d},
) where {d} =
    (first(r) + parent_offset(array, Val(d))):(last(r) + parent_offset(array, Val(d)))

@inline parent_index_dim(
    array::CompactDeviceView{<:Any, <:Any, <:Any, R},
    s::Base.Slice,
    ::Val{d},
) where {R, d} =
    d in R ?
    ((first(s) + parent_offset(array, Val(d))):(last(s) + parent_offset(array, Val(d)))) : s

@inline parent_index_dim(
    array::CompactDeviceView{<:Any, <:Any, <:Any, R},
    ::Colon,
    ::Val{d},
) where {R, d} =
    d in R ?
    ((1 + parent_offset(array, Val(d))):(size(array, d) + parent_offset(array, Val(d)))) :
    (:)

# Index into the parent array that corresponds to an index into the view,
# shifted by the stored offset along each restricted dimension. The dimensions
# are expanded with literal Val(d) arguments, since a Val built from the index
# passed by ntuple only constant-folds when constant propagation reaches it.
@generated parent_index(array::CompactDeviceView{<:Any, N}, index::Tuple) where {N} =
    :(Base.@_propagate_inbounds_meta;
    Base.Cartesian.@ntuple $N d -> parent_index_dim(array, index[d], Val(d)))

# Linear index into the parent array of a Cartesian index into the view,
# computed in 32 bits (compact_device_view ensures that the parent's length fits
# in them): every 64-bit integer operation is two or more instructions on a GPU,
# and each point of a kernel computes one of these per array it reads or writes.
@inline function parent_linear_index(
    array::CompactDeviceView{<:Any, N},
    index::Tuple,
) where {N}
    parent_dims = size(parent(array))
    parent_idx = parent_index(array, index)
    return Int(
        unrolled_reduce(
            (linear, d) ->
                linear * (parent_dims[d] % UInt32) + (parent_idx[d] % UInt32 - UInt32(1)),
            ntuple(d -> N + 1 - d, Val(N));
            init = UInt32(0),
        ),
    ) + 1
end

Base.@propagate_inbounds function Base.getindex(
    array::CompactDeviceView{<:Any, N},
    index::Vararg{Integer, N},
) where {N}
    @boundscheck checkbounds(array, index...)
    return @inbounds parent(array)[parent_linear_index(array, index)]
end

Base.@propagate_inbounds function Base.getindex(
    array::CompactDeviceView{<:Any, N},
    index::CartesianIndex{N},
) where {N}
    @boundscheck checkbounds(array, index)
    return @inbounds parent(array)[parent_linear_index(array, Tuple(index))]
end

@inline function linear_parent_index(
    array::CompactDeviceView{<:Any, N, E, R, K},
    index::Integer,
) where {N, E, R, K}
    if iszero(K)
        return index
    elseif isone(K)
        f_dim_val = first(R)
        dims = size(array)
        stride = prod(ntuple(d -> dims[d], Val(f_dim_val - 1)))
        Nf = size(parent(array), f_dim_val)
        f0 = Int(first(array.offsets))
        # The index arithmetic is done in 32 bits, which compact_device_view
        # ensures the parent array's length fits in: a 64-bit division by the
        # constant stride is a long chain of emulated multiply-high instructions
        # on a GPU, and this runs at every point.
        i = (index - 1) % UInt32
        stride32 = stride % UInt32
        h0 = i ÷ stride32
        p0 = i - h0 * stride32
        return Int(p0 + (f0 % UInt32) * stride32 + h0 * (stride32 * (Nf % UInt32))) + 1
    else
        cart_idx = Tuple(
            DataLayouts.single_axis_cartesian_index(CartesianIndices(size(array)), index),
        )
        return parent_linear_index(array, cart_idx)
    end
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
    array::CompactDeviceView{<:Any, N},
    value,
    index::CartesianIndex{N},
) where {N}
    @boundscheck checkbounds(array, index)
    @inbounds parent(array)[parent_linear_index(array, Tuple(index))] = value
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

Base.@propagate_inbounds function Base.view(
    array::CompactDeviceView{<:Any, N},
    indices::Vararg{Any, N},
) where {N}
    return @inbounds view(parent(array), parent_index(array, indices)...)
end

# Index types generated by stable_view for unrestricted and restricted
# dimensions of the parent array in a DataLayout
const ViewDimIndex = Union{
    Base.Slice{<:Base.OneTo{<:Integer}},
    Base.OneTo{<:Integer},
    UnitRange{<:Integer},
}

# Extent of every parent array dimension that is available from a DataLayout's
# type, with 0 for dimensions whose extents are only available at runtime
@inline inferred_parent_extents(data, array) = DataLayouts.add_f_dim(
    map(extent -> something(extent, 0), DataLayouts.inferred_size(data)),
    DataLayouts.num_basetypes(eltype(array), eltype(data)),
    Val(DataLayouts.f_dim(data)),
)

compact_device_view(array, data) = array

@inline function compact_device_view(
    array::SubArray{<:Any, N, <:CUDA.CuDeviceArray, <:NTuple{N, ViewDimIndex}},
    data::DataLayouts.DataLayout,
) where {N}
    extents = inferred_parent_extents(data, array)
    length(extents) == N ||
        throw(DimensionMismatch("DataLayout extents do not match its array"))
    indices = parentindices(array)
    restricted =
        filter(d -> indices[d] isa UnitRange, ntuple(identity, Val(N)))
    dynamic = filter(d -> iszero(extents[d]), restricted)
    compact = CompactDeviceView{
        eltype(array),
        N,
        extents,
        restricted,
        length(restricted),
        length(dynamic),
        typeof(parent(array)),
    }(
        parent(array),
        map(d -> Int32(first(indices[d]) - 1), restricted),
        map(d -> Int32(length(indices[d])), dynamic),
    )
    # Validates every extent assumption at launch time, including that any
    # Base.OneTo indices span their full parent dimensions
    size(compact) == size(array) ||
        throw(DimensionMismatch("DataLayout extents do not match its array"))
    return compact
end

Adapt.adapt_structure(to::CUDA.KernelAdaptor, data::DataLayouts.DataLayout) =
    DataLayouts.rebuild(
        data,
        compact_device_view(checked_device_array(Adapt.adapt(to, parent(data))), data),
    )
# Kernels compute the linear indices of points in 32 bits (see DeviceArray)
@inline function checked_device_array(array)
    length(array isa SubArray ? parent(array) : array) <= typemax(UInt32) || throw(
        ArgumentError(
            "GPU arrays with more than $(typemax(UInt32)) elements are not supported",
        ),
    )
    return array
end

# All index arithmetic for the points of GPU arrays is done in 32 bits (every
# 64-bit integer operation is several instructions on a GPU, and a kernel
# computes an index per array it reads or writes at every point). Views of a
# compact device view are resolved to an offset and stride in the device
# array, through the view's offsets (see DataLayouts.parent_array_and_index_args),
# and point views are DevicePointViews rather than SubArrays, whose offsets Base
# computes in 64 bits.
const DeviceArray = Union{CUDA.CuDeviceArray, CompactDeviceView}
const DeviceSubArray{N} = SubArray{
    <:Any,
    N,
    <:DeviceArray,
    <:NTuple{N, Union{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}}},
}

# The device array holding the points of a view, the Cartesian index of a point
# in it (with the F axis at 1), and the ranges of the view in it.
@inline device_index(array::CUDA.CuDeviceArray, index::Tuple) =
    (array, map(i -> i % UInt32, index))
@inline device_index(array::CompactDeviceView, index::Tuple) =
    (parent(array), parent_index(array, index))
@inline device_index(array::DeviceSubArray, index::Tuple) = device_index(
    parent(array),
    unrolled_map(
        (i, range) -> i % UInt32 + (first(range) - 1) % UInt32,
        index,
        array.indices,
    ),
)

# The 1-based offset of a point and the stride of the F axis in the device
# array, both in UInt32 (the F axis of the point's index is inserted at 1).
@inline function device_offset_and_stride(
    array::AbstractArray{<:Any, N},
    index::CartesianIndex,
    ::Val{F},
) where {N, F}
    full_index = isnothing(F) ? Tuple(index) : unrolled_insert(Tuple(index), 1, Val(F))
    (device_array, device_idx) = device_index(array, full_index)
    sizes = size(device_array)
    all_slices = ntuple(Returns(Base.Slice(Base.OneTo(1))), Val(N))
    offset =
        DataLayouts.parent_offset(UInt32, device_idx, all_slices, sizes, Val(0))
    stride =
        isnothing(F) ? UInt32(1) : DataLayouts.axis_stride(UInt32, sizes, Val(F))
    return (device_array, (offset + UInt32(1), stride))
end
@inline device_offset_and_stride(
    array::CUDA.CuDeviceArray,
    index::Integer,
    stride::Integer,
) =
    (array, (index % UInt32, stride % UInt32))
@inline device_offset_and_stride(
    array::CompactDeviceView,
    index::Integer,
    stride::Integer,
) =
    (parent(array), (linear_parent_index(array, index) % UInt32, stride % UInt32))

# One method per kind of device array (rather than one Union method), so that
# the host-side abstract analysis of grid construction (JET) does not
# union-split the point-access call over these methods and report every helper
# call inside them.
for A in (:(CUDA.CuDeviceArray), :CompactDeviceView, :DeviceSubArray)
    @eval Base.@propagate_inbounds DataLayouts.parent_array_and_index_args(
        array::$A,
        (index, val_F)::Tuple{CartesianIndex, Val},
    ) = device_offset_and_stride(array, index, val_F)
end

# A single-point view of a device array: the Nf consecutive (along the F axis)
# entries of the point, read and written with 32-bit linear indices.
struct DevicePointView{T, Nf, A <: AbstractArray{T}} <: AbstractVector{T}
    parent::A
    offset::UInt32 # 1-based
    stride::UInt32
end
Base.parent(view::DevicePointView) = view.parent
Base.size(::DevicePointView{<:Any, Nf}) where {Nf} = (Nf,)
Base.IndexStyle(::Type{<:DevicePointView}) = IndexLinear()
DataLayouts.DataScope(::Type{<:DevicePointView{<:Any, <:Any, A}}) where {A} =
    DataLayouts.DataScope(A)
@inline point_index(view::DevicePointView, i::Integer) =
    view.offset + (i % UInt32 - UInt32(1)) * view.stride
Base.@propagate_inbounds function Base.getindex(view::DevicePointView, i::Integer)
    @boundscheck checkbounds(view, i)
    return @inbounds parent(view)[point_index(view, i)]
end
# Atomic updates (CUDA.@atomic in DSS kernels) take the pointer of an entry.
# Like Base's pointer, this does not check bounds: CUDA.@atomic calls it
# outside of the kernel's @inbounds, where a check would stay in every kernel.
@inline Base.pointer(view::DevicePointView, i::Integer = 1) =
    pointer(parent(view), Int(point_index(view, i)))
Base.@propagate_inbounds function Base.setindex!(view::DevicePointView, value, i::Integer)
    @boundscheck checkbounds(view, i)
    @inbounds parent(view)[point_index(view, i)] = value
    return view
end
@inline function device_point_view(array, ::Type{T}, index_args...) where {T}
    Nf = DataLayouts.num_basetypes(eltype(array), T)
    (device_array, (offset, stride)) = device_offset_and_stride(array, index_args...)
    return DevicePointView{eltype(array), Nf, typeof(device_array)}(
        device_array, offset, stride,
    )
end
for A in (:(CUDA.CuDeviceArray), :CompactDeviceView, :DeviceSubArray)
    @eval @inline DataLayouts.view_struct(
        array::$A,
        ::Type{T},
        index::CartesianIndex,
        val_F::Val,
    ) where {T} = device_point_view(array, T, index, val_F)
end
@inline DataLayouts.view_struct(
    array::DeviceArray,
    ::Type{T},
    index::Integer,
    stride::Integer,
) where {T} = device_point_view(array, T, index, stride)

@inline DataLayouts.field_offset(
    array::CompactDeviceView,
    ::Val{F},
) where {F} = Int(first(array.offsets))

@inline DataLayouts.is_constant_stride_view_type(
    ::Type{<:CompactDeviceView{<:Any, <:Any, E, R, <:Any, <:Any, A}},
    ::Val{F},
) where {E, R, A, F} =
    Base.IndexStyle(A) == Base.IndexLinear() && R == (F,) && isone(E[F])

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
