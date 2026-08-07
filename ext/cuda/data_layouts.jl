import Adapt
import CUDA
import ClimaComms
import ClimaCore: DataLayouts

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
# either 0 or a single tuple field access.
@inline function parent_offset(
    array::CompactDeviceView{<:Any, <:Any, <:Any, R},
    ::Val{d},
) where {R, d}
    position = findfirst(==(d), R)
    return isnothing(position) ? 0 : Int(array.offsets[position])
end

@inline parent_index_dim(array::CompactDeviceView, idx::Integer, ::Val{d}) where {d} =
    idx + parent_offset(array, Val(d))

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
# shifted by the stored offset along each restricted dimension
@inline parent_index(array::CompactDeviceView{<:Any, N}, index::Tuple) where {N} =
    ntuple(d -> parent_index_dim(array, index[d], Val(d)), Val(N))

Base.@propagate_inbounds function Base.getindex(
    array::CompactDeviceView{<:Any, N},
    index::Vararg{Integer, N},
) where {N}
    @boundscheck checkbounds(array, index...)
    return @inbounds parent(array)[parent_index(array, index)...]
end

Base.@propagate_inbounds function Base.getindex(
    array::CompactDeviceView{<:Any, N},
    index::CartesianIndex{N},
) where {N}
    @boundscheck checkbounds(array, index)
    return @inbounds parent(array)[parent_index(array, Tuple(index))...]
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
        h0 = (index - 1) ÷ stride
        p0 = (index - 1) % stride
        return p0 + f0 * stride + h0 * (stride * Nf) + 1
    else
        cart_idx = Tuple(CartesianIndices(size(array))[index])
        parent_cart_idx = parent_index(array, cart_idx)
        return DataLayouts.linear_index(size(parent(array)), parent_cart_idx)
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
    @inbounds parent(array)[parent_index(array, index)...] = value
    return array
end

Base.@propagate_inbounds function Base.setindex!(
    array::CompactDeviceView{<:Any, N},
    value,
    index::CartesianIndex{N},
) where {N}
    @boundscheck checkbounds(array, index)
    @inbounds parent(array)[parent_index(array, Tuple(index))...] = value
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
    array = CompactDeviceView{
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
    size(array) == size(array) ||
        throw(DimensionMismatch("DataLayout extents do not match its array"))
    return array
end

Adapt.adapt_structure(to::CUDA.KernelAdaptor, data::DataLayouts.DataLayout) =
    DataLayouts.rebuild(
        data,
        compact_device_view(Adapt.adapt(to, parent(data)), data),
    )

@inline DataLayouts.field_offset(
    array::CompactDeviceView,
    ::Val{F},
) where {F} = Int(first(array.offsets))

@inline DataLayouts.is_constant_stride_view_type(
    ::Type{<:CompactDeviceView{<:Any, <:Any, E, R, <:Any, <:Any, A}},
    ::Val{F},
) where {E, R, A, F} =
    Base.IndexStyle(A) == Base.IndexLinear() && R == (F,) && isone(E[F])
