import ClimaCore: Operators, MatrixFields
import StaticArrays

# Warp shuffles work on power-of-two segments of a warp, which other sizes would straddle.
const ThisSubWarp =
    Union{map(n -> ThisSubBlock{2^n}, 0:trailing_zeros(THREADS_PER_WARP))...}
@inline shuffles_in_warp(scope) = scope isa ThisSubWarp

# Every active thread of the sub-warp has to call this at once; shfl_sync's checked lane
# conversions fold away for UInt32s and a source lane reduced to its sub-warp offset.
@inline function shuffle_point_value(scope::ThisSubBlock{N}, value, rank) where {N}
    source_lane = ((rank - one(rank)) % UInt32) & UInt32(N - 1) + one(UInt32)
    shfl(x) = CUDA.shfl_sync(subwarp_lanes(scope), x, source_lane, N % UInt32)
    return shuffle_components(shfl, value)
end

# The value of the thread at a distance of offset below (above, when flip) the calling thread
# in its sub-warp, or the calling thread's own value when there is no such thread.
@inline function shuffle_shift(scope::ThisSubBlock{N}, value, offset, flip) where {N}
    shfl(x) =
        flip ? CUDA.shfl_down_sync(subwarp_lanes(scope), x, offset % UInt32, N % UInt32) :
        CUDA.shfl_up_sync(subwarp_lanes(scope), x, offset % UInt32, N % UInt32)
    return shuffle_components(shfl, value)
end

# Mask of the active lanes in the calling thread's sub-warp.
@inline function subwarp_lanes(::ThisSubBlock{N}) where {N}
    lane_index = DataLayouts.thread_rank(ThisWarp()) - one(UInt32)
    lanes =
        N == THREADS_PER_WARP ? CUDA.FULL_MASK :
        ((one(UInt32) << N) - one(UInt32)) << (lane_index & ~UInt32(N - 1))
    return lanes & (CUDA.FULL_MASK >> (THREADS_PER_WARP - num_active_threads(ThisWarp())))
end

@inline shuffle_components(shfl::F, x::Union{Tuple, NamedTuple}) where {F} =
    map(Base.Fix1(shuffle_components, shfl), x)
@inline shuffle_components(
    shfl::F,
    x::Union{Bool, Base.BitInteger, Base.IEEEFloat},
) where {F} = shfl(x)
@inline shuffle_components(shfl::F, x::T) where {F, T} = DataLayouts.bitcast_struct(
    T,
    map(shfl, DataLayouts.bitcast_struct(shuffle_words_type(T), x)),
)
@generated shuffle_words_type(::Type{T}) where {T} =
    sizeof(T) % 4 == 0 ? NTuple{sizeof(T) ÷ 4, UInt32} :
    sizeof(T) % 2 == 0 ? NTuple{sizeof(T) ÷ 2, UInt16} : NTuple{sizeof(T), UInt8}

# Read-only RegisterArray with at most one point per thread, read by shuffles that every
# thread issues together.
struct ShuffledArray{T, N, Sz, F, Stride, S, A <: StaticArrays.SArray{<:Any, T, 1}} <:
       AbstractArray{T, N}
    array::A
end

@inline ShuffledArray(
    array::DataLayouts.RegisterArray{T, N, Sz, F, Stride},
    ::S,
) where {T, N, Sz, F, Stride, S} =
    ShuffledArray{T, N, Sz, F, Stride, S, typeof(StaticArrays.SArray(array.array))}(
        StaticArrays.SArray(array.array),
    )

@inline Base.size(::ShuffledArray{<:Any, <:Any, Sz}) where {Sz} = Sz
# Cartesian, so that register reads have constant indices and owner ranks are bounded.
@inline Base.IndexStyle(::Type{<:ShuffledArray}) = IndexCartesian()

@inline DataLayouts.rebuild(data, array::ShuffledArray, ::Type{T}; params...) where {T} =
    DataLayouts.layout_constructor(data, T; params...)(array)

Operators.allows_branching(
    ::Fields.Field{<:DataLayouts.DataLayout{<:Any, <:Any, <:Any, <:Any, <:ShuffledArray}},
) = false

Base.@propagate_inbounds function Base.getindex(
    array::ShuffledArray{<:Any, N, Sz, F, <:Any, S},
    index::Vararg{Int, N},
) where {N, Sz, F, S}
    (point, f) = DataLayouts.point_and_component(Sz, Val(F), index)
    return shuffle_point_value(S.instance, array.array[f], point)
end

# A thread reads its own points' values from its registers, without shuffles.
@inline Operators.own_values(
    data::DataLayouts.DataLayout{<:Any, <:Any, <:Any, <:Any, <:ShuffledArray},
) = DataLayouts.unchecked_rebuild(data, own_registers(parent(data)))
@inline own_registers(
    array::ShuffledArray{<:Any, <:Any, Sz, F, Stride},
) where {Sz, F, Stride} =
    DataLayouts.RegisterArray{Sz, F, Stride}(array.array)

# Points are RegisterPointViews, as for a RegisterArray (SubArrays would spill registers).
@inline DataLayouts.unchecked_view_struct(
    array::ShuffledArray,
    Nf::Val,
    index::CartesianIndex,
    F::Val,
) = DataLayouts.RegisterPointView(array, Nf, index, F)

# Converted in UInt32 (unlike Base's signed conversion), so register indices can fold.
Base.@propagate_inbounds round_point_index(indices::CartesianIndices, position) =
    DataLayouts.single_axis_cartesian_index(indices, position % UInt32)
Base.@propagate_inbounds round_point_index(indices, position) = indices[position]

# Rounds of every thread for ShuffledArray reads; threads without a point call f at the last one.
@inline function DataLayouts.update_points!(
    f::F,
    op::O,
    data::DataLayouts.DataLayout{<:Any, <:Any, <:Any, <:ThisSubWarp},
    slice_op::S,
) where {F, O, S}
    scope = DataLayouts.DataScope(data)
    indices = DataLayouts.maskable_slice_indices(NoMask(), slice_op, data)
    rank = Int(DataLayouts.thread_rank(scope))
    for offset in 0:Int(DataLayouts.num_threads(scope)):(length(indices) - 1)
        i = offset + rank
        index = @inbounds round_point_index(indices, min(i, length(indices)))
        value = @inline f(index)
        if i <= length(indices)
            (point,) = @inbounds DataLayouts.slice_every_arg(slice_op, index, data)
            @inbounds point[] = DataLayouts.updated_value(op, point, value)
        end
    end
    return data
end

# Arguments read without branching stay in registers if no thread owns more than one point.
@inline Operators.caches_in_registers(scope::ThisSubBlock, arg, ::Val{true}) =
    shuffles_in_warp(scope) &&
    owns_one_point_each(Fields.field_values(Operators.destination_template(arg)), scope)
@inline Operators.cached_values(scope::ThisSubBlock, arg, ::Val{true}) =
    Operators.caches_in_registers(scope, arg, Val(true)) ?
    shuffled_field(Operators.register_values(arg)) : Operators.materialize_buffer(arg)
@inline owns_one_point_each(data, ::ThisSubBlock{N}) where {N} =
    DataLayouts.has_inferred_size(data) && prod(DataLayouts.inferred_size(data)) <= N
@inline function shuffled_field(field)
    data = Fields.field_values(field)
    array = ShuffledArray(parent(data), DataLayouts.DataScope(data))
    return Fields.Field(DataLayouts.unchecked_rebuild(data, array), axes(field))
end

# A sub-warp reads the coefficients of neighboring rows from their threads' registers.
@inline MatrixFields.pcr_neighbors(
    scope::ThisSubWarp,
    (row,)::Tuple{Any},
    positions,
    stride,
    n,
) = ((shuffle_shift(scope, row, stride, false), shuffle_shift(scope, row, stride, true)),)
