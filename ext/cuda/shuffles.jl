import ClimaCore: Operators
import StaticArrays

# Whether the threads of a sub-block can read each other's registers with warp
# shuffles over their own lanes: the sub-block has to fit in a warp, and its
# size has to be a power of two, since shuffles split a warp into segments of
# a power-of-two width, and a sub-block of any other size would straddle two
# segments (or two warps). Sub-blocks are halved from a power of two down to
# MIN_THREADS_PER_SUBBLOCK threads, so every sub-block of at most a warp
# qualifies; the second condition only guards against other sizes.
@inline shuffles_in_warp(::ThisSubBlock{N}) where {N} = N <= THREADS_PER_WARP && ispow2(N)

# Return the value held by the thread with the given rank in a sub-warp, which
# every active thread of the sub-warp has to request at the same time, while
# other sub-warps of the same warp may be idle (e.g., at the end of a strided
# loop over slices).
#
# The lane index and width are passed to CUDA.shfl_sync as UInt32s: it converts
# them with checked conversions, which add an exception branch per shuffle to a
# kernel for a 64-bit rank, and the first lane of the sub-block is the lane
# index with its low bits cleared. The source lane is also reduced to its offset
# within the sub-warp, the only bits that a shuffle of width N reads: shfl_sync
# subtracts an Int one from it before the checked conversion, whose exception
# branch only folds away when the lane is known to be positive.
@inline function shuffle_point_value(scope::ThisSubBlock{N}, value, rank) where {N}
    lane_index = DataLayouts.thread_rank(ThisWarp()) - one(UInt32)
    lanes =
        N == THREADS_PER_WARP ? CUDA.FULL_MASK :
        ((one(UInt32) << N) - one(UInt32)) << (lane_index & ~UInt32(N - 1))
    active = CUDA.FULL_MASK >> (THREADS_PER_WARP - num_active_threads(ThisWarp()))
    source_lane = ((rank - one(rank)) % UInt32) & UInt32(N - 1) + one(UInt32)
    shfl(x) = CUDA.shfl_sync(lanes & active, x, source_lane, N % UInt32)
    return shuffle_components(shfl, value)
end

# Shuffle the components of values that CUDA.shfl_sync does not support as a
# whole, like the Tuples and NamedTuples of multi-component fields. Any other
# value that is not a number (e.g., a struct with Bool entries, which a
# DataLayout stores whole when its fields do not fit the basetype of its array)
# is shuffled as a sequence of words that span its bytes.
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

# Read-only counterpart of a DataLayouts.RegisterArray in which no thread holds
# more than one point, whose entries every thread of the sub-warp S can read: an
# entry is fetched from the registers of the thread that owns its point with a
# warp shuffle, so every thread of S has to read the same entry at the same time,
# including threads that own no point (see update_points! below).
struct ShuffledArray{
    T, N, Sz, F, Stride, S, A <: StaticArrays.SArray{<:Any, T, 1},
} <: AbstractArray{T, N}
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
@inline Base.IndexStyle(::Type{<:ShuffledArray}) = IndexLinear()

@inline DataLayouts.rebuild(data, array::ShuffledArray, ::Type{T}; params...) where {T} =
    DataLayouts.layout_constructor(data, T; params...)(array)

DataLayouts.requires_lockstep(::Type{<:ShuffledArray}) = true

# Like a RegisterArray, a point is read through the Cartesian method below, so
# that the entry this thread shuffles has a compile-time index (see
# DataLayouts.has_cartesian_components).
@inline DataLayouts.has_cartesian_components(::ShuffledArray) = true

# Read component f of the point at a linear index into the full array from the
# thread with the point's rank, which stores it in its own entry f + 1 (see
# DataLayouts.register_index); every divisor is a compile-time constant.
Base.@propagate_inbounds function Base.getindex(
    array::ShuffledArray{<:Any, <:Any, Sz, F, Stride, S},
    index::Int,
) where {Sz, F, Stride, S}
    (; Nf, SB) = DataLayouts.register_array_params(Sz, Val(F))
    (rest, before) = divrem(index - 1, SB)
    (after, f) = divrem(rest, Nf)
    point = after * SB + before
    return shuffle_point_value(S.instance, array.array[f + 1], point + 1)
end
# Like the Cartesian getindex of a RegisterArray, the component and the point
# are taken from the index rather than divided out of a linear index, so that
# the entry read from this thread's registers has a compile-time index, and
# the compiler can bound the rank of the thread that owns the point.
Base.@propagate_inbounds function Base.getindex(
    array::ShuffledArray{<:Any, N, Sz, F, <:Any, S},
    index1::Int,
    index2::Int,
    indices::Int...,
) where {N, Sz, F, S}
    index = (index1, index2, indices...)
    f = isnothing(F) ? 1 : index[F]
    point = DataLayouts.linear_index(
        DataLayouts.drop_f_dim(Sz, Val(F)),
        DataLayouts.drop_f_dim(index, Val(F)),
    )
    return shuffle_point_value(S.instance, array.array[f], point)
end
# A CartesianIndex is splatted into the method above, since Base converts it
# into a linear index for an IndexLinear array.
Base.@propagate_inbounds Base.getindex(
    array::ShuffledArray{<:Any, N},
    index::CartesianIndex{N},
) where {N} = array[Tuple(index)...]

# A point of a ShuffledArray is read through a DataLayouts.RegisterPointView,
# like a point of the RegisterArray it was made from, so that every component
# is read with a compile-time index. The generic point view (a SubArray) reads
# the components at linear indices, from which the getindex above divides the
# component out at run time: the register array is then indexed at run time,
# which moves it to local memory, with a stack load before every shuffle.
@inline function DataLayouts.view_struct(
    array::ShuffledArray{B, N},
    ::Type{T},
    index::CartesianIndex,
    ::Val{F},
) where {B, N, T, F}
    Nf = DataLayouts.num_basetypes(B, T)
    @boundscheck checkbounds(
        array,
        DataLayouts.struct_indices(array, Val(Nf), index, Val(F))...,
    )
    return DataLayouts.RegisterPointView{B, Nf, N, F, typeof(array)}(
        array,
        DataLayouts.add_f_dim(Tuple(index), 1, Val(F)),
    )
end

# The index at a position of a lockstep loop's indices. A Cartesian index is
# converted from its position with unsigned 32-bit shifts and masks (the
# slice's extents are compile-time constants), so that the compiler can bound
# each of its components and fold the register index of the point (see
# DataLayouts.register_index) to a constant; Base's signed conversion hides the
# bounds. Linear indices (point loops over eachindex) are read as they are.
Base.@propagate_inbounds lockstep_point_index(indices::CartesianIndices, position) =
    DataLayouts.single_axis_cartesian_index(indices, position % UInt32)
Base.@propagate_inbounds lockstep_point_index(indices, position) = indices[position]

# Every thread of a sub-warp calls f in each round of the loop over its points,
# so that f can read values that other threads hold in their registers (see
# ShuffledArray); in a round without a point of its own, a thread calls f at the
# last point and discards the result. Like the default method, each thread owns
# the strided subset rank:num_threads(scope):Np of the points.
@inline function DataLayouts.update_points!(
    f::F,
    op::O,
    data::DataLayouts.DataLayout{<:Any, <:Any, <:Any, <:ThisSubBlock},
    slice_op::S,
) where {F, O, S}
    scope = DataLayouts.DataScope(data)
    shuffles_in_warp(scope) ||
        return Base.@invoke DataLayouts.update_points!(f::F, op::O, data::Any, slice_op::S)
    indices = DataLayouts.maskable_slice_indices(scope, NoMask(), slice_op, data)
    rank = Int(DataLayouts.thread_rank(scope))
    for offset in 0:Int(DataLayouts.num_threads(scope)):(length(indices) - 1)
        i = offset + rank
        index = @inbounds lockstep_point_index(indices, min(i, length(indices)))
        value = @inline f(index)
        if i <= length(indices)
            (point,) = @inbounds DataLayouts.slice_every_arg(slice_op, index, data)
            @inbounds point[] = DataLayouts.updated_value(op, point, value)
        end
    end
    return data
end

# On a sub-warp, an argument that is read in lockstep is kept in the registers of
# the threads that compute its values when none of them owns more than one of its
# points, and read from there with warp shuffles; other arguments are published
# through shared memory.
@inline Operators.cached_values(scope::ThisSubBlock, arg, ::Val{true}) =
    shuffles_in_warp(scope) &&
    owns_one_point_each(Fields.field_values(Operators.point_template(arg)), scope) ?
    shuffled_field(Operators.register_values(arg)) : Operators.materialize_buffer(arg)
@inline owns_one_point_each(data, ::ThisSubBlock{N}) where {N} =
    DataLayouts.has_inferred_size(data) && prod(DataLayouts.inferred_size(data)) <= N
@inline shuffled_field(field) =
    Fields.Field(shuffled(Fields.field_values(field)), axes(field))
@inline shuffled(data) = DataLayouts.rebuild(
    data,
    ShuffledArray(parent(data), DataLayouts.DataScope(data)),
    eltype(data),
)
