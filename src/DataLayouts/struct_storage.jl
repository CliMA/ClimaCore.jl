@inline default_basetype_size(::Type{S}) where {S} =
    default_basetype_size(Val(S))
@inline field_type_by_size(::Type{S}, ::Val{num_bytes}) where {S, num_bytes} =
    field_type_by_size(Val(S), Val(num_bytes))

# Wrap each type in a Val to guarantee recursive inlining
@inline default_basetype_size(::Val{S}) where {S} =
    Base.issingletontype(S) || iszero(fieldcount(S)) ? sizeof(S) :
    unrolled_mapreduce(default_basetype_size, gcd, fieldtype_vals(S))
@inline field_type_by_size(::Val{S}, ::Val{num_bytes}) where {S, num_bytes} =
    sizeof(S) == num_bytes ? S :
    Base.issingletontype(S) || iszero(fieldcount(S)) ? nothing :
    unrolled_mapreduce(
        Base.Fix2(field_type_by_size, Val(num_bytes)),
        (option1, option2) -> isnothing(option2) ? option1 : option2,
        fieldtype_vals(S),
    )

"""
    default_basetype(S)

Finds a type that [`set_struct!`](@ref) and [`get_struct`](@ref) can use to
store either a value of type `S`, or any of the fields within such a value. If
possible, this type is found by recursively searching the `fieldtypes` of `S`;
otherwise, an unsigned integer type is selected based on the `fieldtype` sizes.
"""
@inline function default_basetype(::Type{S}) where {S}
    Base.issingletontype(S) && return UInt8
    T = field_type_by_size(S, Val(default_basetype_size(S)))
    !isnothing(T) && return T
    default_basetype_size(S) == 1 && return UInt8
    default_basetype_size(S) == 2 && return UInt16
    default_basetype_size(S) == 4 && return UInt32
    default_basetype_size(S) == 8 && return UInt64
    default_basetype_size(S) == 16 && return UInt128
end

@inline is_valid_basetype(::Type{T}, ::Type{S}) where {T, S} =
    Base.issingletontype(S) || iszero(sizeof(S) % sizeof(T))

@generated invalid_basetype_string(::Type{T}, ::Type{S}) where {T, S} =
    "Cannot store value of type $S ($(sizeof(S)) bytes) using values of type \
     $T ($(sizeof(T)) bytes)"

@inline function invalid_basetype_error(::Type{T}, ::Type{S}) where {T, S}
    F = unrolled_findfirst(Base.Fix1(!is_valid_basetype, T), fieldtypes(S))
    isnothing(F) && return throw(ArgumentError(invalid_basetype_string(T, S)))
    return invalid_basetype_error(T, fieldtype(S, F))
end

"""
    check_basetype(T, S)

Checks whether [`set_struct!`](@ref) and [`get_struct`](@ref) can use values of
type `T` to store a value of type `S`. Throws an error if this is not the case,
printing out an example of a specific field that cannot use `T` as a basetype.
"""
@inline check_basetype(::Type{T}, ::Type{S}) where {T, S} =
    is_valid_basetype(T, S) ? nothing : invalid_basetype_error(T, S)

"""
    checked_valid_basetype(T, S)

Returns either `T` or the [`default_basetype`](@ref) of `S`, depending on
whether `T` satisfies [`check_basetype`](@ref) for `S`.
"""
@inline checked_valid_basetype(::Type{T}, ::Type{S}) where {T, S} =
    is_valid_basetype(T, S) ? T : default_basetype(S)

"""
    num_basetypes(T, S)

Determines how many values of type `T` are required by [`set_struct!`](@ref) and
[`get_struct`](@ref) to store a single value of type `S`.
"""
@inline num_basetypes(::Type{T}, ::Type{S}) where {T, S} = sizeof(S) ÷ sizeof(T)

"""
    struct_field_view(array, S, Val(F), [Val(D)])

Creates a view of the data in `array` that corresponds to a particular field of
`S`, assuming that `array` has been populated by [`set_struct!`](@ref). The
field is specified through a `Val` that contains its index `F`, and it can be
loaded from the resulting view using [`get_struct`](@ref).

For multidimensional arrays with values stored along a particular dimension, the
resulting view contains the specified field from each value. By default, values
are assumed to be stored along the last array dimension, but any other dimension
can be specified through a `Val` that contains its index `D`.
"""
@inline function struct_field_view(
    array,
    ::Type{S},
    ::Val{F},
    ::Val{D} = Val(ndims(array)),
) where {S, F, D}
    num_D_indices = num_basetypes(eltype(array), fieldtype(S, F))
    last_D_index = num_basetypes(eltype(array), Tuple{fieldtypes(S)[1:F]...})
    D_indices = (last_D_index - num_D_indices + 1):last_D_index
    all_indices = Base.setindex(axes(array), D_indices, D)
    @boundscheck checkbounds(array, all_indices...)
    return Base.unsafe_view(array, all_indices...)
end

@inline check_struct_indices(array, ::Val{num_indices}) where {num_indices} =
    checkbounds(array, 1:num_indices)
@inline function check_struct_indices(
    array,
    ::Val{num_indices},
    start::Integer,
    ::Val{D} = Val(ndims(array)),
) where {num_indices, D}
    step = prod(size(array)[1:(D - 1)])
    checkbounds(array, range(start; step, length = num_indices))
end
@inline function check_struct_indices(
    array,
    ::Val{num_indices},
    index::CartesianIndex,
    ::Val{D} = Val(ndims(array)),
) where {num_indices, D}
    start = CartesianIndex(Tuple(index)[1:(D - 1)]..., 1, Tuple(index)[D:end]...)
    checkbounds(array, start:Base.setindex(start, num_indices, D))
end

@inline struct_index(i, array) = i
@inline struct_index(
    i,
    array,
    start::Integer,
    ::Val{D} = Val(ndims(array)),
) where {D} = start + (i - 1) * prod(size(array)[1:(D - 1)])
@inline struct_index(
    i,
    array,
    index::CartesianIndex,
    ::Val{D} = Val(ndims(array)),
) where {D} =
    CartesianIndex(Tuple(index)[1:(D - 1)]..., i, Tuple(index)[D:end]...)

"""
    set_struct!(array, value, [index], [Val(D)])

Populates `array` with data that represents any `isbits` `value`, using
[`bitcast_struct`](@ref) to convert `value` into entries of the array.

For multidimensional arrays with values stored along a particular dimension, an
index is used to identify the location of one value. By default, values will be
stored along the last array dimension, but any other dimension can be specified
as `Val(D)`. The target location's index should be either an integer that
corresponds to its start, or a `CartesianIndex` that contains its coordinate
along every dimension except `D`.

# Examples

```julia-repl
julia> set_struct!(zeros(Int8, 4), Int32(1))
4-element Vector{Int8}:
 1
 0
 0
 0

julia> set_struct!(zeros(Int64, 4), (Int32(2), Int32(0), Int128(1)))
4-element Vector{Int64}:
 2
 0
 1
 0

julia> set_struct!(zeros(Int64, 2, 4), (Int32(2), Int32(0), Int128(1)), 2)
2×4 Matrix{Int64}:
 0  0  0  0
 2  0  1  0

julia> set_struct!(zeros(Int64, 4, 2), (Int32(2), Int32(0), Int128(1)), 5, Val(1))
4×2 Matrix{Int64}:
 0  2
 0  0
 0  1
 0  0
```
"""
Base.@propagate_inbounds function set_struct!(array, value::S, index...) where {S}
    num_indices = num_basetypes(eltype(array), S)
    @boundscheck check_struct_indices(array, Val(num_indices), index...)
    entries = bitcast_struct(NTuple{num_indices, eltype(array)}, value)
    unrolled_foreach(enumerate(entries)) do (i, entry)
        @inbounds array[struct_index(i, array, index...)] = entry
    end
    return array
end

"""
    get_struct(array, S, [index], [Val(D)])

Loads a value of type `S` that [`set_struct!`](@ref) has stored in `array`,
using [`bitcast_struct`](@ref) to convert entries of the array into this value.

For multidimensional arrays with values stored along a particular dimension, an
index is used to identify the location of one value. By default, values are
assumed to be stored along the last array dimension, but any other dimension can
be specified as `Val(D)`. The target location's index should be either an
integer that corresponds to its start, or a `CartesianIndex` that contains its
coordinate along every dimension except `D`.

# Examples

```julia-repl
julia> get_struct(Int8[1, 0, 0, 0], Int32)
1

julia> get_struct([2, 0, 1, 0], Tuple{Int32, Int32, Int128})
(2, 0, 1)

julia> get_struct([0 0 0 0; 2 0 1 0], Tuple{Int32, Int32, Int128}, 2)
(2, 0, 1)

julia> get_struct([0 2; 0 0; 0 1; 0 0], Tuple{Int32, Int32, Int128}, 5, Val(1))
(2, 0, 1)
```
"""
Base.@propagate_inbounds function get_struct(array, ::Type{S}, index...) where {S}
    num_indices = num_basetypes(eltype(array), S)
    @boundscheck check_struct_indices(array, Val(num_indices), index...)
    return bitcast_struct(S, array, Val(num_indices), index...)
end

"""
    parent_array_type(A, [T])

Determines the array type underlying the wrapper type `A`, dropping all
parameters related to array dimensions. A new basetype `T` can be specified to
replace the original `eltype(A)`.
"""
parent_array_type(::Type{A}) where {A} = parent_array_type(A, eltype(A))
parent_array_type(::Type{<:Array}, ::Type{T}) where {T} = Array{T}
parent_array_type(::Type{<:MArray}, ::Type{T}) where {T} = MArray{<:Any, T}
parent_array_type(::Type{<:SubArray{<:Any, <:Any, A}}, ::Type{T}) where {A, T} =
    parent_array_type(A, T)
parent_array_type(
    ::Type{<:Base.ReshapedArray{<:Any, <:Any, A}},
    ::Type{T},
) where {A, T} = parent_array_type(A, T)

"""
    promote_parent_array_type(A1, A2)

Promotes two array types `A1` and `A2` generated by [`parent_array_type`](@ref),
which includes promoting their basetypes `eltype(A1)` and `eltype(A2)`.
"""
promote_parent_array_type(::Type{Array{T1}}, ::Type{Array{T2}}) where {T1, T2} =
    Array{promote_type(T1, T2)}
promote_parent_array_type(
    ::Type{MArray{<:Any, T1}},
    ::Type{MArray{<:Any, T2}},
) where {T1, T2} = MArray{<:Any, promote_type(T1, T2)}
promote_parent_array_type(
    ::Type{MArray{<:Any, T1}},
    ::Type{Array{T2}},
) where {T1, T2} = MArray{<:Any, promote_type(T1, T2)}
promote_parent_array_type(
    ::Type{Array{T1}},
    ::Type{MArray{<:Any, T2}},
) where {T1, T2} = MArray{<:Any, promote_type(T1, T2)}
