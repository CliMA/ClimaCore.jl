@inline default_basetype_size(::Type{T}) where {T} =
    default_basetype_size(Val(T))
@inline field_type_by_size(::Type{T}, ::Val{num_bytes}) where {T, num_bytes} =
    field_type_by_size(Val(T), Val(num_bytes))

# Wrap each type in a Val to guarantee recursive inlining
@inline default_basetype_size(::Val{T}) where {T} =
    Base.issingletontype(T) || iszero(fieldcount(T)) ? sizeof(T) :
    unrolled_mapreduce(default_basetype_size, gcd, fieldtype_vals(T))
@inline field_type_by_size(::Val{T}, ::Val{num_bytes}) where {T, num_bytes} =
    sizeof(T) == num_bytes ? T :
    Base.issingletontype(T) || iszero(fieldcount(T)) ? nothing :
    unrolled_mapreduce(
        Base.Fix2(field_type_by_size, Val(num_bytes)),
        (option1, option2) -> isnothing(option2) ? option1 : option2,
        fieldtype_vals(T),
    )

"""
    default_basetype(T)

Finds a type that [`set_struct!`](@ref) and [`get_struct`](@ref) can use to
store either a value of type `T`, or any of the fields within such a value. If
possible, this type is found by recursively searching the `fieldtypes` of `T`;
otherwise, an unsigned integer type is selected based on the `fieldtype` sizes.
"""
@inline function default_basetype(::Type{T}) where {T}
    Base.issingletontype(T) && return UInt8
    B = field_type_by_size(T, Val(default_basetype_size(T)))
    !isnothing(B) && return B
    default_basetype_size(T) == 1 && return UInt8
    default_basetype_size(T) == 2 && return UInt16
    default_basetype_size(T) == 4 && return UInt32
    default_basetype_size(T) == 8 && return UInt64
    default_basetype_size(T) == 16 && return UInt128
end

@inline is_valid_basetype(::Type{B}, ::Type{T}) where {B, T} =
    Base.issingletontype(T) || iszero(sizeof(T) % sizeof(B))

@inline function invalid_basetype_error(::Type{B}, ::Type{T}) where {B, T}
    i = unrolled_findfirst(Base.Fix1(!is_valid_basetype, B), fieldtypes(T))
    isnothing(i) && return throw(ArgumentError(invalid_basetype_string(B, T)))
    return invalid_basetype_error(B, fieldtype(T, i))
end
@generated invalid_basetype_string(::Type{B}, ::Type{T}) where {B, T} =
    "Cannot store value of type $T ($(sizeof(T)) bytes) using values of type \
     $B ($(sizeof(B)) bytes)"

"""
    check_basetype(B, T)

Checks whether [`set_struct!`](@ref) and [`get_struct`](@ref) can use values of
type `B` to store a value of type `T`. Throws an error if this is not the case,
printing out an example of a specific field that cannot use `B` as a basetype.
"""
@inline check_basetype(::Type{B}, ::Type{T}) where {B, T} =
    is_valid_basetype(B, T) ? nothing : invalid_basetype_error(B, T)

"""
    checked_valid_basetype(B, T)

Returns either `B` or the [`default_basetype`](@ref) of `T`, depending on
whether `B` satisfies [`check_basetype`](@ref) for `T`.
"""
@inline checked_valid_basetype(::Type{B}, ::Type{T}) where {B, T} =
    is_valid_basetype(B, T) ? B : default_basetype(T)

"""
    num_basetypes(B, T)

Determines how many values of type `B` are required by [`set_struct!`](@ref) and
[`get_struct`](@ref) to store a single value of type `T`.
"""
@inline num_basetypes(::Type{B}, ::Type{T}) where {B, T} = sizeof(T) ÷ sizeof(B)

# Base's fieldoffset lowers to a ccall that cannot be compiled in GPU kernels,
# so the generated branch evaluates it at compile time and returns the offset as
# a constant. The identical non-generated branch is still needed for inference
# to properly analyze this function when the field index is a runtime value
# (e.g., data.:(i) in a loop over i), where it infers to an Int and keeps view
# types concrete; a plain @generated function would infer to Any in that case.
@inline stable_fieldoffset(::Type{T}, ::Val{i}) where {T, i} =
    if @generated
        fieldoffset(T, i)
    else
        fieldoffset(T, i)
    end

"""
    struct_field_view(array, T, Val(i), [Val(F)])

Creates a view of the data in `array` that corresponds to a particular field of
`T`, assuming that `array` has been populated by [`set_struct!`](@ref). The
field is specified through a `Val` that contains its index `i`, and it can be
loaded from the resulting view using [`get_struct`](@ref).

For multidimensional arrays with values stored along a particular dimension, the
resulting view contains the specified field from each value, with the dimension
identified by a `Val` that contains its index `F`. When there is no such
dimension, `F` may be replaced with `nothing`.
"""
@inline function struct_field_view(array, ::Type{T}, ::Val{i}, ::Val{F}) where {T, i, F}
    check_basetype(eltype(array), fieldtype(T, i))
    num_D_indices = num_basetypes(eltype(array), fieldtype(T, i))
    first_D_index = Int(stable_fieldoffset(T, Val(i))) ÷ sizeof(eltype(array)) + 1
    D_indices = first_D_index:(first_D_index + num_D_indices - 1)
    other_indices = ntuple(Returns(:), Val(ndims(array))) # Use colons for FastSubArray view.
    all_indices =
        isnothing(F) ? other_indices : unrolled_setindex(other_indices, D_indices, Val(F))
    @boundscheck checkbounds(array, all_indices...)
    return @inbounds stable_view(array, all_indices...)
end

@inline single_index(index, ::Val{Nf}) where {Nf} =
    isone(Nf) ? Tuple(index) : throw(ArgumentError("F axis is required unless Nf = 1"))

@inline struct_index(i, array) = i
@inline struct_indices(array, ::Val{Nf}) where {Nf} = (Base.OneTo(Nf),)

# Split Cartesian index ranges into scalar components and a range from 1 to Nf;
# a Cartesian range would build a much costlier view with singleton dimensions.
@inline struct_index(i, array, index::CartesianIndex, ::Val{F}) where {F} =
    isnothing(F) ? index : CartesianIndex(unrolled_insert(Tuple(index), i, Val(F)))
@inline struct_indices(array, ::Val{Nf}, index::CartesianIndex, ::Val{F}) where {Nf, F} =
    isnothing(F) ? single_index(index, Val(Nf)) :
    unrolled_insert(Tuple(index), Base.OneTo(Nf), Val(F))

@inline struct_index(i, array, index::Integer, ::Val{F}) where {F} =
    isnothing(F) ? index : struct_index(i, array, index, prod(size(array)[1:(F - 1)]))
@inline struct_indices(array, ::Val{Nf}, index::Integer, ::Val{F}) where {Nf, F} =
    isnothing(F) ? single_index(index, Val(Nf)) :
    struct_indices(array, Val(Nf), index, prod(size(array)[1:(F - 1)]))

@inline struct_index(i, array, index::Integer, stride::Integer) = index + (i - 1) * stride
@inline struct_indices(array, ::Val{Nf}, index::Integer, stride::Integer) where {Nf} =
    (range(index; step = stride, length = Nf),)

"""
    set_struct!(array, value, [index, Val(F)])
    set_struct!(array, value, [index, stride])

Populates `array` with data that represents any `isbits` `value`, using
[`bitcast_struct`](@ref) to convert `value` into entries of the array.

For multidimensional arrays with values stored along a particular dimension, an
index is used to identify the location of one value, with the dimension
specified as `Val(F)`. The target values's index should be a `CartesianIndex`
that contains its coordinate along every dimension except `F`. When there is no
such dimension, `F` may be replaced with `nothing`.

Arrays that support linear indexing can also be accessed using two integers,
where one corresponds to the start of a value, and another corresponds to the
stride along the `F` axis between consecutive components of the value.

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

julia> set_struct!(zeros(Int64, 4, 2), (Int32(2), Int32(0), Int128(1)), 5, Val(1))
4×2 Matrix{Int64}:
 0  2
 0  0
 0  1
 0  0

julia> set_struct!(zeros(Int64, 3, 4), (Int32(2), Int32(0), Int128(1)), 2, 3)
3×4 Matrix{Int64}:
 0  0  0  0
 2  0  1  0
 0  0  0  0
```
"""
@inline function set_struct!(array, value::T, index...) where {T}
    Nf = num_basetypes(eltype(array), T)
    @boundscheck checkbounds(array, struct_indices(array, Val(Nf), index...)...)
    entries = bitcast_struct(NTuple{Nf, eltype(array)}, value)
    unrolled_foreach(enumerate(entries)) do (i, entry)
        @inbounds array[struct_index(i, array, index...)] = entry
    end
    return array
end

"""
    get_struct(array, T, [index, Val(F)])
    get_struct(array, T, [index, stride])

Loads a value of type `T` that [`set_struct!`](@ref) has stored in `array`,
using [`bitcast_struct`](@ref) to convert entries of the array into this value.

For multidimensional arrays with values stored along a particular dimension, an
index is used to identify the location of one value, with the dimension
specified as `Val(F)`. The target values's index should be a `CartesianIndex`
that contains its coordinate along every dimension except `F`. When there is no
such dimension, `F` may be replaced with `nothing`.

Arrays that support linear indexing can also be accessed using two integers,
where one corresponds to the start of a value, and another corresponds to the
stride along the `F` axis between consecutive components of the value.

# Examples

```julia-repl
julia> get_struct(Int8[1, 0, 0, 0], Int32)
1

julia> get_struct([2, 0, 1, 0], Tuple{Int32, Int32, Int128})
(2, 0, 1)

julia> get_struct([0 2; 0 0; 0 1; 0 0], Tuple{Int32, Int32, Int128}, 5, Val(1))
(2, 0, 1)

julia> get_struct([0 0 0 0; 2 0 1 0; 0 0 0 0], Tuple{Int32, Int32, Int128}, 2, 3)
(2, 0, 1)
```
"""
@inline function get_struct(array, ::Type{T}, index...) where {T}
    Nf = num_basetypes(eltype(array), T)
    @boundscheck checkbounds(array, struct_indices(array, Val(Nf), index...)...)
    return bitcast_struct(T, array, Val(Nf), index...)
end

"""
    view_struct(array, T, [index, Val(F)])

Analogous to [`get_struct`](@ref), but for a view of the struct data instead of
the value itself. The value may be accessed with `get_struct(struct_view, T)`,
and it can be updated with `set_struct!(struct_view, new_value)`.
"""
@inline function view_struct(array, ::Type{T}, index...) where {T}
    Nf = num_basetypes(eltype(array), T)
    @boundscheck checkbounds(array, struct_indices(array, Val(Nf), index...)...)
    return @inbounds stable_view(array, struct_indices(array, Val(Nf), index...)...)
end
