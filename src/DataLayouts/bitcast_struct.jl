# Constructs a composite type from a tuple of its fields
@generated struct_from_fields(::Type{T}, fields) where {T} =
    Expr(:splatnew, :T, :fields)

# Flattens a data structure into a tuple of its primitive values
@inline primitives_in_struct(struct_val::T) where {T} =
    if Base.issingletontype(T)
        ()
    elseif iszero(fieldcount(T))
        (struct_val,)
    else
        fields = ntuple(Base.Fix1(getfield, struct_val), Val(fieldcount(T)))
        unrolled_flatmap(primitives_in_struct, fields)
    end

# Converts a primitive value into an integer with the same number of bytes
@inline bitcast_to_uint(primitive) =
    primitive isa Unsigned ? primitive :
    Core.bitcast(uint_type_for_size(Val(sizeof(primitive))), primitive)

# Packs a tuple of primitive values into a single value in little-endian order
@inline combine_primitives(primitives::P) where {P} =
    isone(length(primitives)) ? first(primitives) :
    unrolled_mapreduce(
        bitcast_to_uint,
        (result, uint) -> result << (8 * sizeof(uint)) | uint,
        reverse(primitives);
        init = zero(uint_type_for_size(Val(storage_size(P)))),
    )

# Converts a primitive value into a tuple of bytes in little-endian order, and
# drops the specified number of bytes from the beginning and end of the tuple
# (Note: This can be further optimized by using partitions larger than one byte)
@inline function drop_bytes(
    primitive,
    ::Val{drop_first},
    ::Val{drop_last},
) where {drop_first, drop_last}
    uint = bitcast_to_uint(primitive)
    return ntuple(Val(sizeof(uint) - drop_first - drop_last)) do index
        UInt8(uint >> (8 * (index + drop_first - 1)) & typemax(UInt8))
    end
end

"""
    bitcast_struct(T, struct_val)

Converts a data structure to any `isbits` type `T` that spans the same number of
bytes, generalizing Julia's native `Core.bitcast` function to composite types.

# Examples

```julia-repl
julia> bitcast_struct(NTuple{4, Int8}, Int32(1))
(1, 0, 0, 0)

julia> bitcast_struct(NTuple{6, Int32}, (2 * eps(0.0), eps(0.0), 0.0))
(2, 0, 1, 0, 0, 0)

julia> bitcast_struct(Tuple{Int32, Int32, Int128}, (2, 1, 0))
(2, 0, 1)
```

# Extended help

The output of `bitcast_struct` is identical to `reinterpret(T, struct_val)`,
with both functions interpreting sequential bytes of data in
[little-endian order](https://en.wikipedia.org/wiki/Endianness):

```julia-repl
julia> reinterpret(NTuple{4, Int8}, Int32(1))
(1, 0, 0, 0)

julia> reinterpret(NTuple{6, Int32}, (2 * eps(0.0), eps(0.0), 0.0))
(2, 0, 1, 0, 0, 0)

julia> reinterpret(Tuple{Int32, Int32, Int128}, (2, 1, 0))
(2, 0, 1)
```

Although their outputs are identical, `bitcast_struct` and `reinterpret` have
very different implementations. Unlike `bitcast_struct`, which performs all
conversions in register memory by calling `Core.bitcast`, `reinterpret` only
does this for `isprimitivetype` inputs, and otherwise it converts by allocating
`Ref`s and passing them to `unsafe_convert`. Since the heap memory where `Ref`s
are allocated does not exist on GPUs, `reinterpret` cannot be used to convert
composite types inside GPU kernels, and `bitcast_struct` must be used instead.

It is also important to note that `bitcast_struct` is only similar to the method
of `reinterpret` for `isbits` inputs, rather than the more common method for
`AbstractArray` inputs. These methods differ in how they handle
[padding](https://www.gnu.org/software/c-intro-and-ref/manual/html_node/Structure-Layout.html),
which the C code underlying Julia uses to ensure that fields of different sizes
within the same data structure are aligned in register memory. While the array
form of `reinterpret` only converts between types with the same amount of
padding, the `isbits` form is able to support arbitrary padding, so it can store
a padded value of type `T` in an array smaller than `sizeof(T)` (the smallest
possible size of a nonempty `Array{T}` or `ReinterpretArray{T}`).

The previous example of converting `Int64`s into a `Tuple{Int32, Int32, Int128}`
illustrates this difference. The `isbits` form of `reinterpret` only needs three
`Int64`s to perform the conversion, but the array form needs a fourth `Int64`,
spanning the eight bytes of padding used to align the `Int32`s and the `Int128`:

```julia-repl
julia> reinterpret(Tuple{Int32, Int32, Int128}, (2, 1, 0))
(2, 0, 1)

julia> reinterpret(reshape, Tuple{Int32, Int32, Int128}, [2, 1, 0])[1]
ERROR: ArgumentError: [...]

julia> reinterpret(reshape, Tuple{Int32, Int32, Int128}, [2, 0, 1, 0])[1]
(2, 0, 1)
```

For more information about `reinterpret` and padding, see the following:
- https://discourse.julialang.org/t/reinterpret-returns-wrong-values
- https://discourse.julialang.org/t/reinterpret-vector-into-single-struct
- https://discourse.julialang.org/t/reinterpret-vector-of-mixed-type-tuples
"""
@inline bitcast_struct(::Type{T}, struct_val) where {T} =
    struct_val isa T ? struct_val :
    storage_size(T) == storage_size(typeof(struct_val)) ?
    _bitcast_struct((Val(T), primitives_in_struct(struct_val))) :
    throw_wrong_storage_length(T, typeof(struct_val), 1)

@inline _bitcast_struct((_, primitives)::Tuple{Val{T}, P}) where {T, P} =
    if Base.issingletontype(T)
        T.instance
    elseif iszero(fieldcount(T))
        Core.bitcast(T, combine_primitives(primitives))
    elseif isone(fieldcount(T))
        field = _bitcast_struct((Val(fieldtype(T, 1)), primitives))
        struct_from_fields(T, (field,))
    elseif isone(unrolled_count(!Base.issingletontype, fieldtypes(T)))
        primitives_per_field = Base.setindex(
            ntuple(Returns(()), Val(fieldcount(T))),
            primitives,
            unrolled_findfirst(!Base.issingletontype, fieldtypes(T)),
        )
        map_args_per_field = zip(fieldtype_vals(T), primitives_per_field)
        struct_from_fields(T, unrolled_map(_bitcast_struct, map_args_per_field))
    else
        last_byte_index_per_field = unrolled_cumsum(storage_size, fieldtypes(T))
        last_byte_index_per_primitive = unrolled_cumsum(sizeof, fieldtypes(P))
        last_primitive_index_per_field =
            unrolled_map(last_byte_index_per_field) do index
                unrolled_findfirst(>=(index), last_byte_index_per_primitive)
            end

        is_singleton_field = unrolled_map(Base.issingletontype, fieldtypes(T))
        is_last_primitive_unpartitioned =
            unrolled_map(last_byte_index_per_field) do index
                unrolled_in(index, last_byte_index_per_primitive)
            end
        all_unpartitioned = unrolled_all(is_last_primitive_unpartitioned)

        primitives_per_field =
            ntuple(Val(fieldcount(T))) do F
                if is_singleton_field[F]
                    ()
                elseif all_unpartitioned || (
                    (F == 1 || is_last_primitive_unpartitioned[F - 1]) &&
                    is_last_primitive_unpartitioned[F]
                )
                    first_primitive_index =
                        F == 1 ? 1 : last_primitive_index_per_field[F - 1] + 1
                    last_primitive_index = last_primitive_index_per_field[F]
                    primitives[first_primitive_index:last_primitive_index]
                else
                    first_primitive_index =
                        F == 1 ? 1 :
                        last_primitive_index_per_field[F - 1] +
                        is_last_primitive_unpartitioned[F - 1]
                    previous_field_end_index =
                        F == 1 ? 0 : last_byte_index_per_field[F - 1]
                    previous_primitive_end_index =
                        first_primitive_index == 1 ? 0 :
                        last_byte_index_per_primitive[first_primitive_index - 1]
                    drop_first =
                        previous_field_end_index - previous_primitive_end_index

                    last_primitive_index = last_primitive_index_per_field[F]
                    this_field_end_index = last_byte_index_per_field[F]
                    last_primitive_end_index =
                        last_byte_index_per_primitive[last_primitive_index]
                    drop_last =
                        last_primitive_end_index - this_field_end_index

                    if first_primitive_index == last_primitive_index
                        primitive = primitives[first_primitive_index]
                        drop_bytes(primitive, Val(drop_first), Val(drop_last))
                    else
                        first_primitive = primitives[first_primitive_index]
                        first_values =
                            iszero(drop_first) ? (first_primitive,) :
                            drop_bytes(first_primitive, Val(drop_first), Val(0))

                        last_primitive = primitives[last_primitive_index]
                        last_values =
                            iszero(drop_last) ? (last_primitive,) :
                            drop_bytes(last_primitive, Val(0), Val(drop_last))

                        remaining_primitive_indices = range(
                            first_primitive_index + 1,
                            last_primitive_index - 1,
                        )
                        middle_values = primitives[remaining_primitive_indices]
                        (first_values..., middle_values..., last_values...)
                    end
                end
            end
        map_args_per_field = zip(fieldtype_vals(T), primitives_per_field)
        struct_from_fields(T, unrolled_map(_bitcast_struct, map_args_per_field))
    end

if hasfield(Method, :recursion_relation)
    for m in methods(_bitcast_struct)
        m.recursion_relation = Returns(true)
    end # Disable this recursion limit to guarantee type stability on Julia 1.10
end
