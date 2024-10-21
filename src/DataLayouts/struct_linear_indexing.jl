#####
##### Linear indexing
#####

@inline function offset_index_linear(
    base_index::Integer,
    field_offset,
    array_size,
)
    return @inbounds base_index + prod(array_size[1:(end - 1)]) * field_offset
end

Base.@propagate_inbounds @generated function get_struct_linear(
    array::AbstractArray{T},
    ::Type{S},
    start_index::Integer,
    array_size,
    base_index = start_index,
) where {T, S}
    tup = :(())
    for i in 1:fieldcount(S)
        push!(
            tup.args,
            :(get_struct_linear(
                array,
                fieldtype(S, $i),
                offset_index_linear(
                    base_index,
                    $(fieldtypeoffset(T, S, Val(i))),
                    array_size,
                ),
                array_size,
                base_index,
            )),
        )
    end
    return quote
        Base.@_propagate_inbounds_meta
        @inbounds bypass_constructor(S, $tup)
    end
end

# recursion base case: hit array type is the same as the struct leaf type
Base.@propagate_inbounds function get_struct_linear(
    array::AbstractArray{S},
    ::Type{S},
    start_index::Integer,
    array_size,
    base_index = start_index,
) where {S}
    @inbounds return array[start_index]
end

"""
    set_struct!(array, val::S, Val(D), start_index)
Store an object `val` of type `S` packed along the `D` dimension, into `array`,
starting at `start_index`.
"""
Base.@propagate_inbounds @generated function set_struct_linear!(
    array::AbstractArray{T},
    val::S,
    start_index::Integer,
    array_size,
    base_index = start_index,
) where {T, S}
    ex = quote
        Base.@_propagate_inbounds_meta
    end
    for i in 1:fieldcount(S)
        push!(
            ex.args,
            :(set_struct_linear!(
                array,
                getfield(val, $i),
                offset_index_linear(
                    base_index,
                    $(fieldtypeoffset(T, S, Val(i))),
                    array_size,
                ),
                array_size,
                base_index,
            )),
        )
    end
    push!(ex.args, :(return val))
    return ex
end

Base.@propagate_inbounds function set_struct_linear!(
    array::AbstractArray{S},
    val::S,
    start_index::Integer,
    array_size,
    base_index = start_index,
) where {S}
    @inbounds array[start_index] = val
    val
end
