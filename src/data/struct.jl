# 1) given a Vector{T}, T<:Number, construct an object of type S using the same underlying data
# e.g.
#  array = [1.0,2.0,3.0], S = Tuple{Complex{Float64},Float64} => (1.0 + 2.0im, 3.0)

# 2) for an Array{T,N}, and some 1 <= D <= N,
#   reconstruct S from a slice of array#

# The main difference between StructArrays and what we want is that we want 1 underlying (rather than # fields)
# there is no heterogeneity in the struct (can store recursive representations) of field types

# get_offset(array, S, offset) => S(...), next_offset
import Base: @propagate_inbounds

"""
    StructArrays.bypass_constructor(T, args)

Create an instance of type `T` from a tuple of field values `args`, bypassing
possible internal constructors. `T` should be a concrete type.
"""
@generated function bypass_constructor(::Type{T}, args) where {T}
    vars = ntuple(_ -> gensym(), fieldcount(T))
    assign = [
        :($var::$(fieldtype(T, i)) = getfield(args, $i)) for
        (i, var) in enumerate(vars)
    ]
    construct = Expr(:new, :T, vars...)
    Expr(:block, assign..., construct)
end

"""
    fieldtypeoffset(T,S,i)

Similar to `fieldoffset(S,i)`, but gives result in multiples of `sizeof(T)` instead of bytes.
"""
fieldtypeoffset(::Type{T}, ::Type{S}, i) where {T, S} =
    Int(div(fieldoffset(S, i), sizeof(T)))

"""
    typesize(T,S)

Similar to `sizeof(S)`, but gives the result in multiples of `sizeof(T)`.
"""
typesize(::Type{T}, ::Type{S}) where {T, S} = div(sizeof(S), sizeof(T))


"""
    get_struct(array, S[, offset=0])

Construct an object of type `S` from the values of `array`, optionally offset by `offset` from the start of the array.
"""
function get_struct(array::AbstractArray{T}, ::Type{S}, offset) where {T, S}
    if @generated
        tup = :(())
        for i in 1:fieldcount(S)
            push!(
                tup.args,
                :(get_struct(
                    array,
                    fieldtype(S, $i),
                    offset + fieldtypeoffset(T, S, $i),
                )),
            )
        end
        :(bypass_constructor(S, $tup))
    else
        args = ntuple(fieldcount(S)) do i
            get_struct(array, fieldtype(S, i), offset + fieldtypeoffset(T, S, i))
        end
        return bypass_constructor(S, args)
    end
end
@propagate_inbounds function get_struct(
    array::AbstractArray{S},
    ::Type{S},
    offset,
) where {S}
    return array[offset + 1]
end

@propagate_inbounds get_struct(
    array::AbstractArray{T},
    ::Type{S},
) where {T, S} = get_struct(array, S, 0)

function set_struct!(array::AbstractArray{T}, val::S, offset) where {T, S}
    if @generated
        errorstring = "Expected type $T, got type $S"
        ex = quote
            # TODO: need to figure out a better way to handle the case where we require conversion
            # e.g. if T = Dual or Double64
            if isprimitivetype(S)
                error($errorstring)
            end
            # TODO: we get a segfault here when trying to pass propogate_inbounds
            # with a generated function ctx (in the quote block or attached to the generated function)
            # passing it in the quoted expr for args seems to work :( @propagate_inbounds set_struct! ...)
            # https://github.com/JuliaArrays/StaticArrays.jl/blob/52fc10278667dd5fa82ded1edcfd5f7fedfae1c4/src/indexing.jl#L16-L38
            # Base.@_propagate_inbounds_meta
        end
        for i in 1:fieldcount(S)
            push!(
                ex.args,
                :(set_struct!(
                    array,
                    getfield(val, $i),
                    offset + fieldtypeoffset(T, S, $i),
                )),
            )
        end
        ex
    else
        if isprimitivetype(S)
            return error("Expected type $T, got type $S")
        end
        for i in 1:fieldcount(S)
            set_struct!(
                array,
                getfield(val, i),
                offset + fieldtypeoffset(T, S, i),
            )
        end
    end
end
@propagate_inbounds function set_struct!(
    array::AbstractArray{S},
    val::S,
    offset,
) where {S}
    array[offset + 1] = val
end
@propagate_inbounds function set_struct!(array, val)
    set_struct!(array, val, 0)
end
