# 1) given a Vector{T}, T<:Number, construct an object of type S using the same underlying data
# e.g.
#  array = [1.0,2.0,3.0], S = Tuple{Complex{Float64},Float64} => (1.0 + 2.0im, 3.0)

# 2) for an Array{T,N}, and some 1 <= D <= N,
#   reconstruct S from a slice of array#

# The main difference between StructArrays and what we want is that we want 1 underlying (rather than # fields)
# there is no heterogeneity in the struct (can store recursive representations) of field types

# get_offset(array, S, offset) => S(...), next_offset

"""
    is_valid_basetype(::Type{T}, ::Type{S})

Determines whether an object of type `S` can be stored in an array with elements
of type `T` by recursively checking whether the non-empty fields of `S` can be
stored in such an array. If `S` is empty, this is always true.
"""
is_valid_basetype(::Type{T}, ::Type{S}) where {T, S} =
    sizeof(S) == 0 ||
    (fieldcount(S) > 0 && is_valid_basetype(T, fieldtypes(S)...))
is_valid_basetype(::Type{T}, ::Type{<:T}) where {T} = true
is_valid_basetype(::Type{T}, ::Type{S}, Ss...) where {T, S} =
    is_valid_basetype(T, S) && is_valid_basetype(T, Ss...)

"""
    check_basetype(::Type{T}, ::Type{S})

Check whether the types `T` and `S` have well-defined sizes, and whether an
object of type `S` can be stored in an array with elements of type `T`.
"""
function check_basetype(::Type{T}, ::Type{S}) where {T, S}
    if @generated
        if !isbitstype(T)
            estr = "Base type $T has indeterminate size"
            :(error($estr))
        elseif !isbitstype(S)
            estr = "Struct type $S has indeterminate size"
            :(error($estr))
        elseif !is_valid_basetype(T, S)
            estr = "Struct type $S cannot be represented using base type $T"
            :(error($estr))
        else
            :(return nothing)
        end
    else
        isbitstype(T) || error("Base type $T has indeterminate size")
        isbitstype(S) || error("Struct type $S has indeterminate size")
        is_valid_basetype(T, S) ||
            error("Struct type $S cannot be represented using base type $T")
        return nothing
    end
end

"""
    replace_basetype(::Type{T}, ::Type{T′}, ::Type{S})

Changes the type parameters of `S` to produce a new type `S′` such that, if
`is_valid_basetype(T, S)` is true, then so is `is_valid_basetype(T′, S′)`.
"""
replace_basetype(::Type{T}, ::Type{T′}, ::Type{S}) where {T, T′, S} =
    length(S.parameters) == 0 ? S :
    S.name.wrapper{replace_basetypes(T, T′, S.parameters...)...}
replace_basetype(::Type{T}, ::Type{T′}, ::Type{<:T}) where {T, T′} = T′
replace_basetype(::Type{T}, ::Type{T′}, value) where {T, T′} = value
replace_basetypes(::Type{T}, ::Type{T′}, value) where {T, T′} =
    (replace_basetype(T, T′, value),)
replace_basetypes(::Type{T}, ::Type{T′}, value, values...) where {T, T′} =
    (replace_basetype(T, T′, value), replace_basetypes(T, T′, values...)...)
# TODO: This could potentially lead to some annoying bugs, since it replaces
# type parameters instead of field types. So, if `S` has `Float64` as a
# parameter, `replace_basetype(Float64, Float32, S)` will replace that parameter
# with `Float32`, regardless of whether the parameter corresponds to any values
# stored in an object of type `S`. In other words, if a user utilizes types as
# type parameters and specializes their code on those parameters, this may
# change the results of their code.
# Note that there is no way to write `replace_basetype` using field types
# instead of type parameters, since replacing the field types of an object will
# change that object's type parameters, but there is no general way to map field
# types to type parameters.

"""
    parent_array_type(::Type{<:AbstractArray})

Returns the parent array type underlying any wrapper types, with all
dimensionality information removed.
"""
parent_array_type(::Type{<:Array{T}}) where {T} = Array{T}
parent_array_type(::Type{<:MArray{S, T, N, L}}) where {S, T, N, L} =
    MArray{S, T}
parent_array_type(::Type{<:SubArray{T, N, A}}) where {T, N, A} =
    parent_array_type(A)

"""
    promote_parent_array_type(::Type{<:AbstractArray}, ::Type{<:AbstractArray})

Given two parent array types (without any dimensionality information), promote
both the element types and the array types themselves.
"""
promote_parent_array_type(::Type{Array{T1}}, ::Type{Array{T2}}) where {T1, T2} =
    Array{promote_type(T1, T2)}
promote_parent_array_type(
    ::Type{MArray{S, T1}},
    ::Type{MArray{S, T2}},
) where {S, T1, T2} = MArray{S, promote_type(T1, T2)}
promote_parent_array_type(
    ::Type{MArray{S, T1}},
    ::Type{Array{T2}},
) where {S, T1, T2} = MArray{S, promote_type(T1, T2)}
promote_parent_array_type(
    ::Type{Array{T1}},
    ::Type{MArray{S, T2}},
) where {S, T1, T2} = MArray{S, promote_type(T1, T2)}

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

# TODO: this assumes that the field struct zero type is the same as the backing
# zero'd out memory, which should be true in all "real world" cases
# but is something that should be revisited
@inline function _mzero!(out::MArray{S, T, N, L}, FT) where {S, T, N, L}
    TdivFT = DataLayouts.typesize(FT, T)
    Base.GC.@preserve out begin
        @inbounds for i in 1:(L * TdivFT)
            Base.unsafe_store!(
                Base.unsafe_convert(Ptr{FT}, Base.pointer_from_objref(out)),
                zero(FT),
                i,
            )
        end
    end
    return out
end

@inline offset_index(start_index::CartesianIndex{N}, ::Val{D}, offset) where {N, D}= CartesianIndex(ntuple(n -> n == D ? start_index[n] + offset : start_index[n], N))

"""
    get_struct(array, S[, offset=0])

Construct an object of type `S` from the values of `array`, optionally offset by `offset` from the start of the array.
"""
@generated function get_struct(array::AbstractArray{T}, ::Type{S}, ::Val{D}, start_index::CartesianIndex) where {T, S, D}
    #if @generated
        tup = :(())
        for i in 1:fieldcount(S)
            push!(
                tup.args,
                :(get_struct(
                    array,
                    fieldtype(S, $i),
                    Val($D),
                    offset_index(start_index, Val($D), fieldtypeoffset(T, S, $i)),
                )),
            )
        end
        return quote
            Base.@_propagate_inbounds_meta
            bypass_constructor(S, $tup)
        end
    # else
    #     Base.@_propagate_inbounds_meta
    #     args = ntuple(fieldcount(S)) do i
    #         get_struct(array, fieldtype(S, i), Val(D), offset_index(start_index, Val(D), fieldtypeoffset(T, S, i)))
    #     end
    #     return bypass_constructor(S, args)
    # end
end

# recursion base case: hit array type is the same as the struct leaf type
@propagate_inbounds function get_struct(
    array::AbstractArray{S},
    ::Type{S},
    ::Val{D},
    start_index::CartesianIndex,
) where {S, D}
    @inbounds return array[start_index]
end

@generated function set_struct!(array::AbstractArray{T}, val::S, ::Val{D}, start_index::CartesianIndex) where {T, S, D}
    # if @generated
        ex = quote
            Base.@_propagate_inbounds_meta
        end
        for i in 1:fieldcount(S)
            push!(
                ex.args,
                :(set_struct!(
                    array,
                    getfield(val, $i),
                    Val($D),
                    offset_index(start_index, Val($D), fieldtypeoffset(T, S, $i)),
                )),
            )
        end
        push!(ex.args, :(return val))
        return ex
    # else
    #     Base.@_propagate_inbounds_meta
    #     for i in 1:fieldcount(S)
    #         set_struct!(
    #             array,
    #             getfield(val, i),
    #             Val(D),
    #             offset_index(start_index, Val(D), fieldtypeoffset(T, S, i)),
    #         )
    #     end
    #     return val
    # end
end

@propagate_inbounds function set_struct!(
    array::AbstractArray{S},
    val::S,
    ::Val{D},
    index::CartesianIndex,
) where {S,D}
    @inbounds array[index] = val
    val
end

if VERSION >= v"1.7.0-beta1"
    # For complex nested types (ex. wrapped SMatrix) we hit a recursion limit and de-optimize
    # We know the recursion will terminate due to the fact that bitstype fields
    # cannot be self referential so there are no cycles in get/set_struct (bounded tree)
    # TODO: enforce inference termination some other way
    if hasfield(Method, :recursion_relation)
        dont_limit = (args...) -> true
        for m in methods(get_struct)
            m.recursion_relation = dont_limit
        end
        for m in methods(set_struct!)
            m.recursion_relation = dont_limit
        end
    end
end
