"""
    RecursiveApply

This module contains operators to recurse over nested `Tuple`s or `NamedTuple`s.

To extend to another type `T`, define `RecursiveApply.rmap(fn, args::T...)`
"""
module RecursiveApply
using UnrolledUtilities
export ⊞, ⊠, ⊟

# These functions need to be generated for type stability (since T.parameters is
# a SimpleVector, the compiler cannot always infer its size and elements).
@generated params_as_tuple(::Type{T}) where {T} = :($((T.parameters...,)))
@generated first_param(::Type{T}) where {T} = :($(first(T.parameters)))
@generated tail_params(::Type{T}) where {T} =
    :($(Tuple{Base.tail((T.parameters...,))...}))

# Applying `rmaptype` returns `Tuple{...}` for tuple
# types, which cannot follow the recursion pattern as
# it cannot be splatted, so we add a separate method,
# `rmaptype_Tuple`, for the part of the recursion.
rmaptype_Tuple(fn::F, ::Type{Tuple{}}) where {F} = ()
rmaptype_Tuple(fn::F, ::Type{T}) where {F, E, T <: Tuple{E}} =
    (rmaptype(fn, first_param(T)),)
function rmaptype_Tuple(fn::F, ::Type{T}) where {F, T <: Tuple}
    UnrolledUtilities.unrolled_map(params_as_tuple(T)) do param
        rmaptype(fn, param)
    end
end

rmaptype_Tuple(_, ::Type{Tuple{}}, ::Type{Tuple{}}) = ()
rmaptype_Tuple(fn::F, ::Type{<:Tuple{A}}, ::Type{<:Tuple{B}}) where {F, A, B} =
    (rmaptype(fn, A, B),)
rmaptype_Tuple(_, ::Type{Tuple{}}, ::Type{T}) where {T <: Tuple} = ()
rmaptype_Tuple(_, ::Type{T}, ::Type{Tuple{}}) where {T <: Tuple} = ()
function rmaptype_Tuple(
    fn::F,
    ::Type{T1},
    ::Type{T2},
) where {F, T1 <: Tuple, T2 <: Tuple}
    UnrolledUtilities.unrolled_map(
        params_as_tuple(T1),
        params_as_tuple(T2),
    ) do param1, param2
        rmaptype(fn, param1, param2)
    end
end
function rmaptype_Tuple(
    fn::F,
    ::Type{T1},
    ::Type{T1},
) where {F, T1 <: Tuple}
    UnrolledUtilities.unrolled_map(params_as_tuple(T1)) do param1
        rmaptype(fn, param1, param1)
    end
end

"""
    rmap(fn, X...)

Recursively apply `fn` to each element of `X`
"""
rmap(fn::F, X) where {F} = fn(X)
rmap(fn::F, X::Tuple{}) where {F} = ()
@inline function rmap(fn::F, X::T) where {F, T <: Tuple}
    UnrolledUtilities.unrolled_map(X) do x
        rmap(fn, x)
    end
end
rmap(fn::F, X::NT) where {F, NT <: NamedTuple} =
    NamedTuple{nt_names(X)}(rmap(fn, Tuple(X)))

rmap(fn::F, X, Y) where {F} = fn(X, Y)
function rmap(fn::F, X::Tuple, Y::Tuple) where {F}
    UnrolledUtilities.unrolled_map(X, Y) do x, y
        rmap(fn, x, y)
    end
end

function rmap(fn::F, X, Y::Tuple) where {F}
    UnrolledUtilities.unrolled_map(Y) do y
        rmap(fn, X, y)
    end
end

function rmap(fn::F, X::Tuple, Y) where {F}
    UnrolledUtilities.unrolled_map(X) do x
        rmap(fn, x, Y)
    end
end

function rmap(fn::F, X::NamedTuple, Y::NamedTuple) where {F}
    @assert nt_names(X) === nt_names(Y)
    return NamedTuple{nt_names(X)}(rmap(fn, Tuple(X), Tuple(Y)))
end
rmap(fn::F, X::NamedTuple, Y) where {F} =
    NamedTuple{nt_names(X)}(rmap(fn, Tuple(X), Y))
rmap(fn::F, X::NamedTuple, Y::Tuple) where {F} =
    NamedTuple{nt_names(X)}(rmap(fn, Tuple(X), Y))
rmap(fn::F, X::NamedTuple, Y::Tuple{}) where {F} =
    NamedTuple{nt_names(X)}(rmap(fn, Tuple(X)))
rmap(fn::F, X, Y::NamedTuple) where {F} =
    NamedTuple{nt_names(Y)}(rmap(fn, X, Tuple(Y)))
rmap(fn::F, X::Tuple, Y::NamedTuple) where {F} =
    NamedTuple{nt_names(Y)}(rmap(fn, X, Tuple(Y)))
rmap(fn::F, X::Tuple{}, Y::NamedTuple) where {F} =
    NamedTuple{nt_names(Y)}(rmap(fn, Tuple(Y)))

nt_names(::NamedTuple{names}) where {names} = names

rmin(X, Y) = rmap(min, X, Y)
rmax(X, Y) = rmap(max, X, Y)


"""
    rmaptype(fn, T)
    rmaptype(fn, T1, T2)

Recursively apply `fn` to each type parameter of the type `T`, or to each type
parameter of the types `T1` and `T2`, where `fn` returns a type.
"""
rmaptype(fn::F, ::Type{T}) where {F, T} = fn(T)
rmaptype(fn::F, ::Type{T}) where {F, T <: Tuple} =
    Tuple{rmaptype_Tuple(fn, T)...}
rmaptype(fn::F, ::Type{T}) where {F, names, Tup, T <: NamedTuple{names, Tup}} =
    NamedTuple{names, rmaptype(fn, Tup)}

rmaptype(fn::F, ::Type{T1}, ::Type{T2}) where {F, T1, T2} = fn(T1, T2)
rmaptype(fn::F, ::Type{T1}, ::Type{T2}) where {F, T1 <: Tuple, T2 <: Tuple} =
    Tuple{rmaptype_Tuple(fn, T1, T2)...}
rmaptype(
    fn::F,
    ::Type{T1},
    ::Type{T2},
) where {
    F,
    names,
    Tup1,
    Tup2,
    T1 <: NamedTuple{names, Tup1},
    T2 <: NamedTuple{names, Tup2},
} = NamedTuple{names, rmaptype(fn, Tup1, Tup2)}

"""
    rpromote_type(Ts...)

Recursively apply `promote_type` to the input types.
"""
rpromote_type(Ts...) = reduce((T1, T2) -> rmaptype(promote_type, T1, T2), Ts)
rpromote_type() = Union{}

"""
    rzero(X)

Recursively zero out each element of `X`.
"""
rzero(X) = rzero(typeof(X))
rzero(::Type{T}) where {T} = zero(T)
rzero(::Type{T}) where {E, T <: Tuple{E}} = (rzero(E),)
function rzero(::Type{T}) where {T <: Tuple}
    UnrolledUtilities.unrolled_map(params_as_tuple(T)) do param
        rzero(param)
    end
end


rzero(::Type{Tup}) where {names, T, Tup <: NamedTuple{names, T}} =
    NamedTuple{names}(rzero(T))

"""
    rconvert(T, X)

Identical to `convert(T, X)`, but with improved type stability for nested types.
"""
rconvert(::Type{T}, X::T) where {T} = X
rconvert(::Type{T}, X) where {T} =
    rmap((zero_value, x) -> convert(typeof(zero_value), x), rzero(T), X)
# TODO: Remove this function once Julia's default convert function is
# type-stable for nested Tuple/NamedTuple types.

"""
    rmul(X, Y)
    X ⊠ Y

Recursively scale each element of `X` by `Y`.
"""
rmul(X, Y) = rmap(*, X, Y)
const ⊠ = rmul

"""
    radd(X, Y)
    X ⊞ Y

Recursively add elements of `X` and `Y`.
"""
radd(X) = X
radd(X, Y) = rmap(+, X, Y)
const ⊞ = radd

# Adapted from Base/operators.jl for general nary operator fallbacks
for op in (:rmul, :radd)
    @eval begin
        ($op)(a, b, c, xs...) = Base.afoldl($op, ($op)(($op)(a, b), c), xs...)
    end
end

"""
    rsub(X, Y)
    X ⊟ Y

Recursively subtract elements of `Y` from `X`.
"""
rsub(X) = rmap(-, X)
rsub(X, Y) = rmap(-, X, Y)
const ⊟ = rsub

"""
    rdiv(X, Y)

Recursively divide each element of `X` by `Y`
"""
rdiv(X, Y) = rmap(/, X, Y)

"""
    rmuladd(w, X, Y)

Recursively add elements of `w * X + Y`.
"""
rmuladd(w::Number, X, Y) = rmap((x, y) -> muladd(w, x, y), X, Y)
rmuladd(X, w::Number, Y) = rmap((x, y) -> muladd(x, w, y), X, Y)
rmuladd(x::Number, w::Number, Y) = rmap(y -> muladd(x, w, y), Y)

end # module
