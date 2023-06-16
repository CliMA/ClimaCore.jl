"""
    RecursiveApply

This module contains operators to recurse over nested `Tuple`s or `NamedTuple`s.

To extend to another type `T`, define `RecursiveApply.rmap(fn, args::T...)`
"""
module RecursiveApply

export ⊞, ⊠, ⊟

# These functions need to be generated for type stability (since T.parameters is
# a SimpleVector, the compiler cannot always infer its size and elements).
first_param(::Type{T}) where {T} =
    if @generated
        :($(first(T.parameters)))
    else
        first(T.parameters)
    end
tail_params(::Type{T}) where {T} =
    if @generated
        :($(Tuple{Base.tail((T.parameters...,))...}))
    else
        Tuple{Base.tail((T.parameters...,))...}
    end

# This is a type-stable version of map(x -> rmap(fn, x), X) or
# map((x, y) -> rmap(fn, x, y), X, Y).
rmap_tuple(fn::F, X) where {F} =
    isempty(X) ? () : (rmap(fn, first(X)), rmap_tuple(fn, Base.tail(X))...)
rmap_tuple(fn::F, X, Y) where {F} =
    isempty(X) || isempty(Y) ? () :
    (
        rmap(fn, first(X), first(Y)),
        rmap_tuple(fn, Base.tail(X), Base.tail(Y))...,
    )

# This is a type-stable version of map(T′ -> rfunc(fn, T′), T.parameters) or
# map((T1′, T2′) -> rfunc(fn, T1′, T2′), T1.parameters, T2.parameters), where
# rfunc can be either rmaptype or rmap_type2value.
rmap_Tuple(rfunc::R, fn::F, ::Type{Tuple{}}) where {R, F, T} = ()
rmap_Tuple(rfunc::R, fn::F, ::Type{T}) where {R, F, T <: Tuple} =
    (rfunc(fn, first_param(T)), rmap_Tuple(rfunc, fn, tail_params(T))...)

rmap_Tuple(_, _, ::Type{Tuple{}}, ::Type{T}) where {T <: Tuple} = ()
rmap_Tuple(_, _, ::Type{T}, ::Type{Tuple{}}) where {T <: Tuple} = ()

rmap_Tuple(
    rfunc::R,
    fn::F,
    ::Type{T1},
    ::Type{T2},
) where {R, F, T1 <: Tuple, T2 <: Tuple} = (
    rfunc(fn, first_param(T1), first_param(T2)),
    rmap_Tuple(rfunc, fn, tail_params(T1), tail_params(T2))...,
)

"""
    rmap(fn, X...)

Recursively apply `fn` to each element of `X`
"""
rmap(fn::F, X) where {F} = fn(X)
rmap(fn::F, X, Y) where {F} = fn(X, Y)
rmap(fn::F, X::Tuple) where {F} = rmap_tuple(fn, X)
rmap(fn::F, X::Tuple, Y::Tuple) where {F} = rmap_tuple(fn, X, Y)
rmap(fn::F, X::NamedTuple{names}) where {F, names} =
    NamedTuple{names}(rmap(fn, Tuple(X)))
rmap(fn::F, X::NamedTuple{names}, Y::NamedTuple{names}) where {F, names} =
    NamedTuple{names}(rmap(fn, Tuple(X), Tuple(Y)))


rmin(X, Y) = rmap(min, X, Y)
rmax(X, Y) = rmap(max, X, Y)


"""
    rmaptype(fn, T)
    rmaptype(fn, T1, T2)

Recursively apply `fn` to each type parameter of the type `T`, or to each type
parameter of the types `T1` and `T2`, where `fn` returns a type.
"""
rmaptype(fn::F, ::Type{T}) where {F, T} = fn(T)
rmaptype(fn::F, ::Type{T1}, ::Type{T2}) where {F, T1, T2} = fn(T1, T2)
rmaptype(fn::F, ::Type{T}) where {F, T <: Tuple} =
    Tuple{rmap_Tuple(rmaptype, fn, T)...}
rmaptype(fn::F, ::Type{T1}, ::Type{T2}) where {F, T1 <: Tuple, T2 <: Tuple} =
    Tuple{rmap_Tuple(rmaptype, fn, T1, T2)...}
rmaptype(fn::F, ::Type{T}) where {F, names, Tup, T <: NamedTuple{names, Tup}} =
    NamedTuple{names, rmaptype(fn, Tup)}
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
    rmap_type2value(fn, T)

Recursively apply `fn` to each type parameter of the type `T`, where `fn`
returns a value instead of a type.
"""
rmap_type2value(fn::F, ::Type{T}) where {F, T} = fn(T)
rmap_type2value(fn::F, ::Type{T}) where {F, T <: Tuple} =
    rmap_Tuple(rmap_type2value, fn, T)
rmap_type2value(
    fn::F,
    ::Type{T},
) where {F, names, Tup, T <: NamedTuple{names, Tup}} =
    NamedTuple{names}(rmap_type2value(fn, Tup))

"""
    rzero(T)

Recursively compute the zero value of type `T`.
"""
rzero(::Type{T}) where {T} = rmap_type2value(zero, T)

"""
    rmul(X, Y)
    X ⊠ Y

Recursively scale each element of `X` by `Y`.
"""
rmul(X, Y) = rmap(*, X, Y)
rmul(w::Number, X) = rmap(x -> w * x, X)
rmul(X, w::Number) = rmap(x -> x * w, X)
rmul(w1::Number, w2::Number) = w1 * w2
const ⊠ = rmul

"""
    radd(X, Y)
    X ⊞ Y

Recursively add elements of `X` and `Y`.
"""
radd(X) = X
radd(X, Y) = rmap(+, X, Y)
radd(w::Number, X) = rmap(x -> w + x, X)
radd(X, w::Number) = rmap(x -> x + w, X)
radd(w1::Number, w2::Number) = w1 + w2
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
rsub(X, w::Number) = rmap(x -> x - w, X)
rsub(w::Number, X) = rmap(x -> w - x, X)
rsub(w1::Number, w2::Number) = w1 - w2
const ⊟ = rsub

"""
    rdiv(X, Y)

Recursively divide each element of `X` by `Y`
"""
rdiv(X, Y) = rmap(/, X, Y)
rdiv(X, w::Number) = rmap(x -> x / w, X)
rdiv(w::Number, X) = rmap(x -> w / x, X)
rdiv(w1::Number, w2::Number) = w1 / w2

"""
    rmuladd(w, X, Y)

Recursively add elements of `w * X + Y`.
"""
rmuladd(w::Number, X, Y) = rmap((x, y) -> muladd(w, x, y), X, Y)
rmuladd(X, w::Number, Y) = rmap((x, y) -> muladd(x, w, y), X, Y)
rmuladd(X::Number, w::Number, Y) = rmap((x, y) -> muladd(x, w, y), X, Y)
rmuladd(w::Number, x::Number, y::Number) = muladd(w, x, y)

"""
    rmatmul1(W, S, i, j)

Recursive matrix product along the 1st dimension of `S`. Equivalent to:

    mapreduce(⊠, ⊞, W[i,:], S[:,j])

"""
function rmatmul1(W, S, i, j)
    Nq = size(W, 2)
    @inbounds r = W[i, 1] ⊠ S[1, j]
    @inbounds for ii in 2:Nq
        r = rmuladd(W[i, ii], S[ii, j], r)
    end
    return r
end

"""
    rmatmul2(W, S, i, j)

Recursive matrix product along the 2nd dimension `S`. Equivalent to:

    mapreduce(⊠, ⊞, W[j,:], S[i, :])

"""
function rmatmul2(W, S, i, j)
    Nq = size(W, 2)
    @inbounds r = W[j, 1] ⊠ S[i, 1]
    @inbounds for jj in 2:Nq
        r = rmuladd(W[j, jj], S[i, jj], r)
    end
    return r
end

end # module
